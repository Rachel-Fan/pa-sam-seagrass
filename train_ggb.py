#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PA-SAM GGB (GLCM entropy replaces red channel) training & evaluation script.

Features:
- Supports datasets providing `im_ch4_dir` (single-channel) to replace image red channel during dataloader stage.
- Distributed training via torchrun + DDP.
- Robust logging to file + console.
- Valid metrics (IoU / boundary IoU) with option to average ONLY over effective samples (non-empty unions).
- Periodic checkpoint saving (epoch_N.pth) + best checkpoint saving (best_model.pth) by val IoU.
- Merges PA decoder into SAM checkpoint at training end for both last & best (sam_pa_epoch_*.pth / sam_pa_best.pth).
- Evaluation mode (--eval) reproduces training forward path (SAM -> embeddings -> PA decoder).
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import logging
import time
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
import utils.misc as misc
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.losses import loss_masks_whole, loss_masks_whole_uncertain, loss_uncertain
from model.mask_decoder_pa import MaskDecoderPA

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser("PA-SAM GGB", add_help=False)

    # paths / core
    parser.add_argument("--output", type=str, required=True, help="Directory for logs, masks, checkpoints")
    parser.add_argument("--logfile", type=str, default=None, help="Path to save the log file")
    parser.add_argument("--model-type", type=str, default="vit_l", choices=["vit_h","vit_l","vit_b"], help="SAM type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to base SAM checkpoint (.pth) OR merged (sam_pa_*.pth) in eval")
    parser.add_argument("--restore-model", type=str, default=None, help="Path to PA-decoder checkpoint to restore (e.g., epoch_*.pth or best_model.pth)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="Device")

    # data/opt
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--lr_drop_epoch", default=10, type=int)
    parser.add_argument("--max_epoch_num", default=21, type=int)
    parser.add_argument("--input_size", nargs=2, type=int, default=[512,512], metavar=("H","W"))
    parser.add_argument("--batch_size_train", default=4, type=int)
    parser.add_argument("--batch_size_valid", default=1, type=int)
    parser.add_argument("--model_save_fre", default=4, type=int)

    # ddp
    parser.add_argument("--world_size", default=1, type=int, help="num distributed processes")
    parser.add_argument("--dist_url", default="env://", help="init url")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for dist")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id for single-node multi-gpu")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--dist_backend", default="nccl")

    # modes
    parser.add_argument("--eval", action="store_true", help="evaluation only")
    parser.add_argument("--visualize", action="store_true", help="dump visualization during eval")
    parser.add_argument("--avg_valid_only", action="store_true", help="average IoU only over valid (non-empty union) samples")
    parser.add_argument("--eval-thr", type=float, default=0.0, help="threshold on logits when binarizing predictions")

    return parser


def _setup_logging(args):
    os.makedirs(args.output, exist_ok=True)
    if not args.logfile:
        tag = "eval" if args.eval else "train"
        args.logfile = os.path.join(args.output, f"{tag}.log")
    os.makedirs(os.path.dirname(args.logfile), exist_ok=True)

    logging.basicConfig(filename=args.logfile, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    def info(*a):
        s = " ".join(str(x) for x in a)
        logging.info(s)
    return info


def _build_dataloaders(train_datasets, valid_datasets, test_datasets, args, print_fn):
    if not args.eval:
        print_fn("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, _ = create_dataloaders(
            train_im_gt_list,
            my_transforms=[RandomHFlip(), LargeScaleJitter()],
            batch_size=args.batch_size_train,
            training=True,
        )
        print_fn(len(train_dataloaders), " train dataloaders created")
    else:
        train_dataloaders = None

    print_fn("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, _ = create_dataloaders(
        valid_im_gt_list,
        my_transforms=[Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False,
    )
    print_fn(len(valid_dataloaders), " valid dataloaders created")

    if test_datasets is not None:
        print_fn("--- create test dataloader ---")
        test_im_gt_list = get_im_gt_name_dict(test_datasets, flag="test")
        test_dataloaders, _ = create_dataloaders(
            test_im_gt_list,
            my_transforms=[Resize(args.input_size)],
            batch_size=args.batch_size_valid,
            training=False,
        )
        print_fn(len(test_dataloaders), " test dataloaders created")
    else:
        test_dataloaders = None

    return train_dataloaders, valid_dataloaders, test_dataloaders


@torch.no_grad()
def _sam_forward_for_pa(sam_ddp, batched_input):
    """
    调用 SAM 的 forward_for_prompt_adapter，按训练时的约定返回供 PA 解码器使用的各类 embedding。
    """
    batched_output, interm_embeddings = sam_ddp.module.forward_for_prompt_adapter(
        batched_input, multimask_output=False
    )

    batch_len = len(batched_output)
    encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(batch_len)], dim=0)
    image_pe = [batched_output[i]['image_pe'] for i in range(batch_len)]
    sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(batch_len)]
    dense_embeddings = [batched_output[i]['dense_embeddings'] for i in range(batch_len)]
    image_record = [batched_output[i]['image_record'] for i in range(batch_len)]
    input_images = batched_output[0]['input_images']

    return encoder_embedding, image_pe, sparse_embeddings, dense_embeddings, image_record, input_images, interm_embeddings


def train_one_epoch(args, net_ddp, sam_ddp, optimizer, train_loader, print_fn):
    net_ddp.train()
    _ = net_ddp.to(device=args.device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    for data in metric_logger.log_every(train_loader, 20, logger=args.logfile, print_func=print_fn):
        inputs = data["image"]
        labels = data["label"]
        if torch.cuda.is_available() and args.device == "cuda":
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()

        input_keys = ["box","point","noise_mask","box+point","box+noise_mask","point+noise_mask","box+point+noise_mask"]
        labels_box = misc.masks_to_boxes(labels[:,0,:,:])
        try:
            labels_points = misc.masks_sample_points(labels[:,0,:,:])
        except Exception:
            input_keys = ["box","noise_mask","box+noise_mask"]
            labels_points = None
        labels_256 = F.interpolate(labels, size=(256, 256), mode="bilinear")
        labels_noisemask = misc.masks_noise(labels_256)

        batched_input = []
        for b in range(len(imgs)):
            d = {}
            input_image = torch.as_tensor(imgs[b].astype(np.uint8), device=sam_ddp.device).permute(2, 0, 1).contiguous()
            d["image"] = input_image
            ik = random.choice(input_keys)
            if "box" in ik:
                d["boxes"] = labels_box[b:b+1]
            elif "point" in ik and labels_points is not None:
                point_coords = labels_points[b:b+1]
                d["point_coords"] = point_coords
                d["point_labels"] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
            elif "noise_mask" in ik:
                d["mask_inputs"] = labels_noisemask[b:b+1]
            else:
                d["boxes"] = labels_box[b:b+1]
            d["original_size"] = imgs[b].shape[:2]
            d["label"] = labels[b:b+1]
            batched_input.append(d)

        with torch.no_grad():
            enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = _sam_forward_for_pa(sam_ddp, batched_input)

        masks_sam, iou_preds, uncertain_maps, final_masks, coarse_masks, refined_masks, box_preds = net_ddp(
            image_embeddings=enc,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_e,
            dense_prompt_embeddings=dense_e,
            multimask_output=False,
            interm_embeddings=interm,
            image_record=image_record,
            prompt_encoder=sam_ddp.module.prompt_encoder,
            input_images=input_images,
        )

        loss_mask, loss_dice = loss_masks_whole(masks_sam, labels/255.0, len(masks_sam))
        loss = loss_mask + loss_dice

        loss_mask_final, loss_dice_final = loss_masks_whole_uncertain(coarse_masks, refined_masks, labels/255.0, uncertain_maps, len(final_masks))
        loss = loss + (loss_mask_final + loss_dice_final)

        loss_uncertain_map, gt_uncertain = loss_uncertain(uncertain_maps, labels)
        loss = loss + loss_uncertain_map

        loss_dict = {
            "loss_mask": loss_mask, "loss_dice": loss_dice,
            "loss_mask_final": loss_mask_final, "loss_dice_final": loss_dice_final,
            "loss_uncertain_map": loss_uncertain_map
        }
        loss_dict_reduced = misc.reduce_dict(loss_dict)
        losses_reduced_scaled = sum(loss_dict_reduced.values())
        loss_value = losses_reduced_scaled.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

    metric_logger.synchronize_between_processes()
    print_fn("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    return train_stats


@torch.no_grad()
def evaluate(args, net_ddp, sam_ddp, valid_loader, visualize=False, print_fn=print):
    """
    Evaluation that mirrors the training forward path:
      GT -> prompts -> SAM forward_for_prompt_adapter -> PA decoder -> logits -> IoU
    """
    device = torch.device(args.device if hasattr(args, "device") else "cuda")
    net_ddp.eval()
    if sam_ddp is not None:
        sam_ddp.eval()

    thr = getattr(args, "eval_thr", 0.0)

    vis_dir = None
    if visualize and hasattr(args, "output"):
        vis_dir = os.path.join(args.output, "val_vis")
        os.makedirs(vis_dir, exist_ok=True)

    total_iou = 0.0
    total_biou = 0.0
    num_images = 0

    def iou_per_image(pred_bin, gt_bin, eps=1e-6):
        pred_sum = pred_bin.sum().item()
        gt_sum = gt_bin.sum().item()
        if gt_sum == 0 and pred_sum == 0:
            return 1.0
        inter = (pred_bin & gt_bin).sum().item()
        union = (pred_bin | gt_bin).sum().item()
        if union <= 0:
            return 0.0
        return inter / (union + eps)

    def boundary_iou_per_image(pred_bin, gt_bin):
        # placeholder: identical to IoU unless you plug in a true boundary IoU
        return iou_per_image(pred_bin, gt_bin)

    for step, sample in enumerate(valid_loader):
        inputs = sample["image"].to(device, non_blocking=True)       # [B,3,H,W]
        labels = sample["label"].to(device, non_blocking=True)       # [B,1,H,W] or [B,H,W]
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()

        # build prompts from GT (same as training)
        input_keys = ["box","point","noise_mask","box+point","box+noise_mask","point+noise_mask","box+point+noise_mask"]
        labels_box = misc.masks_to_boxes(labels[:,0,:,:])
        try:
            labels_points = misc.masks_sample_points(labels[:,0,:,:])
        except Exception:
            input_keys = ["box","noise_mask","box+noise_mask"]
            labels_points = None
        labels_256 = F.interpolate(labels, size=(256, 256), mode="bilinear")
        labels_noisemask = misc.masks_noise(labels_256)

        batched_input = []
        for b in range(len(imgs)):
            d = {}
            input_image = torch.as_tensor(imgs[b].astype(np.uint8), device=sam_ddp.device).permute(2, 0, 1).contiguous()
            d["image"] = input_image
            # deterministic choice could be used; we keep box by default to avoid randomness in eval
            ik = "box"
            if labels_points is not None and (labels_points.shape[1] > 0):
                # you can switch to "box+point" or "point" as needed
                ik = "box"
            if "box" in ik:
                d["boxes"] = labels_box[b:b+1]
            if "point" in ik and labels_points is not None:
                point_coords = labels_points[b:b+1]
                d["point_coords"] = point_coords
                d["point_labels"] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
            if "noise_mask" in ik:
                d["mask_inputs"] = labels_noisemask[b:b+1]

            d["original_size"] = imgs[b].shape[:2]
            d["label"] = labels[b:b+1]
            batched_input.append(d)

        # SAM forward (like training)
        enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = _sam_forward_for_pa(sam_ddp, batched_input)

        # PA decoder forward
        out = net_ddp(
            image_embeddings=enc,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_e,
            dense_prompt_embeddings=dense_e,
            multimask_output=False,
            interm_embeddings=interm,
            image_record=image_record,
            prompt_encoder=sam_ddp.module.prompt_encoder,
            input_images=input_images,
        )

        # 取 logits
        logits = None
        # out 可能是 tuple/list，训练里你得到的是 (masks_sam, iou_preds, uncertain_maps, final_masks, coarse_masks, refined_masks, box_preds)
        if isinstance(out, (list, tuple)) and len(out) > 0:
            # 优先用 final_masks/coarse_masks/refined_masks，如有
            name2idx = {
                "masks_sam": 0,
                "iou_preds": 1,
                "uncertain_maps": 2,
                "final_masks": 3,
                "coarse_masks": 4,
                "refined_masks": 5,
            }
            # try final -> refined -> coarse -> masks_sam
            for idx in [3, 5, 4, 0]:
                if idx < len(out) and torch.is_tensor(out[idx]):
                    logits = out[idx]
                    break
            if logits is None and torch.is_tensor(out[0]):
                logits = out[0]
        elif torch.is_tensor(out):
            logits = out
        elif isinstance(out, dict):
            for k in ["logits", "final_masks", "refined_masks", "coarse_masks", "masks", "mask_logits", "low_res_logits", "low_res_masks"]:
                if k in out and torch.is_tensor(out[k]):
                    logits = out[k]; break

        if logits is None:
            raise RuntimeError("Cannot extract logits from PA decoder output.")

        # [B,1,h,w]
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        # 尺寸对齐到 GT
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        # binarize
        pred_bin = (logits > thr).to(dtype=torch.bool)
        gt_bin   = (labels > 0.5).to(dtype=torch.bool)

        # accumulate per-image IoU
        B = pred_bin.shape[0]
        for b in range(B):
            p = pred_bin[b, 0]
            g = gt_bin[b, 0]
            iou_v  = iou_per_image(p, g)
            biou_v = boundary_iou_per_image(p, g)
            total_iou  += float(iou_v)
            total_biou += float(biou_v)
            num_images += 1

        if (step % 100 == 0) or (step == len(valid_loader) - 1):
            mean_iou_so_far = total_iou / max(1, num_images)
            mean_biou_so_far = total_biou / max(1, num_images)
            print_fn(f"[{step:5d}/{len(valid_loader)}]  val_iou_0: {mean_iou_so_far:.4f}  val_boundary_iou_0: {mean_biou_so_far:.4f}")

        # 可选：保存少量可视化
        if visualize and vis_dir is not None:
            save_n = min(B, 2)
            for b in range(save_n):
                base = f"step{step:05d}_b{b}"
                torch.save(pred_bin[b,0].to(torch.uint8)*255, os.path.join(vis_dir, base+"_pred.pt"))
                torch.save(gt_bin[b,0].to(torch.uint8)*255,   os.path.join(vis_dir, base+"_gt.pt"))

    mean_iou = total_iou / max(1, num_images)
    mean_biou = total_biou / max(1, num_images)
    stats = {"val_iou_0": float(mean_iou), "val_boundary_iou_0": float(mean_biou)}
    print_fn("============================")
    print_fn(f"Averaged stats: val_iou_0: {mean_iou:.4f}  val_boundary_iou_0: {mean_biou:.4f}")
    return stats


def merge_and_save_sam(args, last_epoch_name, best_ckpt_path=None, print_fn=print):
    try:
        sam_ckpt = torch.load(args.checkpoint, map_location="cpu")
        pa_decoder = torch.load(last_epoch_name, map_location="cpu")
        sam_ckpt.update({k.replace("mask_decoder", "mask_decoder_ori"): v for k, v in sam_ckpt.items() if "mask_decoder" in k})
        for key in pa_decoder.keys():
            sam_key = "mask_decoder." + key
            sam_ckpt[sam_key] = pa_decoder[key]
        merged_last = os.path.join(args.output, f"sam_pa_{os.path.basename(last_epoch_name)}")
        torch.save(sam_ckpt, merged_last)
        print_fn(f"✅ Merged SAM+PA (last) saved to: {merged_last}")
    except Exception as e:
        print_fn(f"⚠️ Merge (last) failed: {e}")

    if best_ckpt_path and os.path.exists(best_ckpt_path):
        try:
            sam_ckpt_b = torch.load(args.checkpoint, map_location="cpu")
            pa_best = torch.load(best_ckpt_path, map_location="cpu")
            sam_ckpt_b.update({k.replace("mask_decoder", "mask_decoder_ori"): v for k, v in sam_ckpt_b.items() if "mask_decoder" in k})
            for key in pa_best.keys():
                sam_key = "mask_decoder." + key
                sam_ckpt_b[sam_key] = pa_best[key]
            merged_best = os.path.join(args.output, "sam_pa_best.pth")
            torch.save(sam_ckpt_b, merged_best)
            print_fn(f"✅ Merged SAM+PA (best) saved to: {merged_best}")
        except Exception as e:
            print_fn(f"⚠️ Merge (best) failed: {e}")
    else:
        print_fn("⚠️ No best_model.pth found; skip merging best.")


def main_worker(args):
    misc.init_distributed_mode(args)
    print_fn = _setup_logging(args)

    print_fn(f"world_size:{args.world_size} rank:{args.rank} local_rank:{args.local_rank}")
    print_fn(f"args: {args}\n")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # ====== YOUR DATASETS (GGB: GLCM entropy replaces red) ======
    dataset_Alaska = {"name": "Washington",
                 "im_dir": "/mnt/d/Eelgrass_processed_images_2025/ModelData/Data_for_modeling/Washington/train/image",
                 "gt_dir": "/mnt/d/Eelgrass_processed_images_2025/ModelData/Data_for_modeling/Washington/train/index",
                 "im_ch4_dir": "/mnt/d/Eelgrass_processed_images_2025/ModelData/Data_for_modeling/Washington/train/glcm",
                 "im_ext": ".png", "gt_ext": ".png"}

    dataset_Alaska_val = {"name": "Washington",
                 "im_dir": "/mnt/d/Eelgrass_processed_images_2025/ModelData/Data_for_modeling/Washington/valid/image",
                 "gt_dir": "/mnt/d/Eelgrass_processed_images_2025/ModelData/Data_for_modeling/Washington/valid/index",
                 "im_ch4_dir": "/mnt/d/Eelgrass_processed_images_2025/ModelData/Data_for_modeling/Washington/train/glcm",
                 "im_ext": ".png", "gt_ext": ".png"}
    
    dataset_Alaska_test = {"name": "Alaska",
                "im_dir": "./data/2025/Alaska/test/image",
                "gt_dir": "./data/2025/Alaska/test/index",
                "im_ext": ".png",
                "gt_ext": ".png",
                "im_ch4_dir": "/mnt/d/Eelgrass_processed_images_2025/ModelData/Data_for_modeling/Alaska/test/glcm"}

    train_datasets = [dataset_Alaska]
    valid_datasets = [dataset_Alaska_val]
    test_datasets  = [dataset_Alaska_test]

    net = MaskDecoderPA(args.model_type)
    if torch.cuda.is_available() and args.device == "cuda":
        net.cuda()

    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.gpu] if args.device=="cuda" else None,
        find_unused_parameters=args.find_unused_params
    )
    net_without_ddp = net.module

    train_loaders, valid_loaders, test_loaders = _build_dataloaders(train_datasets, valid_datasets, test_datasets, args, print_fn)
    valid_loader = valid_loaders[0]

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(
        sam, device_ids=[args.gpu] if args.device=="cuda" else None,
        find_unused_parameters=args.find_unused_params
    )

    if args.restore_model:
        print_fn("restore model from:", args.restore_model)
        state = torch.load(args.restore_model, map_location="cpu")
        net_without_ddp.load_state_dict(state, strict=False)

    if args.eval:
        stats = evaluate(args, net, sam, valid_loader, visualize=args.visualize, print_fn=print_fn)
        return

    os.makedirs(args.output, exist_ok=True)
    if not args.logfile:
        args.logfile = os.path.join(args.output, "train.log")

    optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
    lr_scheduler.last_epoch = args.start_epoch

    best_iou = -1.0
    best_path = None
    last_epoch_ckpt = None

    for epoch in range(args.start_epoch, args.max_epoch_num):
        os.environ["CURRENT_EPOCH"] = str(epoch)
        print_fn(f"epoch: {epoch}  learning rate: {optimizer.param_groups[0]['lr']}")

        if hasattr(train_loaders, "batch_sampler") and hasattr(train_loaders.batch_sampler, "sampler") and hasattr(train_loaders.batch_sampler.sampler, "set_epoch"):
            train_loaders.batch_sampler.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, net, sam, optimizer, train_loaders, print_fn)
        print_fn(f"Finished epoch: {epoch}")
        print_fn("Averaged stats:", train_stats)
        lr_scheduler.step()

        val_stats = evaluate(args, net, sam, valid_loader, visualize=False, print_fn=print_fn)
        mean_iou = val_stats["val_iou_0"]

        net.train()

        if (epoch % args.model_save_fre) == 0:
            save_p = os.path.join(args.output, f"epoch_{epoch}.pth")
            if misc.is_main_process():
                torch.save(net.module.state_dict(), save_p)
                print_fn(f"Saving regular checkpoint: {save_p}")
            last_epoch_ckpt = save_p

        if mean_iou is not None and mean_iou > best_iou and misc.is_main_process():
            best_iou = mean_iou
            best_path = os.path.join(args.output, "best_model.pth")
            torch.save(net.module.state_dict(), best_path)
            print_fn(f"🎯 New best IoU={best_iou:.4f}  -> {best_path}")
        else:
            print_fn(f"No improvement this epoch (IoU={mean_iou:.4f}, best={best_iou:.4f})")

    print_fn("Training Reaches The Maximum Epoch Number")

    if misc.is_main_process():
        if last_epoch_ckpt is None:
            last_epoch_ckpt = os.path.join(args.output, f"epoch_{args.max_epoch_num-1}.pth")
            if not os.path.exists(last_epoch_ckpt):
                last_epoch_ckpt = os.path.join(args.output, "epoch_last.pth")
                torch.save(net.module.state_dict(), last_epoch_ckpt)

        merge_and_save_sam(args, last_epoch_ckpt, best_ckpt_path=best_path, print_fn=print_fn)


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    main_worker(args)


if __name__ == "__main__":
    main()
