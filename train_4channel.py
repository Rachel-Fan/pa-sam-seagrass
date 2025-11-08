#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PA-SAM GGB (GLCM entropy replaces red channel) training & evaluation script.

Changes in this version:
- Read file lists from /splits/all_train.txt and /splits/all-eval.txt (or user-specified).
- Build name_im_gt_list *explicitly from splits* so dataset paths exactly match your lists.
- Keep the "GLCM replaces red channel" logic inside utils.dataloader.OnlineDataset (expects im_ch4_path).
- Other training/eval behaviors match train_3channel.py (DDP, logging, checkpoints, metrics, merge SAM+PA).

Usage (example):
torchrun --nproc_per_node=1 train_ggb.py \
  --im_dir ./data/2025/All/image \
  --gt_dir ./data/2025/All/index \
  --im_ch4_dir ./data/2025/All/glcm \
  --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \
  --output ./output/2025/All/ggb/train_run1 \
  --split_train ./data/2025/All/splits/all_train.txt \
  --split_eval  ./data/2025/All/splits/all-eval.txt \
  --device cuda --batch_size_train 4 --batch_size_valid 1 --input_size 512 512
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


# -------------------------
# Args
# -------------------------
def get_args_parser():
    parser = argparse.ArgumentParser("PA-SAM GGB", add_help=False)

    # I/O & paths
    parser.add_argument("--output", type=str, required=True, help="Directory for logs, masks, checkpoints")
    parser.add_argument("--logfile", type=str, default=None, help="Path to save the log file")

    parser.add_argument("--im_dir", type=str, required=True, help="RGB image root dir")
    parser.add_argument("--gt_dir", type=str, required=True, help="GT mask root dir")
    parser.add_argument("--im_ch4_dir", type=str, required=True, help="GLCM (single-channel) root dir")

    parser.add_argument("--im_ext", type=str, default=".png")
    parser.add_argument("--gt_ext", type=str, default=".png")

    parser.add_argument("--split_train", type=str, default="./data/2025/All/splits/all_train.txt",
                        help="Split list for training")
    parser.add_argument("--split_eval", type=str, default="./data/2025/All/splits/all-eval.txt",
                        help="Split list for validation")
    parser.add_argument("--split_test", type=str, default=None,
                        help="Optional split list for test set (if provided, will build a test loader)")

    parser.add_argument("--model_type", type=str, default="vit_l", choices=["vit_h","vit_l","vit_b"], help="SAM type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to base SAM checkpoint (.pth) or merged (sam_pa_*.pth) for eval")
    parser.add_argument("--restore_model", type=str, default=None, help="Path to PA-decoder checkpoint to restore (epoch_*.pth or best_model.pth)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="Device")

    # train/eval options
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
    parser.add_argument("--eval", action="store_true", help="evaluation only (run on val split)")
    parser.add_argument("--visualize", action="store_true", help="dump visualization during eval")
    parser.add_argument("--avg_valid_only", action="store_true", help="average IoU only over valid (non-empty union) samples")
    parser.add_argument("--eval_thr", type=float, default=0.0, help="threshold on logits when binarizing predictions")

    return parser


# -------------------------
# Logging
# -------------------------
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


# -------------------------
# Split helpers
# -------------------------
def _read_split_file(split_path: str) -> List[str]:
    """
    Read split file; each line may be:
      - 'basename' (no ext)  -> we'll append ext later
      - 'basename.png'       -> we'll strip ext to unify
    Returns list of basenames (no extension).
    """
    bases = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            b = os.path.splitext(os.path.basename(t))[0]
            bases.append(b)
    return bases


def _build_name_im_gt_list_from_splits(
    name: str,
    im_dir: str,
    gt_dir: str,
    im_ch4_dir: str,
    im_ext: str,
    gt_ext: str,
    split_path: str,
    print_fn
) -> List[Dict]:
    """
    Build a single 'dataset item' (the entry used by utils.dataloader.OnlineDataset)
    from a split list. This produces a list of length 1:
      [{"dataset_name": name,
        "im_path": [...],
        "gt_path": [...],
        "im_ext": im_ext,
        "gt_ext": gt_ext,
        "im_ch4_path": [...]}]
    Missing files are skipped with warnings.
    """
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    bases = _read_split_file(split_path)
    im_paths, gt_paths, ch4_paths = [], [], []
    miss_cnt = 0

    for b in bases:
        ip = os.path.join(im_dir,    b + im_ext)
        gp = os.path.join(gt_dir,    b + gt_ext)
        cp = os.path.join(im_ch4_dir, b + im_ext)  # ch4 follows image ext by convention

        ok = True
        if not os.path.isfile(ip):
            print_fn(f"[WARN] missing image: {ip}")
            ok = False
        if not os.path.isfile(gp):
            print_fn(f"[WARN] missing GT:    {gp}")
            ok = False
        if not os.path.isfile(cp):
            print_fn(f"[WARN] missing GLCM:  {cp}")
            ok = False

        if ok:
            im_paths.append(ip)
            gt_paths.append(gp)
            ch4_paths.append(cp)
        else:
            miss_cnt += 1

    print_fn(f"[SPLIT] {name}: total {len(bases)} listed, usable {len(im_paths)}, skipped {miss_cnt}")

    return [{
        "dataset_name": name,
        "im_path": im_paths,
        "gt_path": gt_paths,
        "im_ext": im_ext,
        "gt_ext": gt_ext,
        "im_ch4_path": ch4_paths
    }]


def _build_dataloaders_from_splits(args, print_fn):
    """
    Build train/valid/test dataloaders strictly from split files.
    Requires utils.dataloader.OnlineDataset that supports 'im_ch4_path' and
    replaces RED with GLCM internally.
    """
    # ---- Train ----
    train_im_gt_list = _build_name_im_gt_list_from_splits(
        name="train",
        im_dir=args.im_dir,
        gt_dir=args.gt_dir,
        im_ch4_dir=args.im_ch4_dir,
        im_ext=args.im_ext,
        gt_ext=args.gt_ext,
        split_path=args.split_train,
        print_fn=print_fn
    )
    train_loader, _ = create_dataloaders(
        train_im_gt_list,
        my_transforms=[RandomHFlip(), LargeScaleJitter()],
        batch_size=args.batch_size_train,
        training=True
    )

    # ---- Valid ----
    valid_im_gt_list = _build_name_im_gt_list_from_splits(
        name="valid",
        im_dir=args.im_dir,
        gt_dir=args.gt_dir,
        im_ch4_dir=args.im_ch4_dir,
        im_ext=args.im_ext,
        gt_ext=args.gt_ext,
        split_path=args.split_eval,
        print_fn=print_fn
    )
    valid_loaders, _ = create_dataloaders(
        valid_im_gt_list,
        my_transforms=[Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False
    )
    valid_loader = valid_loaders[0]

    # ---- Test (optional) ----
    test_loader = None
    if args.split_test:
        test_im_gt_list = _build_name_im_gt_list_from_splits(
            name="test",
            im_dir=args.im_dir,
            gt_dir=args.gt_dir,
            im_ch4_dir=args.im_ch4_dir,
            im_ext=args.im_ext,
            gt_ext=args.gt_ext,
            split_path=args.split_test,
            print_fn=print_fn
        )
        test_loaders, _ = create_dataloaders(
            test_im_gt_list,
            my_transforms=[Resize(args.input_size)],
            batch_size=args.batch_size_valid,
            training=False
        )
        test_loader = test_loaders[0]

    return train_loader, valid_loader, test_loader


# -------------------------
# SAM forward helper for PA
# -------------------------
@torch.no_grad()
def _sam_forward_for_pa(sam_ddp, batched_input):
    """
    Call SAM's forward_for_prompt_adapter, return embeddings for PA decoder.
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


# -------------------------
# Train / Eval loops
# -------------------------
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
            ik = "box"  # deterministic for eval
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

        enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = _sam_forward_for_pa(sam_ddp, batched_input)

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

        # extract logits
        logits = None
        if isinstance(out, (list, tuple)) and len(out) > 0:
            for idx in [3, 5, 4, 0]:  # final -> refined -> coarse -> masks_sam
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

        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        pred_bin = (logits > args.eval_thr).to(dtype=torch.bool)
        gt_bin   = (labels > 0.5).to(dtype=torch.bool)

        B = pred_bin.shape[0]
        for b in range(B):
            p = pred_bin[b, 0]
            g = gt_bin[b, 0]
            total_iou  += float(iou_per_image(p, g))
            total_biou += float(boundary_iou_per_image(p, g))
            num_images += 1

        if (step % 100 == 0) or (step == len(valid_loader) - 1):
            mean_iou_so_far = total_iou / max(1, num_images)
            mean_biou_so_far = total_biou / max(1, num_images)
            print_fn(f"[{step:5d}/{len(valid_loader)}]  val_iou_0: {mean_iou_so_far:.4f}  val_boundary_iou_0: {mean_biou_so_far:.4f}")

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


# -------------------------
# Main worker
# -------------------------
def main_worker(args):
    misc.init_distributed_mode(args)
    print_fn = _setup_logging(args)

    print_fn(f"world_size:{args.world_size} rank:{args.rank} local_rank:{args.local_rank}")
    print_fn(f"args: {args}\n")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # ---- Model (PA decoder) ----
    net = MaskDecoderPA(args.model_type)
    if torch.cuda.is_available() and args.device == "cuda":
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.gpu] if args.device=="cuda" else None,
        find_unused_parameters=args.find_unused_params
    )
    net_without_ddp = net.module

    # ---- DataLoaders from SPLITS (GLCM replaces red channel handled in OnlineDataset) ----
    train_loader, valid_loader, test_loader = _build_dataloaders_from_splits(args, print_fn)

    # ---- SAM backbone ----
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(
        sam, device_ids=[args.gpu] if args.device=="cuda" else None,
        find_unused_parameters=args.find_unused_params
    )

    # ---- (Optional) restore PA decoder ----
    if args.restore_model:
        print_fn("restore model from:", args.restore_model)
        state = torch.load(args.restore_model, map_location="cpu")
        net_without_ddp.load_state_dict(state, strict=False)

    # ---- Evaluation-only ----
    if args.eval:
        stats = evaluate(args, net, sam, valid_loader, visualize=args.visualize, print_fn=print_fn)
        return

    # ---- Training ----
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

        # DDP epoch set (if sampler available)
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "sampler") and hasattr(train_loader.batch_sampler.sampler, "set_epoch"):
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, net, sam, optimizer, train_loader, print_fn)
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
