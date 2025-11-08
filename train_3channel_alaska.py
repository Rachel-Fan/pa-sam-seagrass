import os
import argparse
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

from segment_anything_training import sam_model_registry
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.losses import loss_masks_whole, loss_masks_whole_uncertain, loss_uncertain
from utils.function import show_anns
import utils.misc as misc

from model.mask_decoder_pa import MaskDecoderPA

import logging
import warnings
warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('PA-SAM 3-Channel', add_help=False)

    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--logfile", type=str, default=None, help="Optional log file path")
    parser.add_argument("--model-type", type=str, default="vit_l", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to official SAM checkpoint to build SAM")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=21, type=int)
    parser.add_argument('--input_size', nargs=2, type=int, default=[512, 512], metavar=("H", "W"))
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=4, type=int)

    # DDP
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str, help="Path to PA decoder-only (epoch_XX.pth) or merged (sam_pa_*.pth) for eval")

    # Averaging choice
    parser.add_argument('--avg_valid_only', action='store_true', help="Average IoU only over samples with union>0 (skip empty-GT)")

    # Single-source dirs (no train/valid/test subfolders)
    parser.add_argument("--im_dir", type=str, default="./data/2025/Alaska/image")
    parser.add_argument("--gt_dir", type=str, default="./data/2025/Alaska/index")
    parser.add_argument("--im_ch4_dir", type=str, default="./data/2025/Alaska/glcm")

    return parser.parse_args()


def _ensure_logger(args, default_name):
    if not args.logfile:
        args.logfile = os.path.join(args.output, default_name)
    os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
    try:
        if os.path.exists(args.logfile):
            os.remove(args.logfile)
    except Exception:
        pass
    logging.basicConfig(filename=args.logfile, level=logging.INFO)


def _split_7_2_1_full_list(args):
    """
    从单一三目录(im_dir / gt_dir / im_ch4_dir)读取全量样本列表，
    然后按 7:2:1 切分 train / valid / test，并把划分保存为 txt 以便复刻。
    返回: train_list, valid_list, test_list
    """
    dataset_all = [{
        "name": "ALL",
        "im_dir": args.im_dir,
        "gt_dir": args.gt_dir,
        "im_ext": ".png",
        "gt_ext": ".png",
        "im_ch4_dir": args.im_ch4_dir,
    }]
    # 用 flag="train" 只是复用扫描逻辑
    full_list = get_im_gt_name_dict(dataset_all, flag="train")

    # 按文件名排序 + 固定随机种子打乱 => 可复刻
    idxs = list(range(len(full_list)))
    idxs.sort(key=lambda i: full_list[i].get("im_name", ""))
    rng = random.Random(args.seed)
    rng.shuffle(idxs)

    n = len(full_list)
    n_train = int(round(n * 0.7))
    n_valid = int(round(n * 0.2))
    n_test  = max(0, n - n_train - n_valid)  # 剩余给 test，避免因四舍五入导致总数不等

    train_idx = idxs[:n_train]
    valid_idx = idxs[n_train:n_train + n_valid]
    test_idx  = idxs[n_train + n_valid:n_train + n_valid + n_test]

    train_list = [full_list[i] for i in train_idx]
    valid_list = [full_list[i] for i in valid_idx]
    test_list  = [full_list[i] for i in test_idx]

    # 保存 splits
    split_dir = os.path.join(args.output, "splits")
    os.makedirs(split_dir, exist_ok=True)

    def dump_txt(path, L):
        with open(path, "w") as f:
            for item in L:
                f.write(item.get("im_name", "") + "\n")

    dump_txt(os.path.join(split_dir, "train.txt"), train_list)
    dump_txt(os.path.join(split_dir, "valid.txt"), valid_list)
    dump_txt(os.path.join(split_dir, "test.txt"),  test_list)

    print(f"[Split] total={n}  train={len(train_list)}  valid={len(valid_list)}  test={len(test_list)}  (seed={args.seed})")
    print(f"[Split] saved -> {split_dir}/train.txt, valid.txt, test.txt")

    return train_list, valid_list, test_list


def main(net, args):

    misc.init_distributed_mode(args)
    print('world size:', args.world_size)
    print('rank:', args.rank)
    print('local_rank:', args.local_rank)
    print("args:", args, '\\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # === 从三目录读取并按 7:2:1 切分 ===
    train_im_gt_list, valid_im_gt_list, test_im_gt_list = _split_7_2_1_full_list(args)

    if not args.eval:
        print("--- create training dataloader ---")
        train_dataloaders, _ = create_dataloaders(
            train_im_gt_list,
            my_transforms=[RandomHFlip(), LargeScaleJitter()],
            batch_size=args.batch_size_train,
            training=True)
        print(len(train_dataloaders), " train dataloaders created")
    else:
        train_dataloaders = None

    print("--- create valid dataloader ---")
    valid_dataloaders, _ = create_dataloaders(
        valid_im_gt_list,
        my_transforms=[Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    print("--- create test dataloader ---")
    test_dataloaders, _ = create_dataloaders(
        test_im_gt_list,
        my_transforms=[Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False)
    print(len(test_dataloaders), " test dataloaders created")

    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module

    if not args.eval:
        if misc.is_main_process():
            os.makedirs(args.output, exist_ok=True)
            _ensure_logger(args, 'train.log')

        # --- define optimizer ---
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch
        train_loop(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        # Build SAM from official checkpoint
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=args.device)
        sam = torch.nn.parallel.DistributedDataParallel(
            sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

        if args.restore_model:
            print("restore model from:", args.restore_model)
            state = torch.load(args.restore_model, map_location="cpu")
            # decoder-only or merged
            net_state = net_without_ddp.state_dict()
            loaded = 0
            for k in list(net_state.keys()):
                if k in state and net_state[k].shape == state[k].shape:
                    net_state[k] = state[k]; loaded += 1
                elif ('mask_decoder.'+k) in state and net_state[k].shape == state['mask_decoder.'+k].shape:
                    net_state[k] = state['mask_decoder.'+k]; loaded += 1
            try:
                net_without_ddp.load_state_dict(net_state, strict=False)
                print(f"[Eval] Loaded PA decoder params: {loaded}")
            except Exception as e:
                print(f"[Eval] Warning: could not load decoder state: {e}")

        # 这里仍然对 valid 做评估；若也想在 test 上评估，可再调 evaluate 一次
        evaluate(args, net, sam, valid_dataloaders, args.visualize, print_func=print)
        # 想一起评估 test：取消下面两行注释
        # print("\n--- Evaluate on TEST ---")
        # evaluate(args, net, sam, test_dataloaders, args.visualize, print_func=print)


def train_loop(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
        _ensure_logger(args, 'train.log')

    def print(*a, **k):
        msg = ' '.join(str(x) for x in a)
        logging.info(msg)
        __builtins__.print(*a, **k)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    best_val_iou = -1.0
    best_model_path = None
    last_model_name = None

    net.train()
    _ = net.to(device=args.device)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(
        sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

    for epoch in range(epoch_start, epoch_num):
        print("epoch:", epoch, "  learning rate:", optimizer.param_groups[0]["lr"])
        os.environ["CURRENT_EPOCH"] = str(epoch)
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders, 20, logger=args.logfile, print_func=print):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()

            input_keys = ['box','point','noise_mask','box+point','box+noise_mask','point+noise_mask','box+point+noise_mask']
            labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
            try:
                labels_points = misc.masks_sample_points(labels[:, 0, :, :])
            except Exception:
                input_keys = ['box', 'noise_mask', 'box+noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                di = {}
                input_image = torch.as_tensor(imgs[b_i].astype(np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                di['image'] = input_image
                input_type = random.choice(input_keys)
                if 'box' in input_type:
                    di['boxes'] = labels_box[b_i:b_i+1]
                elif 'point' in input_type:
                    point_coords = labels_points[b_i:b_i+1]
                    di['point_coords'] = point_coords
                    di['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
                elif 'noise_mask' in input_type:
                    di['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                di['original_size'] = imgs[b_i].shape[:2]
                di['label'] = labels[b_i:b_i+1]
                batched_input.append(di)

            with torch.no_grad():
                batched_output, interm_embeddings = sam.module.forward_for_prompt_adapter(batched_input, multimask_output=False)

            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(batch_len)], dim=0)
            image_pe = [batched_output[i]['image_pe'] for i in range(batch_len)]
            sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(batch_len)]
            dense_embeddings = [batched_output[i]['dense_embeddings'] for i in range(batch_len)]
            image_record = [batched_output[i]['image_record'] for i in range(batch_len)]
            input_images = batched_output[0]['input_images']

            masks_sam, iou_preds, uncertain_maps, final_masks, coarse_masks, refined_masks, box_preds = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                interm_embeddings=interm_embeddings,
                image_record=image_record,
                prompt_encoder=sam.module.prompt_encoder,
                input_images=input_images
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

        print("Finished epoch:     ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        lr_scheduler.step()

        test_stats = evaluate(args, net, sam, valid_dataloaders, visualize=False, print_func=print)

        if epoch % args.model_save_fre == 0:
            last_model_name = f"/epoch_{epoch}.pth"
            save_path = os.path.join(args.output, last_model_name.strip('/'))
            print("Saving regular checkpoint:", save_path)
            misc.save_on_master(net.module.state_dict(), save_path)

        val_iou_keys = [k for k in test_stats.keys() if "val_iou" in k]
        if val_iou_keys:
            vals = []
            for k in val_iou_keys:
                v = test_stats[k]
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(float(getattr(v, "item", lambda: v)()))
            avg_val_iou = float(np.mean(vals))
        else:
            avg_val_iou = float('nan')

        if (not math.isnan(avg_val_iou)) and (avg_val_iou > best_val_iou):
            best_val_iou = avg_val_iou
            best_model_path = os.path.join(args.output, "best_model.pth")
            print(f"🌟 New best model (IoU={avg_val_iou:.4f}) — saving to {best_model_path}")
            misc.save_on_master(net.module.state_dict(), best_model_path)
        else:
            print(f"No improvement (IoU={avg_val_iou}, best={best_val_iou})")

    print("Training Reaches The Maximum Epoch Number")


def evaluate(args, net, sam, valid_dataloaders, visualize=False, print_func=print):
    print = print_func

    if args.eval and not args.visualize:
        _ensure_logger(args, 'eval.log')

    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        early_n = int(os.environ.get("EVAL_EARLY", "0"))
        seen = 0

        for data_val in metric_logger.log_every(valid_dataloader, 1000, logger=args.logfile, print_func=print):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = \
                data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()

            labels_box = misc.masks_to_boxes(labels_val[:, 0, :, :])  # boxes in resized space
            batched_input = []
            for b_i in range(len(imgs)):
                di = {}
                input_image = torch.as_tensor(imgs[b_i].astype(np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                di['image'] = input_image
                di['boxes'] = labels_box[b_i:b_i+1]
                di['original_size'] = imgs[b_i].shape[:2]
                di['label'] = data_val['label'][b_i:b_i+1]
                batched_input.append(di)

            with torch.no_grad():
                batched_output, interm_embeddings = sam.module.forward_for_prompt_adapter(batched_input, multimask_output=False)

            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(batch_len)], dim=0)
            image_pe = [batched_output[i]['image_pe'] for i in range(batch_len)]
            sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(batch_len)]
            dense_embeddings = [batched_output[i]['dense_embeddings'] for i in range(batch_len)]
            image_record = [batched_output[i]['image_record'] for i in range(batch_len)]
            input_images = batched_output[0]['input_images']

            masks_sam, iou_preds, uncertain_maps, final_masks, coarse_masks, refined_masks, box_preds = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                interm_embeddings=interm_embeddings,
                image_record=image_record,
                prompt_encoder=sam.module.prompt_encoder,
                input_images=input_images
            )

            # === Evaluate against labels_val (resized labels) ===
            labels_target = labels_val
            labels_bin = (labels_target > 0).to(torch.uint8)

            pred_logits_up = F.interpolate(
                masks_sam, size=labels_bin.shape[-2:], mode="bilinear", align_corners=False
            )
            pred_bin = (pred_logits_up > 0).to(torch.uint8)

            inter = (pred_bin & labels_bin).sum(dim=(1, 2, 3)).float()
            union = (pred_bin | labels_bin).sum(dim=(1, 2, 3)).float()

            if args.avg_valid_only:
                valid = union > 0
                iou_b = torch.zeros_like(union)
                iou_b[valid] = inter[valid] / union[valid]
                iou = (iou_b[valid].mean() if valid.any() else torch.tensor(0.0, device=iou_b.device))
            else:
                iou = (inter / union.clamp_min(1)).mean()

            boundary_iou = iou  # placeholder

            print(f"DEBUG pixel count (resized GT): pred={int(pred_bin.sum())} gt={int(labels_bin.sum())}")
            print("DEBUG logits: min={:.3f} max={:.3f} mean={:.3f}".format(
                pred_logits_up.min().item(), pred_logits_up.max().item(), pred_logits_up.mean().item()
            ))
            print(f"DEBUG iou_now={float(iou):.4f}")

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)

            if visualize:
                os.makedirs(args.output, exist_ok=True)
                masks_pa_vis = (pred_bin > 0).cpu()
                for ii in range(len(imgs)):
                    ori_im_path = data_val['ori_im_path'][ii]
                    ori_image_name = ori_im_path.split('/')[-1]
                    save_base = os.path.join(args.output, ori_image_name)
                    imgs_ii = imgs[ii].astype(np.uint8)
                    show_anns(masks_pa_vis[ii], None, None, None, save_base, imgs_ii, torch.tensor([iou.item()]), torch.tensor([boundary_iou.item()]))

    print('============================')
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {kk: meter.global_avg for kk, meter in metric_logger.meters.items() if meter.count > 0}
    test_stats = resstat
    return test_stats


if __name__ == "__main__":
    args = get_args_parser()
    net = MaskDecoderPA(args.model_type)
    main(net, args)
