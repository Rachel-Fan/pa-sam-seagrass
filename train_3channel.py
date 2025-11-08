#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PA-SAM Training (RGB or GGB switchable) WITHOUT modifying utils/dataloader.py.

- Uses your existing OnlineDataset, but builds DataLoaders here to pass replace_red_with_glcm flag.
- Reads split lists (all_train/valid/test.txt) that contain ONLY filenames (one per line).
- RGB mode (default): --use_ggb OFF -> don't replace red, still pass ch4 paths (dataloader compatibility).
- GGB mode:           --use_ggb ON  -> replace red channel by GLCM.

Data layout:
  --im_dir      ./data/2025/All/image
  --gt_dir      ./data/2025/All/index
  --im_ch4_dir  ./data/2025/All/glcm
  --train_list  ./data/2025/All/splits/all_train.txt
  --valid_list  ./data/2025/All/splits/all_valid.txt
  --test_list   ./data/2025/All/splits/all_test.txt
"""

import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms


from segment_anything_training import sam_model_registry
import utils.misc as misc
from utils.losses import loss_masks_whole, loss_masks_whole_uncertain, loss_uncertain
from model.mask_decoder_pa import MaskDecoderPA

# 直接复用你现有的 OnlineDataset 和变换
from utils.dataloader import OnlineDataset, RandomHFlip, Resize, LargeScaleJitter

import warnings
warnings.filterwarnings("ignore")


# =========================
# Args
# =========================
def get_args_parser():
    p = argparse.ArgumentParser("PA-SAM RGB/GGB", add_help=False)

    # core
    p.add_argument("--output", type=str, required=True, help="Directory for logs/checkpoints")
    p.add_argument("--logfile", type=str, default=None, help="Log file path")
    p.add_argument("--model-type", type=str, default="vit_l", choices=["vit_h","vit_l","vit_b"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint (.pth)")
    p.add_argument("--restore-model", type=str, default=None, help="PA-decoder checkpoint to resume (epoch_*.pth/best_model.pth)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])

    # data roots + splits
    p.add_argument("--im_dir", type=str, default="./data/2025/All/image")
    p.add_argument("--gt_dir", type=str, default="./data/2025/All/index")
    p.add_argument("--im_ch4_dir", type=str, default="./data/2025/All/glcm")
    p.add_argument("--train_list", type=str, default="./data/2025/All/splits/all_train.txt")
    p.add_argument("--valid_list", type=str, default="./data/2025/All/splits/all_valid.txt")
    p.add_argument("--test_list",  type=str, default="./data/2025/All/splits/all_test.txt")

    # switch: RGB or GGB
    p.add_argument("--use_ggb", action="store_true",
                   help="If set, replace RED with GLCM (GGB). Otherwise use pure RGB.")
    
    # —— 三步法开关 ——
    p.add_argument("--fg-sampler", action="store_true",
                   help="Use foreground-weighted sampler + auto pos_weight for BCE.")
    p.add_argument("--use-focal", action="store_true",
                   help="Use Focal instead of BCE in the combo loss.")
    p.add_argument("--lam-edge", type=float, default=2.0,
                   help="Edge weight lambda for boundary emphasis.")
    p.add_argument("--criterion", type=str, default="pa",
                   choices=["pa","bce_dice"],
                   help="'pa' = your current losses; 'bce_dice' = BCE(or Focal)+Dice on PA logits.")
    p.add_argument("--sanity", action="store_true",
                   help="Run 8-image overfit sanity check before training.")


    # opt
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--learning_rate", default=1e-3, type=float)
    p.add_argument("--start_epoch", default=0, type=int)
    p.add_argument("--lr_drop_epoch", default=10, type=int)
    p.add_argument("--max_epoch_num", default=11, type=int)
    p.add_argument("--input_size", nargs=2, type=int, default=[512,512], metavar=("H","W"))
    p.add_argument("--batch_size_train", default=4, type=int)
    p.add_argument("--batch_size_valid", default=1, type=int)
    p.add_argument("--model_save_fre", default=4, type=int)

    # ddp
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--dist_url", default="env://")
    p.add_argument("--rank", default=0, type=int)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--find_unused_params", action="store_true")
    p.add_argument("--dist_backend", default="nccl")

    # modes
    p.add_argument("--eval", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--eval-thr", type=float, default=0.0)

    return p


# =========================
# Logging
# =========================
def _setup_logging(args):
    os.makedirs(args.output, exist_ok=True)
    if not args.logfile:
        args.logfile = os.path.join(args.output, ("eval" if args.eval else "train") + ".log")
    os.makedirs(os.path.dirname(args.logfile), exist_ok=True)

    logging.basicConfig(filename=args.logfile, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    def info(*a):
        s = " ".join(str(x) for x in a)
        logging.info(s)
    return info

def collate_fixed(batch):
    def _to_3ch(t):
        # t: [C,H,W] 或 [H,W]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        if t.dim() == 2:
            t = t.unsqueeze(0)                      # [1,H,W]
        C, H, W = t.shape
        # 统一到 3 通道：优先裁到前 3 通道；不足则扩展/复制
        if C == 3:
            return t.contiguous()
        if C > 3:
            return t[:3].contiguous()               # 丢弃多余通道（如 RGBA 的 A 或 第4通道）
        if C == 1:
            return t.repeat(3, 1, 1).contiguous()   # 单通道复制成 RGB
        if C == 2:
            return torch.cat([t, t[:1]], dim=0).contiguous()  # 2通道 -> 3通道
        # 其它奇怪情况：取前3个或pad到3个
        if C == 0:
            raise RuntimeError("Image tensor has 0 channels.")
        return (t[:3] if C > 3 else torch.cat([t, t.new_zeros(3-C, H, W)], dim=0)).contiguous()

    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]

        if k == "image":
            # 逐个样本统一到3通道再 stack
            fixed = []
            for v in vals:
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                # 允许 [H,W,C] 的稀有情况，转成 [C,H,W]
                if v.dim() == 3 and v.shape[-1] in (1,2,3,4) and v.shape[0] not in (1,2,3,4):
                    v = v.permute(2,0,1).contiguous()
                fixed.append(_to_3ch(v))
            # 再次校验尺寸一致
            H0, W0 = fixed[0].shape[-2], fixed[0].shape[-1]
            for i, f in enumerate(fixed):
                if f.shape[-2:] != (H0, W0):
                    raise RuntimeError(f"Found size-mismatch in batch for 'image': {f.shape} vs {(3,H0,W0)}")
            out[k] = torch.stack(fixed, dim=0)

        elif k == "label":
            # label 统一成 [1,H,W] 再 stack；阈值/归一化在后续 loss 中处理
            fixed = []
            for v in vals:
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                elif v.dim() == 3 and v.shape[0] != 1:  # 多通道 mask 取第1通道
                    v = v[:1]
                fixed.append(v.contiguous())
            H0, W0 = fixed[0].shape[-2], fixed[0].shape[-1]
            for f in fixed:
                if f.shape[-2:] != (H0, W0):
                    raise RuntimeError(f"Found size-mismatch in batch for 'label': {f.shape} vs {(1,H0,W0)}")
            out[k] = torch.stack(fixed, dim=0)

        elif k in ("shape",):
            try:
                out[k] = torch.stack([torch.as_tensor(v) for v in vals], dim=0)
            except Exception:
                out[k] = vals

        elif k in ("ori_label", "ori_im", "ori_im_path", "im_path", "gt_path", "name", "imidx"):
            out[k] = vals  # 可变/字符串信息保留列表

        else:
            out[k] = vals

    return out


# =========================
# Split reading helpers
# =========================
def _read_filenames(list_path):
    """Read one-filename-per-line list."""
    files = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            fn = line.strip()
            if fn:
                files.append(fn)
    return files

def _build_name_im_gt_entry(name, im_dir, gt_dir, ch4_dir, im_ext, gt_ext, filenames, use_glcm = False):
    """
    Build the single dict entry expected by OnlineDataset via get_im_gt_name_dict:
      keys: dataset_name, im_path, gt_path, im_ext, gt_ext, im_ch4_path
    """
    im_paths  = [os.path.join(im_dir,  fn) for fn in filenames]
    gt_paths  = [os.path.join(gt_dir,  fn) for fn in filenames]

    if ch4_dir:
        ch4_paths = [os.path.join(ch4_dir, fn) for fn in filenames]
    else:
        ch4_paths = []  # <- important

    return {
        "dataset_name": name,
        "im_path": im_paths,
        "gt_path": gt_paths,
        "im_ext": im_ext,
        "gt_ext": gt_ext,
        "im_ch4_path": ch4_paths,   # <- always present
    }


# =========================
# SAM forward packer
# =========================
@torch.no_grad()
@torch.no_grad()
def _sam_forward_for_pa(sam_ddp, batched_input):
    """
    Call SAM.forward_for_prompt_adapter and pack required outputs for PA decoder.
    """
    sam_mod = sam_ddp.module if hasattr(sam_ddp, "module") else sam_ddp
    batched_output, interm_embeddings = sam_mod.forward_for_prompt_adapter(
        batched_input, multimask_output=False
    )
    B = len(batched_output)
    enc = torch.cat([batched_output[i]['encoder_embedding'] for i in range(B)], dim=0)
    image_pe = [batched_output[i]['image_pe'] for i in range(B)]
    sparse_e = [batched_output[i]['sparse_embeddings'] for i in range(B)]
    dense_e  = [batched_output[i]['dense_embeddings']  for i in range(B)]
    image_record = [batched_output[i]['image_record'] for i in range(B)]
    input_images = batched_output[0]['input_images']
    return enc, image_pe, sparse_e, dense_e, image_record, input_images, interm_embeddings


# =========================
# DataLoader builders (no change to utils/dataloader.py)
# =========================

def _use_ddp():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

# 2) 更新带前景加权采样器，让它也用 collate_fn
def _build_fg_weighted_loader(dataset, batch_size, num_workers=4):
    has_fg, pos_weight = _scan_fg_stats(dataset)
    w_fg, w_bg = 1.0, 0.2
    weights = np.where(has_fg == 1, w_fg, w_bg).astype(np.float32)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fixed,          # ★ 关键
        pin_memory=False,
        persistent_workers=False if num_workers > 0 else False,
    )
    logging.info(f"[FG Sampler] with-foreground {has_fg.sum()}/{len(has_fg)} | pos_weight={pos_weight:.2f}")
    return loader, pos_weight

 
def build_loader_from_split(args, list_path, transforms, batch_size, training, replace_red_with_glcm):
    """
    Build a DataLoader using OnlineDataset directly, so we can pass replace_red_with_glcm switch.
    """
    filenames = _read_filenames(list_path)
    # 打印规模（自检）
    logging.info(f"[split] {os.path.basename(list_path)} -> {len(filenames)} samples")

    entry = _build_name_im_gt_entry(
        name="All",
        im_dir=args.im_dir,
        gt_dir=args.gt_dir,
        ch4_dir=args.im_ch4_dir,
        im_ext=".png",
        gt_ext=".png",
        filenames=filenames,
        use_glcm=replace_red_with_glcm
    )
    dataset = OnlineDataset(
        name_im_gt_list=[entry],
        transform=transforms,
        eval_ori_resolution=not training,
        replace_red_with_glcm=replace_red_with_glcm
    )

    # DDP-friendly sampler & loader（复刻你原先 create_dataloaders 的风格）
    if training:
        if getattr(args, "fg_sampler", False):
            num_workers = 2 if batch_size <= 4 else (4 if batch_size <= 8 else 8)
            loader, pos_weight = _build_fg_weighted_loader(dataset, batch_size=batch_size, num_workers=num_workers)
            dataset._pos_weight = pos_weight
            return loader, dataset
        else:
            num_workers = 2 if batch_size <= 4 else (4 if batch_size <= 8 else 8)
            if _use_ddp():
                sampler = DistributedSampler(dataset, shuffle=True)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    drop_last=True,
                    num_workers=num_workers,
                    collate_fn=collate_fixed,
                    pin_memory=False,
                    persistent_workers=False if num_workers > 0 else False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=num_workers,
                    collate_fn=collate_fixed,
                    pin_memory=False,
                    persistent_workers=False if num_workers > 0 else False,
                )
            dataset._pos_weight = 1.0
            return loader, dataset
    else:
        num_workers = 1
        if _use_ddp():
            sampler = DistributedSampler(dataset, shuffle=False)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
                num_workers=num_workers,
                collate_fn=collate_fixed, 
                pin_memory=False,
                persistent_workers=False,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                collate_fn=collate_fixed, 
                pin_memory=False,
                persistent_workers=False,
            )
        return loader, dataset

import numpy as np
from torch.utils.data import WeightedRandomSampler

def _scan_fg_stats(dataset):
    """遍历一次dataset，统计每个样本是否含前景 & 全局前景/背景像素数，用于采样权重与pos_weight。"""
    has_fg = np.zeros(len(dataset), dtype=np.uint8)
    fg_pixels, bg_pixels = 0, 0
    for i in range(len(dataset)):
        m = dataset[i]["label"]          # [1,H,W] uint8 0/255
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        m = m.squeeze()
        if m.max() > 1:
            m_bin = (m > 127).astype(np.uint8)
        else:
            m_bin = (m > 0).astype(np.uint8)
        s = int(m_bin.sum())
        has_fg[i] = 1 if s > 0 else 0
        fg_pixels += s
        bg_pixels += int(m_bin.size - s)
    pos_weight = max(1.0, bg_pixels / max(fg_pixels, 1))
    return has_fg, pos_weight



import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1); targets = targets.flatten(1)
        inter = (probs*targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1) + self.eps
        dice = (2*inter + self.eps) / denom
        return 1 - dice.mean()

def _make_edge_weight(targets, k=3, lam=2.0):
    # targets: [B,1,H,W] 0/1
    dil = F.max_pool2d(targets, kernel_size=k, stride=1, padding=k//2)
    ero = 1 - F.max_pool2d(1 - targets, kernel_size=k, stride=1, padding=k//2)
    edge = (dil - ero).clamp(0,1)
    return torch.ones_like(edge) + lam * edge

class WeightedBCEWithLogits(nn.Module):
    def __init__(self, pos_weight=None): super().__init__(); self.pos_weight = pos_weight
    def forward(self, logits, targets, pixel_weight=None):
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, dtype=torch.float32, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pw)
        if pixel_weight is not None: loss = loss * pixel_weight
        return loss.mean()

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0): super().__init__(); self.a=alpha; self.g=gamma
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = targets * p + (1 - targets) * (1 - p)
        return (self.a * (1 - p_t).pow(self.g) * ce).mean()

def build_combo_criterion(pos_weight=1.0, use_focal=False, lam_edge=2.0):
    bce = None if use_focal else WeightedBCEWithLogits(pos_weight=pos_weight)
    focal = BinaryFocalLoss(alpha=0.75, gamma=2.0) if use_focal else None
    dice = SoftDiceLoss()
    def _crit(logits, gts):
        # logits: [B,1,H,W]（未sigmoid）；gts: [B,1,H,W] uint8 0/255 或 float 0/1
        if gts.dtype != torch.float32: gts = gts.float()
        if gts.max() > 1: gts = (gts > 127).float()
        ew = _make_edge_weight(gts, k=3, lam=lam_edge)
        base = focal(logits, gts) if use_focal else bce(logits, gts, pixel_weight=ew)
        return 0.7*base + 0.3*dice(logits, gts)
    return _crit

@torch.no_grad()
def _batch_iou_from_logits(logits, gts):
    p = (logits > 0).float()                 # 与训练口径一致：logit>0 等价于 prob>0.5
    g = (gts > 0.5).float()
    inter = (p*g).sum(dim=(1,2,3))
    union = (p + g - p*g).sum(dim=(1,2,3)).clamp_min(1.0)
    return (inter/union).mean().item()

def sanity_overfit_8(args, pa_ddp, sam_ddp, train_loader, print_fn):
    # 取8张含前景样本
    idxs = []
    ds = train_loader.dataset
    for i in range(len(ds)):
        m = ds[i]["label"]
        if isinstance(m, torch.Tensor): m = m.cpu().numpy()
        if (m>0).sum() > 0:
            idxs.append(i)
        if len(idxs) >= 8: break
    if len(idxs) == 0:
        print_fn("[SANITY] No foreground samples found — skip sanity."); return

    sub = torch.utils.data.Subset(ds, idxs)
    loader = DataLoader(sub, batch_size=2, shuffle=True, num_workers=0)

    # 只做几百步小学习率快速拟合
    params = [p for p in pa_ddp.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    crit = build_combo_criterion(pos_weight=1.0, use_focal=False, lam_edge=args.lam_edge)

    pa_ddp.train()
    for it in range(300):
        for sample in loader:
            imgs_t = sample["image"].to(args.device, non_blocking=True)
            gts_t  = sample["label"].to(args.device, non_blocking=True)
            labels_box = misc.masks_to_boxes(gts_t[:,0,:,:])
            batched_input = []
            imgs_np = imgs_t.permute(0,2,3,1).cpu().numpy()
            for b in range(len(imgs_np)):
                d = {
                    "image": torch.as_tensor(imgs_np[b].astype(np.uint8), device=sam_ddp.device).permute(2,0,1).contiguous(),
                    "boxes": labels_box[b:b+1],
                    "original_size": imgs_np[b].shape[:2],
                    "label": gts_t[b:b+1],
                }
                batched_input.append(d)
            with torch.no_grad():
                enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = _sam_forward_for_pa(sam_ddp, batched_input)
            logits = pa_ddp(
                image_embeddings=enc, image_pe=image_pe,
                sparse_prompt_embeddings=sparse_e, dense_prompt_embeddings=dense_e,
                multimask_output=False, interm_embeddings=interm,
                image_record=image_record, 
                prompt_encoder=(sam_ddp.module.prompt_encoder if hasattr(sam_ddp, "module") else sam_ddp.prompt_encoder),

                input_images=input_images,
            )
            if isinstance(logits, (list,tuple)): logits = logits[0]
            if logits.dim()==3: logits = logits.unsqueeze(1)
            if logits.shape[-2:] != gts_t.shape[-2:]:
                logits = F.interpolate(logits, size=gts_t.shape[-2:], mode="bilinear", align_corners=False)

            loss = crit(logits, gts_t)
            opt.zero_grad(); loss.backward(); opt.step()

        if (it+1) % 50 == 0:
            iou = _batch_iou_from_logits(logits, gts_t)
            print_fn(f"[SANITY {it+1}/300] loss={loss.item():.4f}  iou={iou:.4f}")


# =========================
# Train / Eval
# =========================
def train_one_epoch(args, pa_ddp, sam_ddp, optimizer, train_loader, print_fn):
    pa_ddp.train(); _ = pa_ddp.to(device=args.device)
    metric_logger = misc.MetricLogger(delimiter="  ")

    # —— 组合损失（若启用）——
    combo_criterion = None
    pos_weight = getattr(train_loader.dataset, "_pos_weight", 1.0)
    if args.criterion == "bce_dice":
        combo_criterion = build_combo_criterion(
            pos_weight=pos_weight,
            use_focal=args.use_focal,
            lam_edge=args.lam_edge
        )
        print_fn(f"[Loss] Using BCE{'(Focal)' if args.use_focal else ''}+Dice with pos_weight={pos_weight:.2f}, lam_edge={args.lam_edge}")

    for sample in metric_logger.log_every(train_loader, 20, logger=args.logfile, print_func=print_fn):
        imgs_t = sample["image"].to(args.device, non_blocking=True)
        gts_t  = sample["label"].to(args.device, non_blocking=True)   # [B,1,H,W] 0/255

        imgs_np = imgs_t.permute(0,2,3,1).cpu().numpy()

        # —— SAM prompts（保持你现在逻辑）——
        input_keys = ["box","point","noise_mask","box+point","box+noise_mask","point+noise_mask","box+point+noise_mask"]
        labels_box = misc.masks_to_boxes(gts_t[:,0,:,:])
        try:
            labels_points = misc.masks_sample_points(gts_t[:,0,:,:])
        except Exception:
            input_keys = ["box","noise_mask","box+noise_mask"]
            labels_points = None
        labels_256 = F.interpolate(gts_t, size=(256,256), mode="bilinear")
        labels_noisemask = misc.masks_noise(labels_256)

        batched_input = []
        for b in range(len(imgs_np)):
            d = {}
            # input_image = torch.as_tensor(imgs_np[b].astype(np.uint8), device=sam_ddp.device).permute(2,0,1).contiguous()
            dev = imgs_t.device if 'imgs_t' in locals() else torch.device(args.device)
            input_image = torch.as_tensor(imgs_np[b].astype(np.uint8), device=dev).permute(2,0,1).contiguous()

            d["image"] = input_image
            ik = random.choice(input_keys)
            if "box" in ik: d["boxes"] = labels_box[b:b+1]
            elif "point" in ik and labels_points is not None:
                point_coords = labels_points[b:b+1]
                d["point_coords"] = point_coords
                d["point_labels"] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
            elif "noise_mask" in ik: d["mask_inputs"] = labels_noisemask[b:b+1]
            else: d["boxes"] = labels_box[b:b+1]
            d["original_size"] = imgs_np[b].shape[:2]
            d["label"] = gts_t[b:b+1]
            batched_input.append(d)

        with torch.no_grad():
            enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = _sam_forward_for_pa(sam_ddp, batched_input)

        out = pa_ddp(
            image_embeddings=enc,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_e,
            dense_prompt_embeddings=dense_e,
            multimask_output=False,
            interm_embeddings=interm,
            image_record=image_record,
            prompt_encoder=(sam_ddp.module.prompt_encoder if hasattr(sam_ddp, "module") else sam_ddp.prompt_encoder),

            input_images=input_images,
        )

        # —— 提取 logits（final/refined/coarse/masks_sam 任选其一，与你eval一致）——
        logits = None
        if isinstance(out, (list, tuple)) and len(out) > 0:
            for idx in [3,5,4,0]:
                if idx < len(out) and torch.is_tensor(out[idx]):
                    logits = out[idx]; break
            if logits is None and torch.is_tensor(out[0]): logits = out[0]
        elif torch.is_tensor(out):
            logits = out
        elif isinstance(out, dict):
            for k in ["logits","final_masks","refined_masks","coarse_masks","masks","mask_logits","low_res_logits","low_res_masks"]:
                if k in out and torch.is_tensor(out[k]): logits = out[k]; break
        if logits is None:
            raise RuntimeError("Cannot extract logits from PA output in training.")

        if logits.dim() == 3: logits = logits.unsqueeze(1)
        if logits.shape[-2:] != gts_t.shape[-2:]:
            logits = F.interpolate(logits, size=gts_t.shape[-2:], mode="bilinear", align_corners=False)

        # —— 两种 loss 口径：保持兼容 —— 
        if combo_criterion is None:
            # 你的原始 PA 损失口径（不改）
            loss_mask, loss_dice = loss_masks_whole(logits, gts_t/255.0, len(logits))
            loss = loss_mask + loss_dice
            loss_mask_final, loss_dice_final = loss_masks_whole_uncertain(None, logits, gts_t/255.0, None, len(logits))
            loss = loss + (loss_mask_final + loss_dice_final)
            loss_uncertain_map, _ = loss_uncertain(None, gts_t)
            loss = loss + loss_uncertain_map
            loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice,
                        "loss_mask_final": loss_mask_final, "loss_dice_final": loss_dice_final,
                        "loss_uncertain_map": loss_uncertain_map}
        else:
            # 我们的组合损失（BCE/Focal + Dice + 边界加权）
            loss = combo_criterion(logits, gts_t)
            loss_dict = {"loss_combo": loss}

        loss_dict_red = misc.reduce_dict(loss_dict)
        loss_val = sum(loss_dict_red.values()).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(training_loss=loss_val, **loss_dict_red)

    metric_logger.synchronize_between_processes()
    print_fn("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items() if m.count > 0}


@torch.no_grad()
def evaluate(args, pa_ddp, sam_ddp, valid_loader, print_fn=print):
    pa_ddp.eval()
    sam_ddp.eval()

    thr = args.eval_thr
    total_iou = 0.0
    total_biou = 0.0
    n = 0

    def iou_per_image(p, g, eps=1e-6):
        if g.sum().item() == 0 and p.sum().item() == 0:
            return 1.0
        inter = (p & g).sum().item()
        union = (p | g).sum().item()
        if union <= 0: return 0.0
        return inter / (union + eps)

    for sample in valid_loader:
        imgs_t = sample["image"].to(args.device, non_blocking=True)
        gts_t  = sample["label"].to(args.device, non_blocking=True)
        if gts_t.dim() == 3: gts_t = gts_t.unsqueeze(1)

        imgs_np = imgs_t.permute(0,2,3,1).cpu().numpy()
        labels_box = misc.masks_to_boxes(gts_t[:,0,:,:])

        batched_input = []
        for b in range(len(imgs_np)):
            d = {}
            input_image = torch.as_tensor(imgs_np[b].astype(np.uint8), device=sam_ddp.device).permute(2,0,1).contiguous()
            d["image"] = input_image
            d["boxes"] = labels_box[b:b+1]  # fixed prompt for eval
            d["original_size"] = imgs_np[b].shape[:2]
            d["label"] = gts_t[b:b+1]
            batched_input.append(d)

        enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = _sam_forward_for_pa(sam_ddp, batched_input)

        out = pa_ddp(
            image_embeddings=enc,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_e,
            dense_prompt_embeddings=dense_e,
            multimask_output=False,
            interm_embeddings=interm,
            image_record=image_record,
            prompt_encoder=(sam_ddp.module.prompt_encoder if hasattr(sam_ddp, "module") else sam_ddp.prompt_encoder),

            input_images=input_images,
        )

        # 取 logits（final -> refined -> coarse -> masks_sam）
        logits = None
        if isinstance(out, (list, tuple)) and len(out) > 0:
            for idx in [3,5,4,0]:
                if idx < len(out) and torch.is_tensor(out[idx]):
                    logits = out[idx]; break
            if logits is None and torch.is_tensor(out[0]):
                logits = out[0]
        elif torch.is_tensor(out):
            logits = out
        elif isinstance(out, dict):
            for k in ["logits","final_masks","refined_masks","coarse_masks","masks","mask_logits","low_res_logits","low_res_masks"]:
                if k in out and torch.is_tensor(out[k]):
                    logits = out[k]; break
        if logits is None:
            raise RuntimeError("Cannot extract logits from PA output.")

        if logits.dim() == 3: logits = logits.unsqueeze(1)
        if logits.shape[-2:] != gts_t.shape[-2:]:
            logits = F.interpolate(logits, size=gts_t.shape[-2:], mode="bilinear", align_corners=False)

        pred = (logits > thr).to(torch.bool)
        gt   = (gts_t > 0.5).to(torch.bool)

        B = pred.shape[0]
        for b in range(B):
            iou = iou_per_image(pred[b,0], gt[b,0])
            total_iou += float(iou)
            total_biou += float(iou)  # placeholder for boundary IoU
            n += 1

    mean_iou = total_iou / max(1,n)
    mean_biou = total_biou / max(1,n)
    print_fn(f"============================\nAveraged stats: val_iou_0: {mean_iou:.4f}  val_boundary_iou_0: {mean_biou:.4f}")
    return {"val_iou_0": mean_iou, "val_boundary_iou_0": mean_biou}

# 放在文件顶部或 main_worker 前
def maybe_wrap_ddp(module, device, gpu, find_unused=False):
    try:
        import torch.distributed as dist
        use_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    except Exception:
        use_ddp = False
    if use_ddp:
        return torch.nn.parallel.DistributedDataParallel(
            module, device_ids=[gpu] if str(device)=="cuda" else None,
            find_unused_parameters=find_unused
        )
    else:
        return module


# =========================
# Main
# =========================
def main_worker(args):
    misc.init_distributed_mode(args)
    print_fn = _setup_logging(args)

    print_fn(f"world_size:{args.world_size} rank:{args.rank} local_rank:{args.local_rank}")
    print_fn(f"args: {args}\n")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # ===== DataLoaders built here (split-aware) =====
    train_tfms = torch.nn.Sequential()  # placeholder
    valid_tfms = torch.nn.Sequential()

    # 用你原有的变换（与 create_dataloaders 一致）
    train_tfms = None
    valid_tfms = None
    # 训练：RandomHFlip + LargeScaleJitter
    train_transform = torch.nn.Sequential()  # not used; we'll pass Compose list directly via OnlineDataset
    train_transform = None
    train_loader, _ = build_loader_from_split(
        args,
        args.train_list,
        transforms = transforms.Compose([RandomHFlip(), LargeScaleJitter()]),
        batch_size = args.batch_size_train,
        training = True,
        replace_red_with_glcm = args.use_ggb
    )
    # 验证：Resize
    valid_loader, _ = build_loader_from_split(
        args,
        args.valid_list,
        transforms = transforms.Compose([Resize(args.input_size)]),
        batch_size = args.batch_size_valid,
        training = False,
        replace_red_with_glcm = args.use_ggb
    )
    # 可选：测试集（如评估需要）
    test_loader, _ = build_loader_from_split(
        args,
        args.test_list,
        transforms = transforms.Compose([Resize(args.input_size)]),
        batch_size = args.batch_size_valid,
        training = False,
        replace_red_with_glcm = args.use_ggb
    )

    # ===== Models =====
    '''
    pa = MaskDecoderPA(args.model_type)
    if torch.cuda.is_available() and args.device == "cuda":
        pa.cuda()
    pa = torch.nn.parallel.DistributedDataParallel(
        pa, device_ids=[args.gpu] if args.device=="cuda" else None,
        find_unused_parameters=args.find_unused_params
    )
    pa_wo_ddp = pa.module

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(
        sam, device_ids=[args.gpu] if args.device=="cuda" else None,
        find_unused_parameters=args.find_unused_params
    )
    '''
    
    pa = MaskDecoderPA(args.model_type).to(args.device)
    pa = maybe_wrap_ddp(pa, args.device, args.gpu, args.find_unused_params)
    pa_wo_ddp = pa.module if hasattr(pa, "module") else pa

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    sam = maybe_wrap_ddp(sam, args.device, args.gpu, args.find_unused_params)

    if args.sanity and (not args.eval):
        print_fn("[SANITY] 8-image overfit check starting…")
        sanity_overfit_8(args, pa, sam, train_loader, print_fn)
        print_fn("[SANITY] done.\n")


    # restore
    if args.restore_model:
        print_fn("restore model from:", args.restore_model)
        state = torch.load(args.restore_model, map_location="cpu")
        pa_wo_ddp.load_state_dict(state, strict=False)

    # eval only
    if args.eval:
        _ = evaluate(args, pa, sam, valid_loader, print_fn=print_fn)
        return

    # ===== Optim =====
    os.makedirs(args.output, exist_ok=True)
    if not args.logfile:
        args.logfile = os.path.join(args.output, "train.log")
    optimizer = optim.Adam(pa_wo_ddp.parameters(), lr=args.learning_rate, betas=(0.9,0.999), eps=1e-08, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
    lr_scheduler.last_epoch = args.start_epoch

    best_iou = -1.0
    best_path = None
    last_epoch_ckpt = None

    # ===== Train loop =====
    for epoch in range(args.start_epoch, args.max_epoch_num):
        os.environ["CURRENT_EPOCH"] = str(epoch)
        print_fn(f"epoch: {epoch}  learning rate: {optimizer.param_groups[0]['lr']}")

        # 让 DDP sampler 也按 epoch 变化
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, pa, sam, optimizer, train_loader, print_fn)
        print_fn(f"Finished epoch: {epoch}")
        print_fn("Averaged stats:", train_stats)
        lr_scheduler.step()

        val_stats = evaluate(args, pa, sam, valid_loader, print_fn=print_fn)
        mean_iou = val_stats["val_iou_0"]

        pa.train()

        # 常规保存
        if (epoch % args.model_save_fre) == 0:
            save_p = os.path.join(args.output, f"epoch_{epoch}.pth")
            if misc.is_main_process():
                torch.save(pa.module.state_dict(), save_p)
                print_fn(f"Saving regular checkpoint: {save_p}")
            last_epoch_ckpt = save_p

        # 最优
        if mean_iou is not None and mean_iou > best_iou and misc.is_main_process():
            best_iou = mean_iou
            best_path = os.path.join(args.output, "best_model.pth")
            torch.save(pa.module.state_dict(), best_path)
            print_fn(f"🎯 New best IoU={best_iou:.4f}  -> {best_path}")
        else:
            print_fn(f"No improvement this epoch (IoU={mean_iou:.4f}, best={best_iou:.4f})")

    print_fn("Training Reaches The Maximum Epoch Number")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    main_worker(args)


if __name__ == "__main__":
    main()
