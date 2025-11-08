#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PA-SAM (RGB only) — clean training script.
- No uncertain-map losses, no 4th channel/GLCM requirement.
- Reads split lists with one filename per line (相对/仅文件名均可).
- Uses OnlineDataset from utils.dataloader with replace_red_with_glcm=False.
- Loss: BCE(with logits) + Soft Dice (对 0/255 GT 友好).
"""

import os, random, argparse, numpy as np, torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from segment_anything_training import sam_model_registry
from model.mask_decoder_pa import MaskDecoderPA
import utils.misc as misc
from utils.dataloader import OnlineDataset, RandomHFlip, Resize, LargeScaleJitter
import torchvision.transforms as T


# -------------------------
# utils
# -------------------------
def read_list(p):
    files=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            # 只取文件名
            files.append(os.path.basename(s))
    return files

def build_entry(name, im_dir, gt_dir, filenames, im_ext=".png", gt_ext=".png"):
    # 规范化：如果 split 里是裸文件名，这里会拼出全路径；如果 split 里已经是全路径，也能处理
    def _norm_path(root, fn):
        return fn if os.path.isabs(fn) or os.path.splitext(fn)[1].lower() in [im_ext, gt_ext] and os.path.sep in fn \
               else os.path.join(root, fn)
    im_paths  = [_norm_path(im_dir, fn) for fn in filenames]
    gt_paths  = [_norm_path(gt_dir, fn) for fn in filenames]

    ch4_paths = list(im_paths)

    return {
        "dataset_name": name,
        "im_path": im_paths,
        "gt_path": gt_paths,
        "im_ext": im_ext,
        "gt_ext": gt_ext,
        "im_ch4_path": ch4_paths,
    }



def build_loader(args, list_path, transforms, batch_size, training: bool):
    names = read_list(list_path)
    print(f"[split] {os.path.basename(list_path)} -> {len(names)} samples")
    entry = build_entry("All", args.im_dir, args.gt_dir, names, im_ext=".png", gt_ext=".png")
    ds = OnlineDataset(
        name_im_gt_list=[entry],
        transform=transforms,
        eval_ori_resolution=not training,
        replace_red_with_glcm=False  # 只训 RGB
    )
    num_workers = 2 if batch_size<=4 else (4 if batch_size<=8 else 8)
    if training and misc.get_world_size()>1:
        sampler = DistributedSampler(ds, shuffle=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=num_workers)
    elif training:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    else:
        if misc.get_world_size()>1:
            sampler = DistributedSampler(ds, shuffle=False)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=1)
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    return loader

class SoftDiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, logits, targets):
        # targets: 0/1
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1); targets = targets.flatten(1)
        inter = (probs*targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1) + self.eps
        dice = (2*inter + self.eps) / denom
        return 1 - dice.mean()

def bce_dice_loss(logits, masks_0_255):
    gts = masks_0_255
    if gts.dtype != torch.float32: gts = gts.float()
    if gts.max() > 1: gts = (gts > 127).float()
    # 对正样本轻度增权（可调）
    pos = gts.mean().clamp_min(1e-6)
    neg = 1 - pos
    pos_weight = (neg / pos).clamp(1.0, 10.0)  # 上限避免过度
    bce = F.binary_cross_entropy_with_logits(logits, gts, pos_weight=torch.tensor(pos_weight, device=logits.device))
    dice = SoftDiceLoss()(logits, gts)
    return 0.7*bce + 0.3*dice, bce, dice

@torch.no_grad()
def sam_forward_for_pa(sam_ddp, batched_input):
    sam = sam_ddp
    if hasattr(sam_ddp, "module"): sam = sam_ddp.module
    out, interm = sam.forward_for_prompt_adapter(batched_input, multimask_output=False)
    B = len(out)
    enc = torch.cat([out[i]['encoder_embedding'] for i in range(B)], dim=0)
    image_pe = [out[i]['image_pe'] for i in range(B)]
    sparse_e = [out[i]['sparse_embeddings'] for i in range(B)]
    dense_e  = [out[i]['dense_embeddings']  for i in range(B)]
    image_record = [out[i]['image_record'] for i in range(B)]
    input_images = out[0]['input_images']
    return enc, image_pe, sparse_e, dense_e, image_record, input_images, interm


# -------------------------
# main pieces
# -------------------------
def train_one_epoch(args, pa, sam, optimizer, loader):
    pa.train()
    metric_logger = misc.MetricLogger(delimiter="  ")

    for batch in metric_logger.log_every(loader, 20):
        imgs = batch["image"].to(args.device, non_blocking=True)  # [B,3,H,W]
        gts  = batch["label"].to(args.device, non_blocking=True)  # [B,1,H,W] 0/255
        imgs_np = imgs.permute(0,2,3,1).cpu().numpy()

        # 使用 box 作为 prompt
        labels_box = misc.masks_to_boxes(gts[:,0,:,:])

        batched_input=[]
        for b in range(len(imgs_np)):
            inp = {
                "image": torch.as_tensor(imgs_np[b].astype(np.uint8), device=imgs.device).permute(2,0,1).contiguous(),
                "boxes": labels_box[b:b+1],
                "original_size": imgs_np[b].shape[:2],
                "label": gts[b:b+1],
            }
            batched_input.append(inp)

        with torch.no_grad():
            enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = sam_forward_for_pa(sam, batched_input)

        out = pa(
            image_embeddings=enc,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_e,
            dense_prompt_embeddings=dense_e,
            multimask_output=False,
            interm_embeddings=interm,
            image_record=image_record,
            prompt_encoder=(sam.module.prompt_encoder if hasattr(sam, "module") else sam.prompt_encoder),
            input_images=input_images,
        )

        # 取 logits（final->refined->coarse->masks_sam）
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
            raise RuntimeError("Cannot extract logits from PA output.")

        if logits.dim()==3: logits = logits.unsqueeze(1)
        if logits.shape[-2:] != gts.shape[-2:]:
            logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)

        loss, loss_bce, loss_dice = bce_dice_loss(logits, gts)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        metric_logger.update(training_loss=float(loss.item()),
                             loss_bce=float(loss_bce.item()),
                             loss_dice=float(loss_dice.item()))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items() if m.count>0}

@torch.no_grad()
def evaluate(args, pa, sam, loader, thr=0.0):
    pa.eval(); sam.eval()
    total_iou=0.0; n=0

    def iou_one(p,g,eps=1e-6):
        if g.sum().item()==0 and p.sum().item()==0: return 1.0
        inter=(p & g).sum().item()
        union=(p | g).sum().item()
        return inter/max(1,union)

    for batch in loader:
        imgs = batch["image"].to(args.device)
        gts  = batch["label"].to(args.device)
        if gts.dim()==3: gts=gts.unsqueeze(1)
        imgs_np = imgs.permute(0,2,3,1).cpu().numpy()
        labels_box = misc.masks_to_boxes(gts[:,0,:,:])

        batched_input=[]
        for b in range(len(imgs_np)):
            inp = {
                "image": torch.as_tensor(imgs_np[b].astype(np.uint8), device=imgs.device).permute(2,0,1).contiguous(),
                "boxes": labels_box[b:b+1],
                "original_size": imgs_np[b].shape[:2],
            }
            batched_input.append(inp)

        enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = sam_forward_for_pa(sam, batched_input)
        out = pa(
            image_embeddings=enc, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_e, dense_prompt_embeddings=dense_e,
            multimask_output=False, interm_embeddings=interm,
            image_record=image_record,
            prompt_encoder=(sam.module.prompt_encoder if hasattr(sam, "module") else sam.prompt_encoder),
            input_images=input_images,
        )

        logits=None
        if isinstance(out,(list,tuple)) and len(out)>0:
            for idx in [3,5,4,0]:
                if idx<len(out) and torch.is_tensor(out[idx]): logits=out[idx]; break
            if logits is None and torch.is_tensor(out[0]): logits=out[0]
        elif torch.is_tensor(out): logits=out
        elif isinstance(out,dict):
            for k in ["final_masks","refined_masks","coarse_masks","masks","logits","mask_logits"]:
                if k in out and torch.is_tensor(out[k]): logits=out[k]; break
        if logits is None: raise RuntimeError("Cannot extract logits for eval.")

        if logits.dim()==3: logits=logits.unsqueeze(1)
        if logits.shape[-2:] != gts.shape[-2:]:
            logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
        pred = (logits > thr).to(torch.bool)
        gt   = (gts > 0.5).to(torch.bool)

        B=pred.shape[0]
        for b in range(B):
            total_iou += iou_one(pred[b,0], gt[b,0])
            n += 1

    miou = total_iou/max(1,n)
    print(f"============================\nAveraged stats: val_iou_0: {miou:.4f}")
    return {"val_iou_0": miou}


# -------------------------
# arg & main
# -------------------------
def get_args():
    p = argparse.ArgumentParser("PA-SAM RGB clean", add_help=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-type", default="vit_l", choices=["vit_h","vit_l","vit_b"])
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])

    p.add_argument("--im_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--train_list", required=True)
    p.add_argument("--valid_list", required=True)
    p.add_argument("--test_list", required=True)

    p.add_argument("--input_size", nargs=2, type=int, default=[512,512])
    p.add_argument("--batch_size_train", type=int, default=4)
    p.add_argument("--batch_size_valid", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--start_epoch", type=int, default=0)
    p.add_argument("--max_epoch_num", type=int, default=6)
    p.add_argument("--model_save_fre", type=int, default=2)

    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--dist_url", default="env://")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--find_unused_params", action="store_true")
    return p.parse_args()

def main():
    args = get_args()
    misc.init_distributed_mode(args)
    os.makedirs(args.output, exist_ok=True)
    print(f"world_size:{args.world_size} rank:{args.rank} local_rank:{args.local_rank}")
    print(f"args: {args}\n")
    if args.device=="cuda" and not torch.cuda.is_available():
        args.device="cpu"

    # loaders（训练增强：Flip+LSJ；验证：Resize）
    train_loader = build_loader(
        args, args.train_list,
        transforms=T.Compose([RandomHFlip(), LargeScaleJitter()]),
        batch_size=args.batch_size_train, training=True
    )
    valid_loader = build_loader(
        args, args.valid_list,
        transforms=T.Compose([Resize(args.input_size)]),
        batch_size=args.batch_size_valid, training=False
    )

    # models
    pa = MaskDecoderPA(args.model_type).to(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

    # optimizer
    opt = optim.Adam(pa.parameters(), lr=args.learning_rate, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)

    best_iou=-1.0; best_path=None
    for epoch in range(args.start_epoch, args.max_epoch_num):
        os.environ["CURRENT_EPOCH"] = str(epoch)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        print(f"epoch: {epoch}  learning rate: {opt.param_groups[0]['lr']}")

        _ = train_one_epoch(args, pa, sam, opt, train_loader)
        stats = evaluate(args, pa, sam, valid_loader, thr=0.0)

        # save checkpoints
        if (epoch % args.model_save_fre)==0:
            pth = os.path.join(args.output, f"epoch_{epoch}.pth")
            torch.save(pa.state_dict(), pth)
            print(f"[save] {pth}")
        if stats["val_iou_0"] > best_iou:
            best_iou = stats["val_iou_0"]
            best_path = os.path.join(args.output, "best_model.pth")
            torch.save(pa.state_dict(), best_path)
            print(f"🎯 new best IoU={best_iou:.4f} -> {best_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()


'''
python train_rgb_clean.py \
  --output ./output/2025/WA/rgb/train_run2 \
  --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \
  --model-type vit_l \
  --device cuda \
  --im_dir ./data/2025/WA/image \
  --gt_dir ./data/2025/WA/index \
  --train_list ./data/2025/WA/splits/train.txt \
  --valid_list ./data/2025/WA/splits/valid.txt \
  --test_list  ./data/2025/WA/splits/test.txt \
  --batch_size_train 4 \
  --batch_size_valid 1 \
  --input_size 512 512 \
  --learning_rate 1e-3 \
  --start_epoch 0 --max_epoch_num 16

'''