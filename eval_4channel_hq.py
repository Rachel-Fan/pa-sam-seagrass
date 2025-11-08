# eval_4channel.py
import os
import argparse
import json
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.transform import resize as sk_resize

# === PA-SAM bits ===
from segment_anything_training import sam_model_registry
from model.mask_decoder_pa import MaskDecoderPA

# === project utils ===
import utils.misc as misc  # for MetricLogger pacing


# ---------------------------
# Utilities
# ---------------------------
def read_split_list(split_txt: str) -> List[str]:
    """Return basename stems from a split file (one per line, accepts full paths)."""
    names = []
    with open(split_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = os.path.basename(s)
            s = os.path.splitext(s)[0]
            names.append(s)
    return names


def list_pairs(im_dir: str, gt_dir: str, im_ext: str, names_filter: List[str] = None) -> List[Tuple[str, str]]:
    """Match images and labels by stem name. If names_filter given, only keep those."""
    ims = sorted(glob(os.path.join(im_dir, f"*{im_ext}")))
    if names_filter is not None:
        names_set = set(names_filter)
        ims = [p for p in ims if os.path.splitext(os.path.basename(p))[0] in names_set]

    pairs = []
    missing = 0
    for ip in ims:
        name = os.path.splitext(os.path.basename(ip))[0]
        gp = os.path.join(gt_dir, f"{name}{im_ext}")
        if os.path.exists(gp):
            pairs.append((ip, gp))
        else:
            missing += 1
    if missing:
        print(f"[pairs] ⚠️ {missing} images missing GT under {gt_dir}")
    return pairs


# ---------------------------
# Dataset (HQ-style resize + 可选 GLCM→R)
# ---------------------------
class SimpleResizeDataset(Dataset):
    """
    - Reads RGB PNG and GT PNG
    - (optional) Read GLCM from im_ch4_dir and replace RED channel
    - Resizes both to input_size (H, W) using bilinear
    - Exposes fields close to HQ evaluate: 'image', 'label' (float), plus paths/names
    """
    def __init__(self, pairs, input_size=(512, 512),
                 im_ch4_dir=None, replace_red_with_glcm=False, im_ext=".png"):
        self.pairs = list(pairs)
        self.H, self.W = int(input_size[0]), int(input_size[1])
        self.im_ch4_dir = im_ch4_dir
        self.replace_red_with_glcm = replace_red_with_glcm
        self.im_ext = im_ext

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, gp = self.pairs[idx]
        name = os.path.splitext(os.path.basename(ip))[0]

        im = io.imread(ip)      # H,W,C or H,W
        gt = io.imread(gp)      # H,W or H,W,C
        if gt.ndim == 3:
            gt = gt[:, :, 0]

        # ---- optional: read GLCM and replace RED ----
        if self.replace_red_with_glcm:
            assert self.im_ch4_dir is not None, "replace_red_with_glcm=True 需要 --im_ch4_dir"
            ch4_path = os.path.join(self.im_ch4_dir, f"{name}{self.im_ext}")
            ch4 = io.imread(ch4_path)
            if ch4.ndim == 3:
                ch4 = ch4[:, :, 0]

            H0, W0 = im.shape[:2]
            if ch4.shape[:2] != (H0, W0):
                ch4 = sk_resize(ch4, (H0, W0), order=0, preserve_range=True, anti_aliasing=False).astype(im.dtype)
            else:
                if ch4.dtype != im.dtype:
                    ch4 = ch4.astype(im.dtype)

            if im.ndim == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)

            # 覆盖红通道
            im[:, :, 0] = ch4
        else:
            if im.ndim == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)

        # ---- resize to model input size (HQ-SAM eval 风格) ----
        im_t = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)   # 1,3,H,W
        gt_t = torch.tensor(gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)       # 1,1,H,W

        im_r = F.interpolate(im_t, size=(self.H, self.W), mode="bilinear", align_corners=False).squeeze(0)
        gt_r = F.interpolate(gt_t, size=(self.H, self.W), mode="bilinear", align_corners=False).squeeze(0)

        return {
            "image": im_r,           # 3,H,W float32
            "label": gt_r,           # 1,H,W float32（注意：仍是 0/255 或 0/1 的原值域，后续阈值化）
            "name": name,
            "im_path": ip,
            "gt_path": gp
        }


# ---------------------------
# SAM + PA decoder inference (HQ evaluate 风格)
# ---------------------------
@torch.no_grad()
def infer_and_save_masks(args, net: torch.nn.Module, sam, loader: DataLoader):
    """
    HQ-SAM 风格的数据处理：
      GT → 生成 box prompt → sam.forward_for_prompt_adapter → PA 解码器
    导出：
      512×512 的 0/255 单通道 PNG mask
    指标：
      可选 IoU/Dice（overall & 非空 GT）
    """
    os.makedirs(args.vis_dir, exist_ok=True)
    do_metrics = bool(args.metrics)
    if do_metrics:
        os.makedirs(os.path.join(args.output, "metrics"), exist_ok=True)
        rows = []

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net.eval()
    sam.eval()

    logger = misc.MetricLogger(delimiter="  ")

    for batch in logger.log_every(loader, 100, logger=None, print_func=print):
        img = batch["image"].to(device)    # B,3,512,512
        gt  = batch["label"].to(device)    # B,1,512,512 (float)
        names = batch["name"]

        B, _, H, W = img.shape

        # --- 构建 box prompt（来自 GT）---
        # HQ-SAM 的 evaluate 用 GT 生成 prompt；这里用 box（也最稳）
        # 先把 GT 二值化成 uint8（兼容 0/255 与 0/1）
        gt_bin_uint8 = (gt > 127).to(torch.uint8) if gt.max() > 1.5 else (gt > 0.5).to(torch.uint8)
        # 用项目里的工具把二值 mask → boxes（N,4），我们取每张图的 box（若多连通/空则兜底全图框）
        try:
            boxes_all = misc.masks_to_boxes(gt_bin_uint8[:, 0, :, :])  # B,4
        except Exception:
            boxes_all = None

        # SAM 的图像需要 uint8 HWC
        imgs_uint8 = (img.clamp(0, 255) if img.max() > 1.5 else (img * 255.0)).byte()
        imgs_uint8 = imgs_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # B,H,W,3 uint8

        batched_input = []
        for b in range(B):
            di = {}
            di["image"] = torch.as_tensor(imgs_uint8[b], device=device).permute(2, 0, 1).contiguous()  # 3,H,W uint8

            if boxes_all is not None:
                # 使用由 GT 提取的 box（x0,y0,x1,y1）
                di["boxes"] = boxes_all[b:b+1]
            else:
                # 兜底：全图框
                di["boxes"] = torch.tensor([[0, 0, W-1, H-1]], dtype=torch.float32, device=device)

            di["original_size"] = (H, W)
            batched_input.append(di)

        # ---- SAM forward（返回 encoder + prompt embeddings）----
        batched_output, interm_embeddings = sam.forward_for_prompt_adapter(batched_input, multimask_output=False)

        # 聚合入 PA 解码器
        encoder_embedding = torch.cat([batched_output[i]["encoder_embedding"] for i in range(B)], dim=0)  # B,C,h,w
        image_pe          = [batched_output[i]["image_pe"] for i in range(B)]
        sparse_embeddings = [batched_output[i]["sparse_embeddings"] for i in range(B)]
        dense_embeddings  = [batched_output[i]["dense_embeddings"] for i in range(B)]
        image_record      = [batched_output[i]["image_record"] for i in range(B)]
        input_images      = batched_output[0]["input_images"]  # shared

        # ---- PA 解码 ----
        masks_sam, iou_preds, uncertain_maps, final_masks, coarse_masks, refined_masks, box_preds = net(
            image_embeddings=encoder_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            image_record=image_record,
            prompt_encoder=sam.prompt_encoder,
            input_images=input_images
        )

        # 统一到 512×512 并二值化（logits>0）
        pred_logits_up = F.interpolate(masks_sam, size=(H, W), mode="bilinear", align_corners=False)  # B,1,H,W
        pred_bin = (pred_logits_up > 0).to(torch.uint8)  # B,1,H,W (0/1)

        # 保存 mask（0/255 单通道）
        for b in range(B):
            out_mask = (pred_bin[b, 0].cpu().numpy() * 255).astype(np.uint8)
            save_path = os.path.join(args.vis_dir, f"{names[b]}.png")
            io.imsave(save_path, out_mask, check_contrast=False)

        # 可选指标
        if do_metrics:
            inter = (pred_bin & gt_bin_uint8).sum(dim=(1, 2, 3)).float()
            union = (pred_bin | gt_bin_uint8).sum(dim=(1, 2, 3)).float()
            gt_pix = gt_bin_uint8.sum(dim=(1, 2, 3)).float()
            pr_pix = pred_bin.sum(dim=(1, 2, 3)).float()

            iou_overall = (inter / union.clamp_min(1)).cpu().numpy()
            dice_overall = (2 * inter / (pr_pix + gt_pix).clamp_min(1)).cpu().numpy()

            for b in range(B):
                rows.append({
                    "filename": names[b],
                    "pixels_pred": int(pr_pix[b].item()),
                    "pixels_gt": int(gt_pix[b].item()),
                    "inter": int(inter[b].item()),
                    "union": int(union[b].item()),
                    "iou": float(iou_overall[b]),
                    "dice": float(dice_overall[b]),
                })

    # 保存指标文件
    if do_metrics:
        import csv
        metrics_dir = os.path.join(args.output, "metrics")
        csv_path = os.path.join(metrics_dir, "metrics_test.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "pixels_pred", "pixels_gt", "inter", "union", "iou", "dice"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

        ious = [r["iou"] for r in rows]
        dices = [r["dice"] for r in rows]
        ious_ne = [r["iou"] for r in rows if r["pixels_gt"] > 0]
        dices_ne = [r["dice"] for r in rows if r["pixels_gt"] > 0]

        summary = {
            "count": len(rows),
            "mean_iou_overall": float(np.mean(ious)) if len(ious) else 0.0,
            "mean_dice_overall": float(np.mean(dices)) if len(dices) else 0.0,
            "mean_iou_non_empty": float(np.mean(ious_ne)) if len(ious_ne) else 0.0,
            "mean_dice_non_empty": float(np.mean(dices_ne)) if len(dices_ne) else 0.0,
        }
        with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("[Metrics]", json.dumps(summary, indent=2))


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("PA-SAM eval (4ch style) — HQ evaluate-like reading & processing", add_help=True)

    # data
    p.add_argument("--im_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--im_ext", type=str, default=".png")
    p.add_argument("--split_txt", type=str, default=None,
                   help="Optional test list (one name per line, with/without extension; full paths OK)")
    p.add_argument("--im_ch4_dir", type=str, default=None,
                   help="Folder of single-channel GLCM (e.g., entropy) images that share the same basename.")
    p.add_argument("--replace_red_with_glcm", action="store_true",
                   help="If set, read GLCM from --im_ch4_dir and replace the red channel before feeding SAM.")

    # model
    p.add_argument("--checkpoint", type=str, required=True, help="Official SAM checkpoint")
    p.add_argument("--decoder_ckpt", type=str, required=True, help="Trained PA decoder checkpoint (best_model.pth)")
    p.add_argument("--model_type", type=str, default="vit_l", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # loader
    p.add_argument("--input_size", nargs=2, type=int, default=[512, 512])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")

    # output
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--vis_dir", type=str, required=True, help="Where to save 512×512 BW masks")
    p.add_argument("--metrics", action="store_true", help="Compute IoU/Dice and save CSV/JSON")

    # torchrun 兼容参数（本脚本 eval 不用 DDP，但保留参数避免出错）
    p.add_argument("--dist_url", default="env://")
    p.add_argument("--dist_backend", default="nccl")
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--distributed", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # --- Build file list (with explicit diagnostics like you saw) ---
    names = None
    if args.split_txt:
        split_abs = os.path.abspath(args.split_txt)
        print(f"[split] requested file: {args.split_txt} (abs: {split_abs})")
        if os.path.isfile(split_abs):
            try:
                names = read_split_list(split_abs)
                print(f"[split] loaded names: {len(names)}")
            except Exception as e:
                print(f"[split] ✗ Failed to read split file: {e}. Falling back to glob-all.")
        else:
            print(f"[split] ✗ Not found on disk. Falling back to glob-all.")

    pairs = list_pairs(args.im_dir, args.gt_dir, args.im_ext, names_filter=names)
    print(f"[pairs] total pairs: {len(pairs)}")
    if len(pairs) == 0:
        print("No test pairs found. Check --im_dir/--gt_dir/--im_ext and the split stems.")
        return

    # Dataset / DataLoader
    ds = SimpleResizeDataset(
        pairs, input_size=args.input_size,
        im_ch4_dir=args.im_ch4_dir,
        replace_red_with_glcm=args.replace_red_with_glcm,
        im_ext=args.im_ext
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Build SAM & PA decoder
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    net = MaskDecoderPA(args.model_type).to(device)

    # Load decoder checkpoint (supports state dict with/without "mask_decoder." prefix)
    dec_state = torch.load(args.decoder_ckpt, map_location="cpu")
    net_state = net.state_dict()
    loaded = 0
    for k in list(net_state.keys()):
        if k in dec_state and net_state[k].shape == dec_state[k].shape:
            net_state[k] = dec_state[k]; loaded += 1
        elif ("mask_decoder." + k) in dec_state and net_state[k].shape == dec_state["mask_decoder." + k].shape:
            net_state[k] = dec_state["mask_decoder." + k]; loaded += 1
    net.load_state_dict(net_state, strict=False)
    print(f"[Decoder] loaded: {args.decoder_ckpt} (matched {loaded} tensors)")

    # Run
    infer_and_save_masks(args, net, sam, loader)


if __name__ == "__main__":
    main()



'''
torchrun --nproc_per_node=1 eval_4channel_hq.py \
  --im_dir ./data/2025/BC/image \
  --gt_dir ./data/2025/BC/index \
  --im_ext .png \
  --split_txt ./data/2025/BC/splits/test.txt \
  --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \
  --decoder_ckpt ./output/2025/BC/rgb/train_run2/best_model.pth \
  --model_type vit_l \
  --device cuda \
  --input_size 512 512 \
  --batch_size 1 \
  --metrics \
  --output ./output/2025/BC/rgb/train_run2/eval \
  --vis_dir ./output/2025/BC/rgb/train_run2/eval/masks_512_bw

'''