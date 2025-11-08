# eval_fullres_sliding_3channel.py
# ==========================================================
# PA-SAM full-resolution inference (RGB only) with sliding windows.
# - Keeps original image size; outputs masks same size as input (no quadrant bug).
# - Sliding window with overlap and smooth blending to avoid seams.
# - Saves 0/255 single-channel masks; optional overlay/panel; optional metrics.
#
# Author: RF + assistant | 2025-10
# ==========================================================

import os
import argparse
import json
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# === PA-SAM bits ===
from segment_anything_training import sam_model_registry
from model.mask_decoder_pa import MaskDecoderPA

# === project utils
import utils.misc as misc  # MetricLogger; optional

# ---------------------------
# Utilities
# ---------------------------
def read_split_list(split_txt: str, im_ext: str) -> List[str]:
    """
    读取测试文件清单，返回“无后缀的文件名”列表。
    允许行内是：名字（无后缀）、名字+后缀、或带路径。
    空行/注释行(#)会被忽略。
    """
    names = []
    with open(split_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            base = os.path.basename(s)
            root, ext = os.path.splitext(base)
            names.append(root if root else base.replace(ext, ""))  # 防御
    # 去重保持顺序
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); ordered.append(n)
    return ordered


def list_pairs(im_dir: str, gt_dir: str, im_ext: str, names_filter: List[str]) -> List[Tuple[str, str]]:
    """
    严格依据 names_filter 生成 (image_path, gt_path) 对。
    若图像或GT缺失，打印告警但跳过。
    """
    assert names_filter is not None and len(names_filter) > 0
    pairs, miss = [], 0
    for name in names_filter:
        ip = os.path.join(im_dir, f"{name}{im_ext}")
        gp = os.path.join(gt_dir, f"{name}{im_ext}")
        ok = True
        if not os.path.exists(ip):
            print(f"[WARN] image not found: {ip}")
            ok = False
        if not os.path.exists(gp):
            print(f"[WARN] GT not found   : {gp}")
            ok = False
        if ok:
            pairs.append((ip, gp))
        else:
            miss += 1
    if miss:
        print(f"[INFO] missing pairs skipped: {miss}")
    return pairs


def to_uint8_01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return (x * 255.0).round().astype(np.uint8)


def make_hann2d(h: int, w: int) -> np.ndarray:
    """2D Hann window (seam-smoothing weight map for blending)."""
    wy = 0.5 * (1 - np.cos(2 * np.pi * np.arange(h) / max(1, h)))
    wx = 0.5 * (1 - np.cos(2 * np.pi * np.arange(w) / max(1, w)))
    w2d = np.outer(wy, wx).astype(np.float32)
    # stabilize: avoid zero at borders to keep denominator > 0
    eps = 1e-3
    return np.clip(w2d, eps, 1.0)


# ---------------------------
# Dataset (no resize; original size)
# ---------------------------
class FullResImageDataset(Dataset):
    """
    - Reads RGB PNG/TIF and GT PNG
    - NO resize; keep original H,W
    - Returns numpy arrays (for SAM encoder expectation)
    """
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, gp = self.pairs[idx]
        im = np.array(Image.open(ip))  # H, W, C?
        gt = np.array(Image.open(gp))

        # ensure RGB uint8
        if im.ndim == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        if im.shape[2] > 3:
            im = im[:, :, :3]
        if im.dtype != np.uint8:
            # assume [0,1] or [0,255]-like float; normalize then to uint8
            m = im.max()
            im = (im * (255.0 if m <= 1.5 else 1.0)).clip(0, 255).astype(np.uint8)

        # GT to single channel
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        # binarization deferred to metrics

        sample = {
            "image_np": im,                 # H, W, 3 uint8
            "gt_np": gt,                    # H, W (uint8 or 0/255)
            "name": os.path.splitext(os.path.basename(ip))[0],
            "im_path": ip,
            "gt_path": gp
        }
        return sample


# ---------------------------
# SAM + PA decoder: per-patch inference helper
# ---------------------------
@torch.inference_mode()
def pa_sam_forward_patch(device, sam, net, patch_uint8: np.ndarray) -> np.ndarray:
    """
    Run PA-SAM on a single RGB uint8 patch and return probability map in [0,1].
    We normalize the shapes of prompt embeddings BEFORE passing to the PA decoder so that:
      - sparse_prompt_embeddings: (B, N, C)
      - dense_prompt_embeddings : (B, C, H', W')
      - input_images            : Tensor (B, 3, H_in, W_in)
      - image_record            : list-like (indexable by batch)
    This avoids changing the decoder code.
    """
    h, w, _ = patch_uint8.shape
    B_expected = 1  # we run one patch at a time

    # Build SAM input (CHW uint8) and a full-image box prompt
    image_chw = torch.as_tensor(patch_uint8, device=device).permute(2, 0, 1).contiguous()  # (3,h,w) uint8
    full_box = torch.tensor([[0, 0, w - 1, h - 1]], dtype=torch.float32, device=device)

    batched_input = [{
        "image": image_chw,            # (3,h,w) uint8
        "boxes": full_box,             # (1,4) float
        "original_size": (h, w),
    }]

    # Get embeddings from SAM helper
    batched_output, interm_embeddings = sam.forward_for_prompt_adapter(
        batched_input, multimask_output=False
    )

    enc_emb       = batched_output[0]["encoder_embedding"]    # (1,C,hh,ww) or (C,hh,ww)
    image_pe      = batched_output[0]["image_pe"]
    sparse_emb    = batched_output[0]["sparse_embeddings"]    # many variants → normalize below
    dense_emb     = batched_output[0]["dense_embeddings"]     # (B,C,H',W') or (C,H',W')
    image_record  = batched_output[0]["image_record"]         # decoder indexes by [i]
    input_images  = batched_output[0]["input_images"]         # MUST be Tensor (B,3,H_in,W_in)

    # ---------- container/type guards ----------
    if not isinstance(image_record, (list, tuple)):
        image_record = [image_record]  # make it indexable

    if isinstance(input_images, (list, tuple)):
        # common case: [tensor] -> tensor
        if len(input_images) == 1 and torch.is_tensor(input_images[0]):
            input_images = input_images[0]
        else:
            input_images = torch.stack(
                [t if torch.is_tensor(t) else torch.as_tensor(t, device=device) for t in input_images],
                dim=0
            )

    # Ensure enc_emb has batch dim
    if torch.is_tensor(enc_emb) and enc_emb.dim() == 3:
        enc_emb = enc_emb.unsqueeze(0)  # -> (1,C,hh,ww)

    # ---------- shape normalization helpers ----------
    def norm_sparse_to_BNC(x, B: int) -> torch.Tensor:
        """
        Normalize sparse prompt embeddings to shape (B, N, C).
        Accepts inputs of shapes: (C,), (N,C), (B,C), (B,N,C), list/tuple of tensors.
        """
        if x is None:
            # no prompts → make empty (B,0,C) that will not change concat result
            C = enc_emb.shape[-1] if enc_emb.dim() >= 1 else 256
            return enc_emb.new_zeros((B, 0, C))

        if isinstance(x, (list, tuple)):
            # Flatten list elements to (N,C) and then expand to (B,N,C)
            toks = []
            for t in x:
                t = t if torch.is_tensor(t) else torch.as_tensor(t, device=device)
                if t.dim() == 1:            # (C,)
                    t = t.unsqueeze(0)      # -> (1,C)
                elif t.dim() > 2:
                    # collapse any extra leading dims except channel: (..., C) -> (N,C)
                    t = t.reshape(-1, t.shape[-1])
                toks.append(t)
            if len(toks) == 0:
                C = enc_emb.shape[-1]
                stacked = enc_emb.new_zeros((0, C))
            else:
                stacked = torch.cat(toks, dim=0)   # (N,C)
            return stacked.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B,N,C)

        # Tensor branch
        if x.dim() == 1:
            # (C,) -> (1,1,C) -> expand to (B,1,C)
            return x.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).contiguous()
        if x.dim() == 2:
            # Ambiguous: could be (N,C) or (B,C). Heuristic: if first dim == B → treat as (B,C)
            if x.shape[0] == B:
                return x.unsqueeze(1)  # (B,1,C)
            else:
                return x.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B,N,C)
        if x.dim() == 3:
            # Already (B,N,C) or (1,N,C)
            if x.shape[0] == B:
                return x
            if x.shape[0] == 1 and B > 1:
                return x.expand(B, -1, -1).contiguous()
            return x  # best effort
        # Fallback: reshape to (B,1,C)
        return x.reshape(B, 1, -1)

    def norm_dense_to_BCHW(x, B: int) -> torch.Tensor:
        """Normalize dense embeddings to (B, C, H, W)."""
        if isinstance(x, (list, tuple)):
            xs = []
            for t in x:
                t = t if torch.is_tensor(t) else torch.as_tensor(t, device=device)
                if t.dim() == 3:  # (C,H,W)
                    t = t.unsqueeze(0)
                xs.append(t)
            return torch.cat(xs, dim=0)  # assume already B
        if torch.is_tensor(x):
            if x.dim() == 3:     # (C,H,W)
                return x.unsqueeze(0)
            return x
        # Fallback: make an empty tensor to avoid crash
        C = enc_emb.shape[1] if torch.is_tensor(enc_emb) and enc_emb.dim() == 4 else 256
        return enc_emb.new_zeros((B, C, 1, 1))

    # ---------- apply normalization ----------
    sparse_emb = norm_sparse_to_BNC(sparse_emb, B_expected)       # -> (B,N,C)
    dense_emb  = norm_dense_to_BCHW(dense_emb,  B_expected)       # -> (B,C,H',W')

    # Optional one-time shape diagnostics
    # print("[DBG shapes]",
    #       "enc", tuple(enc_emb.shape),
    #       "sparse", tuple(sparse_emb.shape),
    #       "dense", tuple(dense_emb.shape),
    #       "input_images", tuple(input_images.shape) if torch.is_tensor(input_images) else type(input_images))

    # ---------- PA decoder forward ----------
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        masks_sam, _, _, _, _, _, _ = net(
            image_embeddings=enc_emb,                 # (B,C,hh,ww)
            image_pe=image_pe,                        # tensor
            sparse_prompt_embeddings=sparse_emb,      # (B,N,C)
            dense_prompt_embeddings=dense_emb,        # (B,C,H',W')
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            image_record=image_record,                # list-like
            prompt_encoder=sam.prompt_encoder,
            input_images=input_images                 # Tensor (B,3,H_in,W_in)
        )

    # Resize logits back to patch size and sigmoid
    logits_up = F.interpolate(masks_sam, size=(h, w), mode="bilinear", align_corners=False)  # (B,1,h,w)
    prob = torch.sigmoid(logits_up)[0, 0].float().cpu().numpy()
    return prob



def sliding_window_infer_full(
    device,
    sam,
    net,
    img_uint8: np.ndarray,
    win: int = 1024,
    overlap: float = 0.25
) -> np.ndarray:
    """
    Full-resolution sliding window inference with blending.
    Returns prob map (H,W) float32 in [0,1].
    """
    H, W, _ = img_uint8.shape
    stride = max(1, int(win * (1 - overlap)))

    # accumulators
    acc = np.zeros((H, W), dtype=np.float32)
    wmap = np.zeros((H, W), dtype=np.float32)

    # blending weights per window
    w_patch = make_hann2d(win, win)

    # pad if smaller than window
    pad_h = (win - H % win) % win if H < win else 0
    pad_w = (win - W % win) % win if W < win else 0
    if pad_h or pad_w:
        img_pad = np.pad(img_uint8, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        img_pad = img_uint8
    HP, WP, _ = img_pad.shape

    # slide
    ys = list(range(0, max(HP - win, 0) + 1, stride))
    xs = list(range(0, max(WP - win, 0) + 1, stride))
    if ys[-1] != HP - win:
        ys.append(HP - win)
    if xs[-1] != WP - win:
        xs.append(WP - win)

    for y in ys:
        for x in xs:
            patch = img_pad[y:y+win, x:x+win, :]  # (win,win,3) uint8
            prob = pa_sam_forward_patch(device, sam, net, patch)  # (win,win) [0,1]
            # write back (crop to original canvas region if near pad border)
            y0, y1 = y, min(y + win, H)
            x0, x1 = x, min(x + win, W)
            ph, pw = y1 - y0, x1 - x0

            acc[y0:y1, x0:x1] += prob[:ph, :pw] * w_patch[:ph, :pw]
            wmap[y0:y1, x0:x1] += w_patch[:ph, :pw]

    prob_full = np.divide(acc, np.clip(wmap, 1e-6, None), dtype=np.float32)
    return np.clip(prob_full, 0.0, 1.0)


def save_overlay(rgb_u8: np.ndarray, prob01: np.ndarray, save_path: str, alpha=0.45, color=(255, 0, 0)):
    m = (prob01 > 0.5).astype(np.uint8)
    overlay = rgb_u8.copy()
    color_arr = np.zeros_like(overlay, dtype=np.uint8)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    overlay[m == 1] = (alpha * color_arr[m == 1] + (1 - alpha) * overlay[m == 1]).astype(np.uint8)
    Image.fromarray(overlay).save(save_path)


def save_quad_panel(rgb_u8: np.ndarray, gt01: np.ndarray, prob01: np.ndarray, save_path: str):
    H, W, _ = rgb_u8.shape
    gt_u8 = to_uint8_01(gt01)
    pr_u8 = to_uint8_01(prob01)
    m_u8 = ((prob01 > 0.5).astype(np.uint8) * 255)

    gt_rgb = np.repeat(gt_u8[..., None], 3, axis=2)
    pr_rgb = np.repeat(pr_u8[..., None], 3, axis=2)
    m_rgb  = np.repeat(m_u8[..., None], 3, axis=2)

    top = np.concatenate([rgb_u8, gt_rgb], axis=1)
    bot = np.concatenate([pr_rgb, m_rgb], axis=1)
    panel = np.concatenate([top, bot], axis=0)
    Image.fromarray(panel).save(save_path)


# ---------------------------
# Inference loop (full-res)
# ---------------------------
@torch.inference_mode()
def infer_fullres_and_save(args, net: torch.nn.Module, sam, loader: DataLoader):
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    do_metrics = bool(args.metrics)
    if do_metrics:
        os.makedirs(os.path.join(args.output, "metrics"), exist_ok=True)
        rows = []

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net.eval()
    sam.eval()

    logger = misc.MetricLogger(delimiter="  ")

    for batch in logger.log_every(loader, 1, logger=None, print_func=print):
        name = batch["name"][0]
        img_u8 = batch["image_np"][0].numpy() if isinstance(batch["image_np"], torch.Tensor) else batch["image_np"][0]
        gt_np  = batch["gt_np"][0].numpy()    if isinstance(batch["gt_np"], torch.Tensor) else batch["gt_np"][0]

        if isinstance(img_u8, np.ndarray) and img_u8.ndim == 3:
            rgb_u8 = img_u8
        else:
            rgb_u8 = np.array(img_u8)

        H, W, _ = rgb_u8.shape

        # === Full-res sliding window inference ===
        prob_full = sliding_window_infer_full(
            device=device,
            sam=sam,
            net=net,
            img_uint8=rgb_u8,
            win=args.win_size,
            overlap=args.overlap
        )
        mask_bin = (prob_full > 0.5).astype(np.uint8)

        # Save mask (same size as input)
        mask_path = os.path.join(args.vis_dir, f"{name}.png")
        Image.fromarray((mask_bin * 255).astype(np.uint8)).save(mask_path)

        if args.save_overlay:
            save_overlay(rgb_u8, prob_full, os.path.join(args.vis_dir, f"{name}_overlay.png"))
        if args.save_panel:
            gt01 = (gt_np > 127).astype(np.float32) if gt_np.max() > 1.5 else (gt_np > 0.5).astype(np.float32)
            save_quad_panel(rgb_u8, gt01, prob_full, os.path.join(args.vis_dir, f"{name}_panel.png"))

        # Metrics on full-res
        if do_metrics:
            gt_bin = (gt_np > 127).astype(np.uint8) if gt_np.max() > 1.5 else (gt_np > 0.5).astype(np.uint8)
            inter = np.logical_and(mask_bin, gt_bin).sum()
            union = np.logical_or(mask_bin, gt_bin).sum()
            pr_pix = mask_bin.sum()
            gt_pix = gt_bin.sum()

            iou = (inter / max(union, 1)) if union > 0 else 0.0
            dice = (2 * inter / max(pr_pix + gt_pix, 1)) if (pr_pix + gt_pix) > 0 else 0.0

            rows.append({
                "filename": name,
                "H": H, "W": W,
                "pixels_pred": int(pr_pix),
                "pixels_gt": int(gt_pix),
                "inter": int(inter),
                "union": int(union),
                "iou": float(iou),
                "dice": float(dice),
            })

    if do_metrics:
        import csv
        metrics_dir = os.path.join(args.output, "metrics")
        csv_path = os.path.join(metrics_dir, "metrics_test_fullres.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filename","H","W","pixels_pred","pixels_gt","inter","union","iou","dice"])
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
        with open(os.path.join(metrics_dir, "summary_fullres.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("[Metrics-FullRes]", json.dumps(summary, indent=2))


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("PA-SAM full-res eval (RGB) — sliding-window, same-size output", add_help=True)

    # data
    p.add_argument("--im_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--im_ext", type=str, default=".png")
    p.add_argument("--split_txt", type=str, required=True)

    # model
    p.add_argument("--checkpoint", type=str, required=True, help="Official SAM checkpoint")
    p.add_argument("--decoder_ckpt", type=str, required=True, help="Trained PA decoder checkpoint (best_model.pth)")
    p.add_argument("--model_type", type=str, default="vit_l", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # sliding window
    p.add_argument("--win_size", type=int, default=1024, help="patch window size (e.g., 512/768/1024)")
    p.add_argument("--overlap", type=float, default=0.25, help="overlap ratio (0~0.75), 0.25 recommended")

    # loader (we iterate one-by-one; batching at full-res usually not needed)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")

    # output
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--vis_dir", type=str, required=True, help="Where to save full-size BW masks/overlays")
    p.add_argument("--save_overlay", action="store_true")
    p.add_argument("--save_panel", action="store_true")
    p.add_argument("--metrics", action="store_true", help="Compute IoU/Dice on full-size masks")

    # (harmless) distributed args for torchrun compatibility
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

    # Build file list (MUST from splits.txt)
    if not (args.split_txt and os.path.exists(args.split_txt)):
        raise FileNotFoundError(f"--split_txt not found: {args.split_txt}")

    names = read_split_list(args.split_txt, args.im_ext)
    if len(names) == 0:
        raise RuntimeError(f"Empty split list: {args.split_txt}")

    pairs = list_pairs(args.im_dir, args.gt_dir, args.im_ext, names_filter=names)
    print(f"Test samples in split: {len(names)} | valid pairs found: {len(pairs)}")
    if len(pairs) == 0:
        print("No valid pairs. Check names in splits.txt, --im_dir/--gt_dir/--im_ext.")
        return


    # Dataset / DataLoader
    ds = FullResImageDataset(pairs)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory,
                        collate_fn=lambda x: {
                            "image_np": [x[0]["image_np"]],
                            "gt_np": [x[0]["gt_np"]],
                            "name": [x[0]["name"]],
                            "im_path": [x[0]["im_path"]],
                            "gt_path": [x[0]["gt_path"]],
                        })

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
    infer_fullres_and_save(args, net, sam, loader)


if __name__ == "__main__":
    main()
