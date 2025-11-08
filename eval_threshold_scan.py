
import os
import argparse
import time
import math
import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
    pass
else:
    # 让 dataloader 认为是单进程
    import torch
    torch.distributed = None

from segment_anything_training import sam_model_registry
import utils.misc as misc
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
from model.mask_decoder_pa import MaskDecoderPA

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def add_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument("--output", type=str, required=True, help="Dir to write reports/visualizations")
    parser.add_argument("--logfile", type=str, default=None, help="Log file path")
    parser.add_argument("--model-type", type=str, default="vit_l", choices=["vit_h","vit_l","vit_b"])
    parser.add_argument("--checkpoint", type=str, required=True, help="SAM checkpoint path (or merged sam_pa)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])

    parser.add_argument("--restore-model", type=str, default=None,
                        help="Optional PA decoder checkpoint. If None, we assume checkpoint is merged SAM+PA.")

    parser.add_argument("--input_size", nargs=2, type=int, default=[512,512], metavar=("H","W"),
                        help="Eval size H W; ground-truth will be resized to this size.")

    # Threshold sweep
    parser.add_argument("--thr_min", type=float, default=-4.0, help="Min logit threshold (inclusive)")
    parser.add_argument("--thr_max", type=float, default=4.0, help="Max logit threshold (inclusive)")
    parser.add_argument("--thr_step", type=float, default=0.25, help="Logit threshold step")
    parser.add_argument("--report_csv", type=str, default="eval_threshold_scan.csv", help="CSV name under --output")
    parser.add_argument("--pr_curve_png", type=str, default="pr_curve.png", help="PR curve PNG under --output")

    # Distributed compatibility (ignored)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--distributed", action="store_true")

    return parser

def build_valid_loader(input_size: List[int]):
    dataset_val = {"name":"Alaska",
                   "im_dir":"./data/2025/Alaska/valid/image",
                   "gt_dir":"./data/2025/Alaska/valid/index",
                   "im_ext":".png","gt_ext":".png"}
    # dataloader may expect these extra keys; reuse im_dir for them
    for d in (dataset_val,):
        d.setdefault("im_ch2_dir", d["im_dir"])
        d.setdefault("im_ch3_dir", d["im_dir"])
        d.setdefault("im_ch4_dir", d["im_dir"])

    valid_im_gt = get_im_gt_name_dict([dataset_val], flag="valid")
    valid_loader, _ = create_dataloaders(valid_im_gt,
                                         my_transforms=[Resize(input_size)],
                                         batch_size=1, training=False)
    return valid_loader

def load_models(args):
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    if args.restore_model:
        pa = MaskDecoderPA(args.model_type)
        ckpt = torch.load(args.restore_model, map_location="cpu")
        try:
            pa.load_state_dict(ckpt, strict=False)
        except Exception as e:
            print("[WARN] Loading PA decoder with strict=False failed:", e)
        pa = pa.to(device=args.device)
        pa.eval()
        sam.eval()
        return sam, pa
    else:
        pa = MaskDecoderPA(args.model_type).to(device=args.device).eval()  # dummy; weights inside sam if merged
        sam.eval()
        return sam, pa

def forward_once(sam, pa, batch):
    with torch.no_grad():
        imgs = batch['image']
        if torch.cuda.is_available() and next(sam.parameters()).is_cuda:
            imgs = imgs.cuda(non_blocking=True)

        imgs_np = imgs.permute(0,2,3,1).cpu().numpy()
        labels_val = batch['label']
        if torch.cuda.is_available() and next(sam.parameters()).is_cuda:
            labels_val = labels_val.cuda(non_blocking=True)
        labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])

        batched_input = []
        for b in range(len(imgs_np)):
            d = {}
            d['image'] = torch.as_tensor(imgs_np[b].astype(np.uint8), device=imgs.device).permute(2,0,1).contiguous()
            d['boxes'] = labels_box[b:b+1]
            d['original_size'] = imgs_np[b].shape[:2]
            d['label'] = labels_val[b:b+1]
            batched_input.append(d)

        batched_output, interm_embeddings = sam.forward_for_prompt_adapter(batched_input, multimask_output=False)

        B = len(batched_output)
        encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(B)], dim=0)
        image_pe = [batched_output[i]['image_pe'] for i in range(B)]
        sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(B)]
        dense_embeddings = [batched_output[i]['dense_embeddings'] for i in range(B)]
        image_record = [batched_output[i]['image_record'] for i in range(B)]
        input_images = batched_output[0]['input_images']

        masks_sam, *_ = pa(
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
        return masks_sam, labels_val

def main():
    parser = argparse.ArgumentParser("Eval Threshold Sweep", add_help=True)
    parser = add_eval_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if not args.logfile:
        args.logfile = os.path.join(args.output, "eval.log")
    if os.path.exists(args.logfile):
        os.remove(args.logfile)
    logging.basicConfig(filename=args.logfile, level=logging.INFO)
    def logprint(*a):
        s = " ".join(str(x) for x in a)
        logging.info(s); print(s, flush=True)

    logprint("Args:", args)

    valid_loader = build_valid_loader(args.input_size)
    sam, pa = load_models(args)

    thresholds = []
    t = args.thr_min
    while t <= args.thr_max + 1e-9:
        thresholds.append(round(t, 6))
        t += args.thr_step

    # Global accumulators for pixel-level PR/AP and dataset-level IoU (Jaccard)
    tp = np.zeros(len(thresholds), dtype=np.int64)
    fp = np.zeros(len(thresholds), dtype=np.int64)
    fn = np.zeros(len(thresholds), dtype=np.int64)

    total_images = 0
    start = time.time()

    for i, batch in enumerate(valid_loader):
        logits, gt = forward_once(sam, pa, batch)  # logits: (B,1,h,w)
        logits = F.interpolate(logits, size=tuple(args.input_size), mode="bilinear", align_corners=False)
        gt = F.interpolate(gt.float(), size=tuple(args.input_size), mode="nearest").long()
        probs = torch.sigmoid(logits)  # (B,1,H,W)

        B, _, H, W = probs.shape
        probs_np = probs.detach().cpu().numpy()
        gt_np = (gt.detach().cpu().numpy() > 0).astype(np.uint8)

        for b in range(B):
            total_images += 1
            p = probs_np[b,0]
            g = gt_np[b,0]
            gsum = g.sum()

            for ti, thr in enumerate(thresholds):
                # convert logit threshold to prob threshold
                pthr = 1.0 / (1.0 + math.exp(-thr))
                pred = (p >= pthr).astype(np.uint8)

                inter = np.logical_and(pred, g).sum()
                tp[ti] += inter
                fp[ti] += int(pred.sum() - inter)
                fn[ti] += int(gsum - inter)

        if (i+1) % 200 == 0 or (i+1) == len(valid_loader):
            logprint(f"[{i+1}/{len(valid_loader)}] processed")

    # Produce metrics per threshold
    rows = []
    best_jacc = (-1.0, None)
    prec_list, rec_list = [], []

    for ti, thr in enumerate(thresholds):
        inter = tp[ti]; union = tp[ti] + fp[ti] + fn[ti]
        jaccard = (inter/union) if union>0 else 0.0
        precision = inter/(inter+fp[ti]) if (inter+fp[ti])>0 else 0.0
        recall = inter/(inter+fn[ti]) if (inter+fn[ti])>0 else 0.0

        prec_list.append(precision); rec_list.append(recall)
        rows.append({"thr_logit":thr, "dataset_jaccard":jaccard, "precision":precision, "recall":recall})
        if jaccard > best_jacc[0]:
            best_jacc = (jaccard, thr)

    # 11-point AP
    order = np.argsort(rec_list)
    rec_sorted = np.array(rec_list)[order]
    prec_sorted = np.array(prec_list)[order]
    ap = 0.0
    for r in np.linspace(0,1,11):
        p = prec_sorted[rec_sorted >= r].max() if np.any(rec_sorted >= r) else 0.0
        ap += p
    ap /= 11.0

    # Write CSV
    import csv
    csv_path = os.path.join(args.output, args.report_csv)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["thr_logit","dataset_jaccard","precision","recall"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Save PR curve
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(rec_list, prec_list)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Pixel-level PR (AP={ap:.4f})")
        plt.grid(True)
        fig_path = os.path.join(args.output, args.pr_curve_png)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logprint("[WARN] Failed to save PR curve:", e)

    # Summary
    with open(os.path.join(args.output, "summary.txt"), "w") as f:
        f.write(f"Total images: {total_images}\n")
        f.write(f"Best dataset-level IoU (pixel Jaccard): {best_jacc[0]:.4f} at thr={best_jacc[1]:.3f}\n")
        f.write(f"Pixel-level AP: {ap:.4f}\n")

    logprint(f"=> CSV saved: {csv_path}")
    logprint(f"=> Best dataset Jaccard: {best_jacc[0]:.4f} at thr={best_jacc[1]:.3f}")
    logprint(f"=> Pixel-level AP: {ap:.4f}")
    logprint("Done.")

if __name__ == "__main__":
    main()
