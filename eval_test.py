#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, argparse, datetime, re, time, json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from segment_anything_training import sam_model_registry
import utils.misc as misc
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
from model.mask_decoder_pa import MaskDecoderPA


@torch.no_grad()
def _sam_forward_for_pa(sam_like, batched_input):
    """兼容 DDP 和非 DDP 的 forward_for_prompt_adapter"""
    sam_mod = sam_like.module if hasattr(sam_like, "module") else sam_like
    batched_output, interm_embeddings = sam_mod.forward_for_prompt_adapter(
        batched_input, multimask_output=False
    )
    B = len(batched_output)
    encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(B)], dim=0)
    image_pe          = [batched_output[i]['image_pe']          for i in range(B)]
    sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(B)]
    dense_embeddings  = [batched_output[i]['dense_embeddings']  for i in range(B)]
    image_record      = [batched_output[i]['image_record']      for i in range(B)]
    input_images      = batched_output[0]['input_images']
    return encoder_embedding, image_pe, sparse_embeddings, dense_embeddings, image_record, input_images, interm_embeddings


def bin_metrics(pred, gt, eps=1e-6):
    inter = torch.logical_and(pred, gt).sum().item()
    union = torch.logical_or(pred, gt).sum().item()
    tp = inter
    fp = torch.logical_and(pred, torch.logical_not(gt)).sum().item()
    fn = torch.logical_and(torch.logical_not(pred), gt).sum().item()
    iou  = (tp / (union + eps)) if union>0 else (1.0 if (gt.sum()==0 and pred.sum()==0) else 0.0)
    dice = (2*tp) / (2*tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    return iou, dice, prec, rec


def _infer_state(save_dir: str) -> str:
    return os.path.basename(os.path.normpath(save_dir)) or "UnknownState"


def _infer_mode_tag(restore_model: str, output_dir: str) -> str:
    def scan(p: str) -> str:
        p_low = p.lower()
        parts = [s for s in p_low.replace("\\", "/").split("/") if s]
        if "ggb" in parts: return "ggb"
        if "rgb" in parts: return "rgb"
        return ""
    tag = scan(restore_model)
    if not tag:
        tag = scan(output_dir)
    return tag or "rgb"


def _ensure_loader(loader_like):
    """把 create_dataloaders 返回的 loader 规范化成真正的 DataLoader"""
    # 1) 如果是 (list/tuple) 取第一个
    if isinstance(loader_like, (list, tuple)):
        assert len(loader_like) > 0, "Empty loader list returned by create_dataloaders"
        return loader_like[0]
    # 2) 如果是 dict，尝试常见键
    if isinstance(loader_like, dict):
        for k in ["test", "val", "eval", "loader", 0]:
            if k in loader_like:
                return loader_like[k]
        # 退一步：取第一个 value
        if len(loader_like) > 0:
            return list(loader_like.values())[0]
        raise RuntimeError("Empty loader dict returned by create_dataloaders")
    # 3) 否则直接返回（应当就是 DataLoader）
    return loader_like


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, argparse, datetime, re, time, json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from segment_anything_training import sam_model_registry
import utils.misc as misc
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
from model.mask_decoder_pa import MaskDecoderPA


@torch.no_grad()
def _sam_forward_for_pa(sam_like, batched_input):
    """兼容 DDP 和非 DDP 的 forward_for_prompt_adapter"""
    sam_mod = sam_like.module if hasattr(sam_like, "module") else sam_like
    batched_output, interm_embeddings = sam_mod.forward_for_prompt_adapter(
        batched_input, multimask_output=False
    )
    B = len(batched_output)
    encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(B)], dim=0)
    image_pe          = [batched_output[i]['image_pe']          for i in range(B)]
    sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(B)]
    dense_embeddings  = [batched_output[i]['dense_embeddings']  for i in range(B)]
    image_record      = [batched_output[i]['image_record']      for i in range(B)]
    input_images      = batched_output[0]['input_images']
    return encoder_embedding, image_pe, sparse_embeddings, dense_embeddings, image_record, input_images, interm_embeddings


def bin_metrics(pred, gt, eps=1e-6):
    inter = torch.logical_and(pred, gt).sum().item()
    union = torch.logical_or(pred, gt).sum().item()
    tp = inter
    fp = torch.logical_and(pred, torch.logical_not(gt)).sum().item()
    fn = torch.logical_and(torch.logical_not(pred), gt).sum().item()
    iou  = (tp / (union + eps)) if union>0 else (1.0 if (gt.sum()==0 and pred.sum()==0) else 0.0)
    dice = (2*tp) / (2*tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    return iou, dice, prec, rec


def _infer_state(save_dir: str) -> str:
    return os.path.basename(os.path.normpath(save_dir)) or "UnknownState"


def _infer_mode_tag(restore_model: str, output_dir: str) -> str:
    def scan(p: str) -> str:
        p_low = p.lower()
        parts = [s for s in p_low.replace("\\", "/").split("/") if s]
        if "ggb" in parts: return "ggb"
        if "rgb" in parts: return "rgb"
        return ""
    tag = scan(restore_model)
    if not tag:
        tag = scan(output_dir)
    return tag or "rgb"


def _ensure_loader(loader_like):
    """把 create_dataloaders 返回的 loader 规范化成真正的 DataLoader"""
    # 1) 如果是 (list/tuple) 取第一个
    if isinstance(loader_like, (list, tuple)):
        assert len(loader_like) > 0, "Empty loader list returned by create_dataloaders"
        return loader_like[0]
    # 2) 如果是 dict，尝试常见键
    if isinstance(loader_like, dict):
        for k in ["test", "val", "eval", "loader", 0]:
            if k in loader_like:
                return loader_like[k]
        # 退一步：取第一个 value
        if len(loader_like) > 0:
            return list(loader_like.values())[0]
        raise RuntimeError("Empty loader dict returned by create_dataloaders")
    # 3) 否则直接返回（应当就是 DataLoader）
    return loader_like


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)     # base SAM ckpt
    parser.add_argument("--restore-model", type=str, required=True)  # best_model.pth (PA decoder)
    parser.add_argument("--save_dir", type=str, required=True)       # /mnt/.../Trained/<STATE>
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input_size", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--model-type", type=str, default="vit_l", choices=["vit_h","vit_l","vit_b"])
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--find_unused_params", action="store_true")
    args = parser.parse_args()

    H, W = args.input_size

    # ----------------- DDP init (only once) -----------------
    if not hasattr(args, "world_size"):
        args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not hasattr(args, "rank"):
        args.rank = int(os.environ.get("RANK", "0"))
    if not hasattr(args, "local_rank"):
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not hasattr(args, "gpu"):
        args.gpu = args.local_rank

    misc.init_distributed_mode(args)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # ----------------- dataset/loader -----------------
    print("------------------------------ test --------------------------------")
    dataset_Alaska_test = {
        "name": "Alaska",
        "im_dir": "./data/2025/Alaska/test/image",
        "gt_dir": "./data/2025/Alaska/test/index",
        "im_ext": ".png",
        "gt_ext": ".png",
        "im_ch4_dir": "/mnt/d/Eelgrass_processed_images_2025/ModelData/Data_for_modeling/Alaska/test/glcm",
    }
    test_datasets = [dataset_Alaska_test]

    print(f"--->>> test  dataset  0 / {len(test_datasets)}   {test_datasets[0]['name']} <<<---")
    test_im_gt_list = get_im_gt_name_dict(test_datasets, flag="test")
    test_loader_like, _ = create_dataloaders(
        test_im_gt_list,
        my_transforms=[Resize([H, W])],
        batch_size=1,
        training=False,
    )
    # 关键修复：确保这里是一个真正的 DataLoader，而不是 [DataLoader] 的列表
    test_loader = _ensure_loader(test_loader_like)

    # ----------------- models -----------------
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device).eval()

    pa = MaskDecoderPA(args.model_type)
    pa.to(device)
    state = torch.load(args.restore_model, map_location="cpu")
    missing, unexpected = pa.load_state_dict(state, strict=False)
    if misc.is_main_process():
        print("PA Decoder init from SAM MaskDecoder")
        if missing or unexpected:
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
    pa.eval()

    # DDP（单卡也可）
    if torch.distributed.is_initialized():
        sam = torch.nn.parallel.DistributedDataParallel(
            sam, device_ids=[args.gpu] if args.device == "cuda" else None,
            find_unused_parameters=args.find_unused_params
        )
        pa = torch.nn.parallel.DistributedDataParallel(
            pa, device_ids=[args.gpu] if args.device == "cuda" else None,
            find_unused_parameters=args.find_unused_params
        )

    # ----------------- 输出目录 & 命名 -----------------
    os.makedirs(args.output, exist_ok=True)
    save_pred_dir = os.path.join(args.save_dir, "Predicted")
    save_stat_dir = os.path.join(args.save_dir, "Stats")
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_stat_dir, exist_ok=True)

    state_name = os.path.basename(os.path.normpath(args.save_dir))  # e.g., Alaska
    m = re.search(r"/(rgb|ggb)(/|$)", args.restore_model.replace("\\","/"), flags=re.IGNORECASE)
    mode  = m.group(1).lower() if m else "rgb"
    stamp = time.strftime("%y%m%d%H%M")
    base  = f"{state_name}_{mode}_{stamp}"

    summary_txt  = os.path.join(save_stat_dir, f"summary_{base}.txt")
    metrics_csv  = os.path.join(save_stat_dir, f"test_metrics_{base}.csv")

    # ----------------- 推理 & 指标 -----------------
    total_TP = total_FP = total_TN = total_FN = 0
    macro_list = []  # (IoU, Dice, Prec, Rec, Spec, Acc)

    csv_f = open(metrics_csv, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["filename", "IoU", "Dice", "Precision", "Recall", "Specificity", "Accuracy"])

    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            # --- 获取输入与 Ground Truth ---
            im  = sample["image"].to(device, non_blocking=True)          # [1,3,512,512]
            gtb = sample["label"].to(device, non_blocking=True).float()  # [1,1,512,512]

            # --- 获取源文件名 ---
            if "im_name" in sample:
                im_name = os.path.basename(sample["im_name"][0])
            elif "ori_im_path" in sample:
                im_name = os.path.basename(sample["ori_im_path"][0])
            else:
                im_name = f"{idx:06d}.png"  # fallback

            # 保证输出名为原图名 + ".png"
            if not im_name.lower().endswith(".png"):
                im_name = os.path.splitext(im_name)[0] + ".png"

            # --- 准备 SAM 输入 ---
            labels_box = misc.masks_to_boxes(gtb[:, 0])
            imgs_uint8 = (im.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

            input_image = torch.as_tensor(imgs_uint8[0].astype(np.uint8), device=device).permute(2, 0, 1).contiguous()
            batched_input = [{
                "image": input_image,
                "boxes": labels_box[0:1],
                "original_size": imgs_uint8[0].shape[:2],
                "label": gtb[0:1],
            }]

            # --- SAM + PA 前向 ---
            enc, image_pe, sparse_e, dense_e, image_record, input_images, interm = _sam_forward_for_pa(sam, batched_input)
            masks_sam, iou_preds, uncertain_maps, final_masks, coarse_masks, refined_masks, box_preds = pa(
                image_embeddings=enc,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_e,
                dense_prompt_embeddings=dense_e,
                multimask_output=False,
                interm_embeddings=interm,
                image_record=image_record,
                prompt_encoder=(sam.module.prompt_encoder if hasattr(sam, "module") else sam.prompt_encoder),
                input_images=input_images
            )

            # === logits -> 映射回原图坐标系（修复左上 1/4 问题）===
        # logits: [B,1,h,w] 或 [B,h,w]，来自 PA/SAM（通常是 256×256，对应 1024 画布）
        logits = refined_masks if refined_masks is not None else final_masks
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)  # -> [B,1,h,w]

        # 1) 取 SAM 画布大小（input_size）；没有就回退到 1024×1024
        if isinstance(image_record, (list, tuple)):
            rec0 = image_record[0]
        else:
            rec0 = image_record
        canvas_h, canvas_w = (rec0.get("input_size", (1024, 1024))
                            if isinstance(rec0, dict) else (1024, 1024))

        # 2) 放大到 SAM 正方形画布
        mask_on_canvas = F.interpolate(
            logits, size=(canvas_h, canvas_w), mode="bilinear", align_corners=False
        )  # [B,1,canvas_h,canvas_w]

        # 3) 依据 original_size 计算有效区域、去除 padding
        # original_size 是我们给 SAM 的原始图尺寸（此处即 dataloader Resize 后的 512×512）
        if isinstance(batched_input[0].get("original_size", None), (list, tuple)):
            ori_h, ori_w = batched_input[0]["original_size"]
        else:
            # 兜底：直接用 gtb 尺寸
            ori_h, ori_w = gtb.shape[-2], gtb.shape[-1]

        # SAM 的缩放：把长边缩放到 canvas 的边长（通常 1024），短边按比例 -> 有效区域 = 左上角 [0:valid_h, 0:valid_w]
        scale = float(max(canvas_h, canvas_w)) / float(max(ori_h, ori_w))
        valid_h = int(round(ori_h * scale))
        valid_w = int(round(ori_w * scale))
        valid_h = min(valid_h, canvas_h)
        valid_w = min(valid_w, canvas_w)

        mask_valid = mask_on_canvas[:, :, :valid_h, :valid_w]  # 去掉右/下 padding

        # 4) 把有效区域缩放回原图尺寸（与你的 GT/保存尺寸一致）
        mask_resized = F.interpolate(
            mask_valid, size=(ori_h, ori_w), mode="bilinear", align_corners=False
        )  # [B,1,ori_h,ori_w]

        # === 二值化并保存 ===
        pred_bin = (mask_resized > 0).to(torch.uint8) * 255
        pred_np  = pred_bin[0, 0].cpu().numpy()
        out_png  = os.path.join(save_pred_dir, im_name)
        Image.fromarray(pred_np).save(out_png)

        # === 计算 per-image 指标（与 GT 对齐）===
        p = (pred_np > 127).astype(np.uint8)
        g = (gtb[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        TP = int(((p == 1) & (g == 1)).sum())
        FP = int(((p == 1) & (g == 0)).sum())
        TN = int(((p == 0) & (g == 0)).sum())
        FN = int(((p == 0) & (g == 1)).sum())

        total_TP += TP; total_FP += FP; total_TN += TN; total_FN += FN

        iou  = TP / max(1, (TP + FP + FN))
        dice = (2 * TP) / max(1, (2*TP + FP + FN))
        prec = TP / max(1, (TP + FP))
        rec  = TP / max(1, (TP + FN))
        spec = TN / max(1, (TN + FP))
        acc  = (TP + TN) / max(1, (TP + TN + FP + FN))

        macro_list.append((iou, dice, prec, rec, spec, acc))
        csv_w.writerow([im_name, f"{iou:.4f}", f"{dice:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{spec:.4f}", f"{acc:.4f}"])

                    
    csv_f.close()

    # ----------------- 汇总 Macro / Micro 并写 summary -----------------
    macro_arr = np.array(macro_list) if len(macro_list) else np.zeros((1,6), dtype=float)
    m_iou, m_dice, m_prec, m_rec, m_spec, m_acc = macro_arr.mean(axis=0).tolist()

    micro_denom = max(1, total_TP + total_FP + total_FN)
    micro_iou  = total_TP / micro_denom
    micro_dice = (2*total_TP) / max(1, (2*total_TP + total_FP + total_FN))
    micro_prec = total_TP / max(1, (total_TP + total_FP))
    micro_rec  = total_TP / max(1, (total_TP + total_FN))
    micro_spec = total_TN / max(1, (total_TN + total_FP))
    micro_acc  = (total_TP + total_TN) / max(1, (total_TP + total_TN + total_FP + total_FN))

    if misc.is_main_process():
        with open(summary_txt, "w") as f:
            f.write(f"Base name: {base}\n")
            f.write("Macro mean metrics:\n")
            f.write(f"  IoU: {m_iou:.4f} | Dice: {m_dice:.4f} | Precision: {m_prec:.4f} | Recall: {m_rec:.4f}\n")
            f.write(f"  Specificity: {m_spec:.4f} | Accuracy: {m_acc:.4f}\n")
            f.write("Micro metrics:\n")
            f.write(f"  IoU: {micro_iou:.4f} | Dice: {micro_dice:.4f} | Precision: {micro_prec:.4f} | Recall: {micro_rec:.4f}\n")
            f.write(f"  Specificity: {micro_spec:.4f} | Accuracy: {micro_acc:.4f}\n")

        print("\n✅ Done.")
        print(f"Predicted PNGs -> {save_pred_dir}")
        print(f"Stats -> {save_stat_dir}")
        print(f"Base name: {base}")
        print("Macro mean metrics:")
        print(f"  IoU: {m_iou:.4f} | Dice: {m_dice:.4f} | Precision: {m_prec:.4f} | Recall: {m_rec:.4f}")
        print(f"  Specificity: {m_spec:.4f} | Accuracy: {m_acc:.4f}")
        print("Micro metrics:")
        print(f"  IoU: {micro_iou:.4f} | Dice: {micro_dice:.4f} | Precision: {micro_prec:.4f} | Recall: {micro_rec:.4f}")
        print(f"  Specificity: {micro_spec:.4f} | Accuracy: {micro_acc:.4f}")

    # ----------------- 结束 DDP -----------------
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()



if __name__ == "__main__":
    main()
