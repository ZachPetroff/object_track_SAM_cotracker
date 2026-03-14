"""
Evaluate predicted bounding boxes against ground truth annotations.

Metrics reported per annotation, per subject, and overall:
  - Mean IoU
  - Success Rate at IoU thresholds 0.5 and 0.75
  - Success Rate curve (AUC) across thresholds [0.05, 0.10, ..., 0.95]
  - Mean absolute box-center error (pixels)
  - Missing frame rate (frames where no prediction was made)

Usage:
    python evaluate.py --pred_dir <path> --annot_dir <path> --split <split>
                       [--image_width 1920] [--image_height 1080]
                       [--out_file results.json] [--verbose]
"""

import argparse
import json
import os

import numpy as np


# ──────────────────────────────────────────────
# IoU utilities
# ──────────────────────────────────────────────

def box_iou(box_a, box_b, eps=1e-9):
    """
    Compute IoU between two boxes in xyxy pixel format.

    Args:
        box_a, box_b: sequences of (x0, y0, x1, y1)

    Returns:
        float IoU in [0, 1]
    """
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw  = max(0.0, ix1 - ix0)
    ih  = max(0.0, iy1 - iy0)
    inter = iw * ih

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union  = area_a + area_b - inter

    return inter / (union + eps)


def box_center_error(box_pred, box_gt):
    """Euclidean distance between predicted and ground-truth box centers (pixels)."""
    cx_p = (box_pred[0] + box_pred[2]) / 2.0
    cy_p = (box_pred[1] + box_pred[3]) / 2.0
    cx_g = (box_gt[0]   + box_gt[2])   / 2.0
    cy_g = (box_gt[1]   + box_gt[3])   / 2.0
    return float(np.hypot(cx_p - cx_g, cy_p - cy_g))


# ──────────────────────────────────────────────
# Per-annotation evaluation
# ──────────────────────────────────────────────

def evaluate_annotation(pred_boxes, gt_boxes, frame_ids, image_hw, thresholds):
    """
    Evaluate one annotation sequence.

    Args:
        pred_boxes:  list of [x0,y0,x1,y1] or None, one per frame in frame_ids
        gt_boxes:    list of [x0_norm, y0_norm, x1_norm, y1_norm] (normalized),
                     one per entry in frame_ids
        frame_ids:   list of int frame indices (used for reporting only)
        image_hw:    (H, W) used to convert gt_boxes to pixels
        thresholds:  list of IoU threshold values for success-rate curve

    Returns:
        dict with per-frame and aggregate metrics
    """
    H, W = image_hw

    ious          = []
    center_errors = []
    missing       = []
    per_frame     = []

    for i, fid in enumerate(frame_ids):
        pred = pred_boxes[i] if i < len(pred_boxes) else None
        gt_n = gt_boxes[i]

        # Convert GT from normalized to pixel coords
        gt_px = [gt_n[0] * W, gt_n[1] * H, gt_n[2] * W, gt_n[3] * H]

        if pred is None:
            missing.append(fid)
            per_frame.append({"frame_id": fid, "iou": None, "center_error": None, "missing": True})
            continue

        iou = box_iou(pred, gt_px)
        ce  = box_center_error(pred, gt_px)
        ious.append(iou)
        center_errors.append(ce)
        per_frame.append({"frame_id": fid, "iou": iou, "center_error": ce, "missing": False})

    ious_arr = np.array(ious) if ious else np.array([])
    ce_arr   = np.array(center_errors) if center_errors else np.array([])

    # Success rate at each threshold
    success_rates = {}
    for t in thresholds:
        key = f"sr_{t:.2f}"
        success_rates[key] = float((ious_arr >= t).mean()) if len(ious_arr) > 0 else 0.0

    auc = float(np.mean(list(success_rates.values()))) if success_rates else 0.0

    return {
        "mean_iou":          float(ious_arr.mean())    if len(ious_arr) > 0 else None,
        "mean_center_error": float(ce_arr.mean())      if len(ce_arr)   > 0 else None,
        "sr_0.50":           success_rates.get("sr_0.50"),
        "sr_0.75":           success_rates.get("sr_0.75"),
        "auc":               auc,
        "missing_rate":      len(missing) / max(len(frame_ids), 1),
        "n_frames":          len(frame_ids),
        "n_missing":         len(missing),
        "success_curve":     success_rates,
        "per_frame":         per_frame,
    }


# ──────────────────────────────────────────────
# Top-level evaluation
# ──────────────────────────────────────────────

def evaluate(pred_dir, annot_dir, split, image_hw, thresholds, verbose=False):
    """
    Load all prediction files and ground-truth annotations, evaluate, and
    return a results dict.
    """
    annot_path = os.path.join(annot_dir, f"object_tracking_{split}.json")
    with open(annot_path, "r") as f:
        gt_data = json.load(f)

    all_ious          = []
    all_center_errors = []
    all_missing       = []
    all_sr            = {f"{t:.2f}": [] for t in thresholds}

    results = {"per_subject": {}, "split": split}

    for subject, annots in gt_data.items():
        pred_path = os.path.join(pred_dir, f"predictions_{subject}.json")
        if not os.path.exists(pred_path):
            print(f"[WARN] No predictions found for subject '{subject}', skipping.")
            continue

        with open(pred_path, "r") as f:
            pred_data = json.load(f)

        subject_results = {}

        for annot in annots:
            ann_id     = annot["id"]
            frame_ids  = annot["frame_ids"]
            gt_boxes   = annot["bounding_boxes"]

            if ann_id not in pred_data:
                print(f"[WARN] Annotation '{ann_id}' not found in predictions for '{subject}', skipping.")
                continue

            pred_boxes = pred_data[ann_id]["boxes"]

            ann_result = evaluate_annotation(
                pred_boxes, gt_boxes, frame_ids, image_hw, thresholds
            )
            subject_results[ann_id] = ann_result

            if ann_result["mean_iou"] is not None:
                all_ious.append(ann_result["mean_iou"])
            all_center_errors.extend(
                [f["center_error"] for f in ann_result["per_frame"] if f["center_error"] is not None]
            )
            all_missing.append(ann_result["missing_rate"])
            for t in thresholds:
                key = f"{t:.2f}"
                sr_val = ann_result["success_curve"].get(f"sr_{t:.2f}", 0.0)
                all_sr[key].append(sr_val)

            if verbose:
                print(
                    f"  {subject} / {ann_id}: "
                    f"mIoU={ann_result['mean_iou']:.3f}  "
                    f"SR@0.5={ann_result['sr_0.50']:.3f}  "
                    f"AUC={ann_result['auc']:.3f}  "
                    f"missing={ann_result['missing_rate']:.1%}"
                )

        results["per_subject"][subject] = subject_results

    # Overall aggregate
    results["overall"] = {
        "mean_iou":          float(np.mean(all_ious))          if all_ious          else None,
        "mean_center_error": float(np.mean(all_center_errors)) if all_center_errors else None,
        "mean_missing_rate": float(np.mean(all_missing))       if all_missing       else None,
        "sr_0.50":           float(np.mean(all_sr["0.50"]))    if all_sr["0.50"]    else None,
        "sr_0.75":           float(np.mean(all_sr["0.75"]))    if all_sr["0.75"]    else None,
        "auc":               float(np.mean([np.mean(v) for v in all_sr.values() if v])),
        "success_curve":     {k: float(np.mean(v)) for k, v in all_sr.items() if v},
        "n_subjects":        len(results["per_subject"]),
    }

    return results


def print_summary(results):
    o = results["overall"]
    print("\n" + "="*50)
    print(f"  Evaluation results — split: {results['split']}")
    print("="*50)
    print(f"  Subjects evaluated : {o['n_subjects']}")
    print(f"  Mean IoU           : {o['mean_iou']:.4f}" if o["mean_iou"] is not None else "  Mean IoU           : N/A")
    print(f"  Success Rate @0.50 : {o['sr_0.50']:.4f}" if o["sr_0.50"] is not None else "  SR@0.50            : N/A")
    print(f"  Success Rate @0.75 : {o['sr_0.75']:.4f}" if o["sr_0.75"] is not None else "  SR@0.75            : N/A")
    print(f"  AUC (SR curve)     : {o['auc']:.4f}")
    print(f"  Mean center error  : {o['mean_center_error']:.2f} px" if o["mean_center_error"] is not None else "  Mean center error  : N/A")
    print(f"  Missing frame rate : {o['mean_missing_rate']:.2%}" if o["mean_missing_rate"] is not None else "  Missing rate       : N/A")
    print("="*50 + "\n")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object tracking predictions.")
    parser.add_argument("--pred_dir",   required=True, help="Directory containing predictions_<subject>.json files")
    parser.add_argument("--annot_dir",  required=True, help="Directory containing object_tracking_<split>.json")
    parser.add_argument("--split",      required=True, help="Dataset split (e.g. train, val, test)")
    parser.add_argument("--image_width",  type=int, default=1920, help="Frame width in pixels")
    parser.add_argument("--image_height", type=int, default=1080, help="Frame height in pixels")
    parser.add_argument("--out_file",   default=None, help="Optional path to save results JSON")
    parser.add_argument("--verbose",    action="store_true", help="Print per-annotation results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2).tolist()

    results = evaluate(
        pred_dir=args.pred_dir,
        annot_dir=args.annot_dir,
        split=args.split,
        image_hw=(args.image_height, args.image_width),
        thresholds=thresholds,
        verbose=args.verbose,
    )

    print_summary(results)

    if args.out_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out_file}")
