"""
Video object tracking pipeline using SAM2 + CoTracker.

Usage:
    python track.py --subject <subject_id> --split <split>
                    [--video_dir <path>] [--annot_dir <path>] [--out_dir <path>]
                    [--sam2_checkpoint <path>] [--sam2_config <path>]
                    [--cotracker_checkpoint <path>]
                    [--n_points 50] [--seed 3]
"""

import argparse
import json
import os

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cotracker.predictor import CoTrackerPredictor


# ──────────────────────────────────────────────
# Bounding-box utilities
# ──────────────────────────────────────────────

def box_iou_matrix(boxes_a, boxes_b, eps=1e-9):
    """Compute pairwise IoU between two sets of boxes (N,4) and (M,4) in xyxy format."""
    ax0, ay0, ax1, ay1 = boxes_a[:,0], boxes_a[:,1], boxes_a[:,2], boxes_a[:,3]
    bx0, by0, bx1, by1 = boxes_b[:,0], boxes_b[:,1], boxes_b[:,2], boxes_b[:,3]
    inter_x0 = np.maximum(ax0[:,None], bx0[None,:])
    inter_y0 = np.maximum(ay0[:,None], by0[None,:])
    inter_x1 = np.minimum(ax1[:,None], bx1[None,:])
    inter_y1 = np.minimum(ay1[:,None], by1[None,:])
    iw = np.clip(inter_x1 - inter_x0, 0, None)
    ih = np.clip(inter_y1 - inter_y0, 0, None)
    inter = iw * ih
    area_a = (ax1 - ax0)[:,None] * (ay1 - ay0)[:,None]
    area_b = (bx1 - bx0)[None,:] * (by1 - by0)[None,:]
    union = area_a + area_b - inter + eps
    return inter / union


def average_iou(box, boxes, eps=1e-9):
    """Average IoU between one box and an array of boxes, both in xyxy format."""
    b = np.asarray(box, dtype=np.float64).reshape(4)
    B = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if B.size == 0:
        return float("nan")

    b_x0, b_y0 = min(b[0], b[2]), min(b[1], b[3])
    b_x1, b_y1 = max(b[0], b[2]), max(b[1], b[3])
    b_area = max(0.0, b_x1 - b_x0) * max(0.0, b_y1 - b_y0)

    X0 = np.minimum(B[:,0], B[:,2]); Y0 = np.minimum(B[:,1], B[:,3])
    X1 = np.maximum(B[:,0], B[:,2]); Y1 = np.maximum(B[:,1], B[:,3])

    iw = np.maximum(0.0, np.minimum(b_x1, X1) - np.maximum(b_x0, X0))
    ih = np.maximum(0.0, np.minimum(b_y1, Y1) - np.maximum(b_y0, Y0))
    inter = iw * ih
    union = b_area + np.maximum(0.0, X1 - X0) * np.maximum(0.0, Y1 - Y0) - inter

    iou = inter / np.maximum(union, eps)
    valid = union > 0
    return float(iou[valid].mean()) if np.any(valid) else float("nan")


def box_center(box):
    x0, y0, x1, y1 = box
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=float)


def points_inside_box(pts_xy, box_xyxy):
    x0, y0, x1, y1 = box_xyxy
    return (
        (pts_xy[:,0] >= x0) & (pts_xy[:,0] <= x1) &
        (pts_xy[:,1] >= y0) & (pts_xy[:,1] <= y1)
    )


def mean_interior_margin(pts_xy, box_xyxy):
    """
    Mean signed distance of points to the nearest box edge, normalized by
    min(width, height). Positive = inside, negative = outside.
    """
    x0, y0, x1, y1 = box_xyxy
    w = max(float(x1 - x0), 1.0)
    h = max(float(y1 - y0), 1.0)
    signed_margin = np.minimum(
        np.minimum(pts_xy[:,0] - x0, x1 - pts_xy[:,0]),
        np.minimum(pts_xy[:,1] - y0, y1 - pts_xy[:,1])
    )
    return (signed_margin / max(min(w, h), 1.0)).mean()


def mask_to_xyxy(mask, min_area=1):
    """Return [x0, y0, x1, y1] bounding box of a binary mask, or None if too small."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if (x1 - x0 + 1) * (y1 - y0 + 1) < min_area:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def choose_best_box(
    boxes,
    pts_xy,
    vis_weights=None,
    prev_box=None,
    image_hw=None,
    w_iou=1.0, w_cov=0.02, w_margin=1.2, w_center=1.2, w_area=1.5,
    min_pts_inside=0,
):
    """
    Score candidate boxes and return the best one.

    Score = w_iou*IoUConsensus + w_cov*PointCoverage + w_margin*InteriorMargin
            + w_center*CenterConsistency + w_area*AreaConsistency

    Returns:
        best_index (int or None), scores (np.ndarray)
    """
    boxes = np.asarray(boxes, dtype=float).copy()
    boxes[:,0:2] = np.minimum(boxes[:,0:2], boxes[:,2:4])
    boxes[:,2:4] = np.maximum(boxes[:,0:2], boxes[:,2:4])

    N = len(boxes)
    if N == 0:
        return None, np.array([])

    ious = box_iou_matrix(boxes, boxes)
    np.fill_diagonal(ious, 0.0)
    iou_consensus = ious.mean(axis=1) if N > 1 else np.zeros(N)

    pts_xy = np.asarray(pts_xy, dtype=float) if pts_xy is not None else np.zeros((0, 2))
    vis = (np.asarray(vis_weights, dtype=float).clip(min=0.0)
           if vis_weights is not None else np.ones(len(pts_xy)))

    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    area_prior = (max(1.0, (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1]))
                  if prev_box is not None else None)

    if image_hw is not None:
        H, W = image_hw
        diag_norm = np.hypot(W, H)
    else:
        diag_norm = max(np.median(np.hypot(boxes[:,2] - boxes[:,0], boxes[:,3] - boxes[:,1])), 1.0)

    prev_ctr = box_center(prev_box) if prev_box is not None else None
    scores = np.zeros(N)

    for i, b in enumerate(boxes):
        if len(pts_xy) > 0:
            inside = points_inside_box(pts_xy, b)
            cov = vis[inside].sum() / (vis.sum() + 1e-9) if vis.sum() > 0 else 0.0
            margin = mean_interior_margin(pts_xy, b)
            if inside.sum() < min_pts_inside:
                cov *= 0.2
                margin *= 0.2
        else:
            cov, margin, inside = 0.0, 0.0, np.array([], dtype=bool)

        if area_prior is not None and areas[i] > 0:
            area_cons = np.exp(-abs(np.log(areas[i] / area_prior)))
        else:
            med_area = np.median(areas[areas > 0]) if np.any(areas > 0) else 1.0
            area_cons = np.exp(-abs(np.log(max(areas[i], 1.0) / max(med_area, 1.0))))

        if prev_ctr is not None:
            center_cons = np.exp(-np.linalg.norm(box_center(b) - prev_ctr) / (0.25 * diag_norm))
        else:
            center_cons = 0.0

        scores[i] = (
            w_iou   * iou_consensus[i]
            + w_cov    * cov
            + w_margin * max(0.0, margin)
            + w_center * center_cons
            + w_area   * area_cons
        )

    return int(np.argmax(scores)), scores


# ──────────────────────────────────────────────
# Mask utilities
# ──────────────────────────────────────────────

def erode(mask, k=7, iterations=5, min_active=0, ensure_min_after=None):
    """
    Morphologically erode a binary mask.
    If ensure_min_after is set, stop early rather than drop below that pixel count.
    """
    m = np.asarray(mask) > 0
    if m.sum() < int(min_active):
        return mask

    se = np.ones((k, k), dtype=bool)
    if ensure_min_after is None:
        return binary_erosion(m, structure=se, iterations=iterations, border_value=0).astype(mask.dtype)

    out = m
    for _ in range(int(iterations)):
        nxt = binary_erosion(out, structure=se, iterations=1, border_value=0)
        if nxt.sum() < int(ensure_min_after):
            break
        out = nxt
    return out.astype(mask.dtype)


def sample_points_from_mask(mask, n_points=None, replace=False, bbox_xyxy=None, rng=None):
    """
    Sample up to n_points (x, y) pixel coordinates from the foreground of a binary mask.
    Optionally restrict to a bounding box region.
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask).astype(bool)
    H, W = m.shape[:2]

    if bbox_xyxy is not None:
        x0, y0, x1, y1 = bbox_xyxy
        x0 = max(0, min(W - 1, int(np.floor(x0))))
        x1 = max(0, min(W - 1, int(np.ceil(x1))))
        y0 = max(0, min(H - 1, int(np.floor(y0))))
        y1 = max(0, min(H - 1, int(np.ceil(y1))))
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        bbox_mask = np.zeros_like(m)
        bbox_mask[y0:y1+1, x0:x1+1] = True
        m &= bbox_mask

    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return np.empty((0, 2), dtype=int)

    coords = np.stack([xs, ys], axis=-1)
    if n_points is None or n_points >= len(coords):
        return coords

    if rng is None:
        rng = np.random.default_rng()
    return coords[rng.choice(len(coords), size=n_points, replace=replace)]


# ──────────────────────────────────────────────
# Video I/O
# ──────────────────────────────────────────────

def read_video_from_path(path, start_frame=0, end_frame_exclusive=None, step=1):
    """Read a video file into a (T, H, W, C) uint8 numpy array."""
    if step <= 0:
        raise ValueError("step must be a positive integer")
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print(f"Error opening video '{path}': {e}")
        return None

    frames = []
    try:
        for i, frame in enumerate(reader):
            if i < start_frame:
                continue
            if end_frame_exclusive is not None and i >= end_frame_exclusive:
                break
            if (i - start_frame) % step != 0:
                continue
            frames.append(np.asarray(frame))
    finally:
        reader.close()

    return np.stack(frames) if frames else None


# ──────────────────────────────────────────────
# Serialization helper
# ──────────────────────────────────────────────

def to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ──────────────────────────────────────────────
# Main tracking logic
# ──────────────────────────────────────────────

def track_subject(
    subject,
    split,
    video_dir,
    annot_dir,
    out_dir,
    sam2_checkpoint,
    sam2_config,
    cotracker_checkpoint,
    n_points=50,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path = os.path.join(video_dir, f"{subject}.mp4")
    annot_path = os.path.join(annot_dir, f"object_tracking_{split}.json")
    os.makedirs(out_dir, exist_ok=True)

    # Load annotations
    with open(annot_path, "r") as f:
        data = json.load(f)
    annots = data[subject]

    # Load models
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    tracker = CoTrackerPredictor(checkpoint=cotracker_checkpoint).to(device).eval()

    predictions = {}

    for annot in annots:
        bbs_norm = annot["bounding_boxes"]
        ids = annot["frame_ids"]
        init_idx = ids[0]

        # Load video
        clip_np = read_video_from_path(video_path)
        if clip_np is None:
            print(f"Could not read video: {video_path}")
            continue
        print(f"Video shape: {clip_np.shape}")
        T, H, W, C = clip_np.shape

        video_tensor = torch.from_numpy(clip_np).permute(0, 3, 1, 2)[None].float().to(device)

        # Convert initial box from normalized to pixel coordinates
        x0n, y0n, x1n, y1n = bbs_norm[0]
        init_box_pix = np.array([x0n * W, y0n * H, x1n * W, y1n * H], dtype=float)

        # Segment the object in the initial frame
        predictor.set_image(clip_np[init_idx])
        masks, _, _ = predictor.predict(box=init_box_pix[None, :], multimask_output=False)
        init_mask = erode(masks[0].astype(np.uint8), k=5, iterations=3,
                          min_active=400, ensure_min_after=500)

        # Sample anchor points from the mask
        seed_points = sample_points_from_mask(init_mask, n_points, bbox_xyxy=init_box_pix)
        queries = torch.tensor(
            [[init_idx, pt[0], pt[1]] for pt in seed_points],
            device=device, dtype=torch.float32
        )[None]

        # Track points across all frames
        with torch.inference_mode():
            pred_tracks, pred_visibility = tracker(video_tensor, queries=queries)

        print(f"CoTracker output: {pred_tracks.shape}")

        # Per-frame box prediction
        box_preds = []
        best_box = None

        for frame_idx in range(T):
            curr_frame = clip_np[frame_idx]
            pts_xy = pred_tracks[0, frame_idx].detach().cpu().numpy()       # (K, 2), pixel (x, y)
            vis    = pred_visibility[0, frame_idx].to(torch.bool).detach().cpu().numpy()

            # Generate candidate boxes from each tracked point
            predictor.set_image(curr_frame)
            boxes = []
            for (x, y) in pts_xy:
                masks, _, _ = predictor.predict(
                    point_coords=np.array([[x, y]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=False,
                )
                box = mask_to_xyxy(masks[0].astype(np.uint8))
                if box is not None:
                    boxes.append(box)

            if len(boxes) == 0:
                box_preds.append(best_box)
                continue

            # Dynamically adjust scoring weights based on point visibility
            n_pts = len(pts_xy)
            q = float(np.clip(vis.sum() / max(n_pts, 1), 0, 1))

            if best_box is not None:
                best_idx, _ = choose_best_box(
                    boxes, pts_xy,
                    prev_box=best_box,
                    image_hw=(H, W),
                    w_iou=1.1 - 0.3 * q,
                    w_cov=0.02 + 0.7 * q,
                    w_margin=0.02 + 0.7 * q,
                    w_center=1.2 - 0.7 * q,
                    w_area=1.2 - 0.4 * q,
                )
            else:
                best_idx, _ = choose_best_box(boxes, pts_xy)

            if best_idx is None:
                box_preds.append(best_box)
                continue

            best_box = boxes[best_idx]
            box_preds.append(best_box)

        predictions[annot["id"]] = {"boxes": box_preds, "frames": ids}

    # Save predictions
    out_file = os.path.join(out_dir, f"predictions_{subject}.json")
    with open(out_file, "w") as f:
        json.dump(predictions, f, default=to_serializable, indent=2)
    print(f"Wrote predictions to {out_file}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Track objects in video using SAM2 + CoTracker.")
    parser.add_argument("--subject",  required=True,  help="Subject/video identifier")
    parser.add_argument("--split",    required=True,  help="Dataset split (e.g. train, val, test)")
    parser.add_argument("--video_dir",  default="data/videos",
                        help="Directory containing <subject>.mp4 files")
    parser.add_argument("--annot_dir",  default="data/annotations",
                        help="Directory containing object_tracking_<split>.json files")
    parser.add_argument("--out_dir",    default="predictions",
                        help="Output directory for prediction JSON files")
    parser.add_argument("--sam2_checkpoint",    default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_config",        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--cotracker_checkpoint", default="checkpoints/scaled_offline.pth")
    parser.add_argument("--n_points", type=int, default=50,
                        help="Number of anchor points sampled from the initial mask")
    parser.add_argument("--seed",     type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    track_subject(
        subject=args.subject,
        split=args.split,
        video_dir=args.video_dir,
        annot_dir=args.annot_dir,
        out_dir=args.out_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        cotracker_checkpoint=args.cotracker_checkpoint,
        n_points=args.n_points,
    )