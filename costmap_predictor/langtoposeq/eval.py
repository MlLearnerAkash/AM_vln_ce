"""
Evaluation script for LangTopoSeg.

Metrics computed per-batch and averaged over the val/test split:

  e3d regression
  ──────────────
  • MAE   – mean absolute error between pred_e3d and GT e3d
  • RMSE  – root mean squared error
  • Pearson ρ   – per-sample, then averaged
  • Spearman ρ  – per-sample rank correlation, then averaged

  Node selection  (threshold τ=0.5 on node_mask)
  ──────────────────────────────────────────────
  • Precision, Recall, F1 – averaged over valid nodes

  Direction prediction  (only for samples with direction priors)
  ─────────────────────────────────────────────────────────────
  • Angular error (degrees) between dir_2d and prior direction

Usage
-----
    python eval.py --ckpt_path checkpoints/langtoposeq/epoch_049.pt
    python eval.py --ckpt_path ckpt.pt --data_root /my/data --split val
"""

import os
import math
import logging
import argparse
import csv
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from scipy.stats import pearsonr, spearmanr
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from config  import LangTopoSegConfig
from dataset import LangTopoDataset
from model   import LangTopoSeg
from losses  import extract_direction_gt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-batch metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _per_sample_e3d_metrics(
    pred: torch.Tensor,  # [B, K]
    gt:   torch.Tensor,  # [B, K]
    k_valid: torch.Tensor,  # [B]
) -> Dict[str, List[float]]:
    """Return lists of per-sample MAE, RMSE, Pearson-ρ, Spearman-ρ."""
    mae_list, rmse_list, pearson_list, spearman_list = [], [], [], []
    for b in range(pred.shape[0]):
        kv = int(k_valid[b].item())
        if kv == 0:
            continue
        p = pred[b, :kv].cpu().float().numpy()
        g = gt[b,   :kv].cpu().float().numpy()
        ae   = np.abs(p - g)
        mae_list.append(float(ae.mean()))
        rmse_list.append(float(np.sqrt((ae ** 2).mean())))
        if _SCIPY_OK and kv >= 2:
            rp, _ = pearsonr(p, g)
            rs, _ = spearmanr(p, g)
            pearson_list.append(float(rp) if not math.isnan(rp) else 0.0)
            spearman_list.append(float(rs) if not math.isnan(rs) else 0.0)
    return {
        "mae":      mae_list,
        "rmse":     rmse_list,
        "pearson":  pearson_list,
        "spearman": spearman_list,
    }


def _per_sample_node_metrics(
    node_mask: torch.Tensor,  # [B, K]  predicted (probabilities)
    gt_e3d:    torch.Tensor,  # [B, K]  GT e3d
    k_valid:   torch.Tensor,  # [B]
    threshold: float = 0.5,
) -> Dict[str, List[float]]:
    """
    GT node label = 1 if e3d > median(e3d) for that sample.
    Threshold node_mask at `threshold` to get binary predictions.
    """
    prec_list, rec_list, f1_list = [], [], []
    for b in range(node_mask.shape[0]):
        kv = int(k_valid[b].item())
        if kv == 0:
            continue
        nm  = node_mask[b, :kv].cpu().float().numpy()
        g   = gt_e3d[b,    :kv].cpu().float().numpy()
        med = float(np.median(g))
        label = (g > med).astype(np.float32)
        pred  = (nm >= threshold).astype(np.float32)
        tp = float((pred * label).sum())
        fp = float((pred * (1 - label)).sum())
        fn = float(((1 - pred) * label).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
    return {"precision": prec_list, "recall": rec_list, "f1": f1_list}


def _angular_errors(
    dir_pred:     torch.Tensor,  # [B, 2]
    instructions: List[str],
    device:       torch.device,
) -> List[float]:
    """
    Compute angular error in degrees between predicted and prior direction.
    Only for samples that contain direction words.
    """
    gt, has_dir = extract_direction_gt(instructions, device)
    if gt is None or not has_dir.any():
        return []
    errors = []
    for b in range(dir_pred.shape[0]):
        if not has_dir[b]:
            continue
        p = F.normalize(dir_pred[b:b+1], dim=-1)
        g = F.normalize(gt[b:b+1],       dim=-1)
        cos_sim = (p * g).sum().clamp(-1.0, 1.0)
        deg = math.degrees(math.acos(float(cos_sim.cpu())))
        errors.append(deg)
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: LangTopoSeg,
    loader: DataLoader,
    cfg: LangTopoSegConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run model over `loader` and return a flat dict of averaged metrics.
    """
    model.eval()

    accum: Dict[str, List[float]] = {
        "mae": [], "rmse": [], "pearson": [], "spearman": [],
        "precision": [], "recall": [], "f1": [],
        "angular_error_deg": [],
    }

    for batch in loader:
        rgb          = batch["rgb"].to(device, non_blocking=True)
        masks        = batch["masks"].to(device, non_blocking=True)
        centroids    = batch["centroids"].to(device, non_blocking=True)
        areas        = batch["areas"].to(device, non_blocking=True)
        k_valid      = batch["k_valid"].to(device, non_blocking=True)
        instructions = batch["instruction"]

        outputs = model(
            rgb=rgb, masks=masks, centroids=centroids,
            areas=areas, k_valid=k_valid, instructions=instructions,
        )

        pred_e3d   = outputs["pred_e3d"]    # [B, K]
        node_mask  = outputs["node_mask"]   # [B, K]
        dir_2d     = outputs["dir_2d"]      # [B, 2]

        gt_e3d  = batch["e3d_gt"][:, -1, :].to(device)   # last frame [B, K]
        kv_last = batch["k_valid"][:, -1].to(device)       # [B]

        # e3d metrics
        m = _per_sample_e3d_metrics(pred_e3d, gt_e3d, kv_last)
        for k, lst in m.items():
            accum[k].extend(lst)

        # Node selection metrics
        n = _per_sample_node_metrics(node_mask, gt_e3d, kv_last)
        for k, lst in n.items():
            accum[k].extend(lst)

        # Direction metrics
        ang = _angular_errors(dir_2d, instructions, device)
        accum["angular_error_deg"].extend(ang)

    results = {}
    for k, lst in accum.items():
        results[k] = float(np.mean(lst)) if lst else float("nan")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate LangTopoSeg")
    p.add_argument("--ckpt_path",  required=True,  help="Path to .pt checkpoint")
    p.add_argument("--data_root",  default=None,   help="Override data_root in config")
    p.add_argument("--split",      default="val",  help="Dataset split to evaluate on")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--output_csv", default=None,   help="Optional path to save results CSV")
    return p.parse_args()


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    args = _parse_args()
    cfg  = LangTopoSegConfig()
    cfg.ckpt_path = args.ckpt_path
    if args.data_root:
        cfg.data_root = args.data_root
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = LangTopoSeg(cfg).to(device)
    ckpt  = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint from {args.ckpt_path}  "
                f"(trained epoch {ckpt.get('epoch', '?')})")

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds = LangTopoDataset(
        data_root    = cfg.data_root,
        split        = args.split,
        n_frames     = cfg.n_frames,
        max_instances= cfg.max_instances,
        image_h      = cfg.image_h,
        image_w      = cfg.image_w,
        augment      = False,
    )
    loader = DataLoader(
        ds,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    results = evaluate(model, loader, cfg, device)

    header = "─" * 55
    logger.info(header)
    logger.info(f"  Evaluation on split='{args.split}'  ({len(ds)} windows)")
    logger.info(header)
    for k, v in results.items():
        logger.info(f"  {k:<25s} {v:.4f}")
    logger.info(header)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "value"])
            w.writeheader()
            for k, v in results.items():
                w.writerow({"metric": k, "value": v})
        logger.info(f"Results saved → {args.output_csv}")

    return results


if __name__ == "__main__":
    main()
