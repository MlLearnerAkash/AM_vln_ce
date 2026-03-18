"""
Loss functions for LangTopoSeg training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


DIRECTION_PRIORS = {
    "left":    [-1.0,  0.0],
    "right":   [ 1.0,  0.0],
    "up":      [ 0.0, -1.0],
    "down":    [ 0.0,  1.0],
}


def extract_direction_gt(instructions: list, device: torch.device) -> Optional[torch.Tensor]:
    """
    Parse direction words from each instruction and return mean direction vector.
    Returns None if no direction words found.

    Returns : [B, 2] if any direction found, else None
    """
    B = len(instructions)
    gt = torch.zeros(B, 2, device=device)
    has_dir = torch.zeros(B, dtype=torch.bool, device=device)

    for b, instr in enumerate(instructions):
        words = instr.lower().split()
        vecs  = []
        for w in words:
            if w in DIRECTION_PRIORS:
                vecs.append(DIRECTION_PRIORS[w])
        if vecs:
            v = torch.tensor(vecs, dtype=torch.float32, device=device).mean(0)
            v = F.normalize(v.unsqueeze(0), dim=-1).squeeze(0)
            gt[b] = v
            has_dir[b] = True

    if has_dir.any():
        return gt, has_dir
    return None, has_dir


def loss_observed_e3d(
    pred_e3d: torch.Tensor,   # [B, K]
    gt_e3d:   torch.Tensor,   # [B, K]   last frame target
    k_valid:  torch.Tensor,   # [B]      number of valid instances
) -> torch.Tensor:
    """MSE loss over valid (non-padded) instances of the current frame."""
    B, K = pred_e3d.shape
    mask = torch.zeros_like(pred_e3d, dtype=torch.bool)
    for b in range(B):
        mask[b, :k_valid[b]] = True
    if mask.sum() == 0:
        return pred_e3d.sum() * 0.0
    return F.mse_loss(pred_e3d[mask], gt_e3d[mask])


def loss_ranking(
    pred_e3d: torch.Tensor,  # [B, K]
    gt_e3d:   torch.Tensor,  # [B, K]
    k_valid:  torch.Tensor,  # [B]
    margin:   float = 0.05,
) -> torch.Tensor:
    """
    Pairwise ranking loss: for all ordered pairs (i,j) where gt_i < gt_j,
    penalise pred_i >= pred_j - margin.
    """
    total_loss = 0.0
    count = 0
    B, K = pred_e3d.shape
    for b in range(B):
        kv = k_valid[b].item()
        if kv < 2:
            continue
        p = pred_e3d[b, :kv]    # [kv]
        g = gt_e3d[b, :kv]      # [kv]
        # All pairs (i, j) where g_i < g_j
        diff_g = g.unsqueeze(1) - g.unsqueeze(0)   # [kv, kv]
        should_be_less = diff_g < -1e-4             # i should score less than j
        diff_p = p.unsqueeze(1) - p.unsqueeze(0)   # p_i - p_j
        # Penalise when predicted ranking is wrong: max(0, p_i - p_j + margin)
        loss_ij = F.relu(diff_p + margin)
        total_loss = total_loss + (loss_ij * should_be_less.float()).sum()
        count += should_be_less.sum().item()
    if count == 0:
        return pred_e3d.sum() * 0.0
    return total_loss / count


def loss_node_selection(
    node_mask: torch.Tensor,  # [B, K]  predicted
    gt_e3d:    torch.Tensor,  # [B, K]  GT e3d (high value = salient node)
    k_valid:   torch.Tensor,  # [B]
) -> torch.Tensor:
    """
    BCE loss: nodes with above-median e3d (far objects, thus salient for navigation)
    should be selected.  gt_node = (e3d > median) for each sample.
    """
    B, K = node_mask.shape
    mask = torch.zeros_like(node_mask, dtype=torch.bool)
    gt   = torch.zeros_like(gt_e3d)
    for b in range(B):
        kv = int(k_valid[b].item())
        mask[b, :kv] = True
        if kv > 0:
            med = gt_e3d[b, :kv].median()
            gt[b, :kv] = (gt_e3d[b, :kv] > med).float()
    if mask.sum() == 0:
        return node_mask.sum() * 0.0
    # Clamp to valid BCE range; guards against float32 precision drift and NaN
    return F.binary_cross_entropy(node_mask[mask].clamp(1e-7, 1.0 - 1e-7), gt[mask])


def loss_direction(
    dir_pred:  torch.Tensor,   # [B, 2]
    instructions: list,
    device: torch.device,
) -> torch.Tensor:
    """MSE between predicted and prior direction vectors (only for samples with direction words)."""
    gt, has_dir = extract_direction_gt(instructions, device)
    if gt is None or not has_dir.any():
        return dir_pred.sum() * 0.0
    return F.mse_loss(dir_pred[has_dir], gt[has_dir])


def loss_symmetry(pred_edges: torch.Tensor) -> torch.Tensor:
    """Penalise asymmetry in predicted edge matrix."""
    return (pred_edges - pred_edges.transpose(-1, -2)).pow(2).mean()


def compute_total_loss(
    outputs:      Dict,
    batch:        Dict,
    cfg,
    device:       torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute all losses and return dict of scalars + total."""
    pred_e3d    = outputs["pred_e3d"]         # [B, K]
    pred_edges  = outputs["pred_edges"]       # [B, K, K]
    node_mask   = outputs["node_mask"]        # [B, K]
    dir_2d      = outputs["dir_2d"]           # [B, 2]

    gt_e3d      = batch["e3d_gt"][:, -1, :].to(device)   # last frame GT  [B, K]
    k_valid     = batch["k_valid"][:, -1].to(device)      # last frame valid K
    instructions = batch["instruction"]

    L_obs  = loss_observed_e3d(pred_e3d, gt_e3d, k_valid)
    L_rank = loss_ranking(pred_e3d, gt_e3d, k_valid, margin=cfg.rank_margin)
    L_node = loss_node_selection(node_mask, gt_e3d, k_valid)
    L_dir  = loss_direction(dir_2d, instructions, device)
    L_sym  = loss_symmetry(pred_edges)

    total = (
        cfg.lambda_obs  * L_obs  +
        cfg.lambda_rank * L_rank +
        cfg.lambda_node * L_node +
        cfg.lambda_dir  * L_dir  +
        cfg.lambda_sym  * L_sym
    )

    return {
        "total": total,
        "obs":   L_obs,
        "rank":  L_rank,
        "node":  L_node,
        "dir":   L_dir,
        "sym":   L_sym,
    }
