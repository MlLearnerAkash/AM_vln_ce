"""
Loss functions for LangGeoNetV3.

Combined loss
─────────────
    L = L_regression  +  λ_rank · L_bt_ranking  +  λ_div · L_diversity

Changes from V2
───────────────
  1. Bradley-Terry soft ranking replaces margin/hinge ranking:
       L_bt = -log σ(pred_j − pred_i)   for all pairs where gt_j > gt_i + ε
     No dead zone → non-zero gradient for EVERY valid pair regardless of
     how well-ordered the predictions already are.

  2. Diversity penalty (anti-collapse) replaces the numerically-unstable
     ScaleInvariant log loss:
       L_div = mean_b( max(0, min_std − std(predictions_b))² )
     If all objects in image b get the same predicted cost, std → 0 and the
     penalty is large.  This directly punishes the collapse mode.
     min_std=0.12 means we ask for at least 0.12 spread in [0,1] predictions.

  3. The old OrdinalRankingLoss and ScaleInvariantLogLoss are kept for
     backwards compatibility but are NOT used by the default LangGeoNetLoss.

API is fully backward-compatible with train.py. λ_si is accepted but ignored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicRegressionLoss(nn.Module):
    """SmoothL1 between predicted and GT normalised geodesic costs."""

    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta = beta

    def forward(self, predictions, targets):
        total, n = 0.0, 0
        for pred, gt in zip(predictions, targets):
            if pred.numel() == 0:
                continue
            # Filter out NaN GT values that arrive from empty path_rows
            valid = torch.isfinite(gt)
            if not valid.any():
                continue
            total += F.smooth_l1_loss(pred[valid], gt[valid],
                                      beta=self.beta, reduction='sum')
            n += valid.sum().item()
        if n == 0:
            return _zero(predictions)
        return total / n


class BradleyTerryRankingLoss(nn.Module):
    """
    Soft pairwise ranking via the Bradley-Terry model.

    For every pair (i, j) in the same image where  gt_j > gt_i + threshold:
        L = -log σ(pred_j − pred_i) = softplus(pred_i − pred_j)

    Advantages over margin / hinge ranking:
      • Gradient is non-zero for ALL valid pairs (no dead zone).
      • Gradient magnitude = σ(pred_i − pred_j): large when i is ranked above
        j (wrong), small when j is already ranked above i (correct) — natural
        curriculum without any hard-mining heuristics.
    """

    def __init__(self, gt_threshold: float = 1e-4):
        super().__init__()
        self.gt_threshold = gt_threshold

    def forward(self, predictions, targets):
        total, n = 0.0, 0
        for pred, gt in zip(predictions, targets):
            K = pred.shape[0]
            if K < 2:
                continue
            valid_gt = torch.isfinite(gt)
            if valid_gt.sum() < 2:
                continue
            p = pred[valid_gt]
            g = gt[valid_gt]

            # [K', K'] pairwise: valid[i,j] = gt_j > gt_i + threshold
            valid_pairs = (g.unsqueeze(0) - g.unsqueeze(1)) > self.gt_threshold

            if valid_pairs.sum() == 0:
                continue

            # softplus(-diff) = -log σ(diff)  where diff = pred_j - pred_i
            diff = p.unsqueeze(0) - p.unsqueeze(1)   # [K', K']: pred_j - pred_i
            pair_losses = F.softplus(-diff)            # -log σ(diff)

            total += pair_losses[valid_pairs].sum()
            n     += int(valid_pairs.sum().item())

        if n == 0:
            return _zero(predictions)
        return total / n


class DiversityPenaltyLoss(nn.Module):
    """
    Anti-collapse regulariser.

    Penalises images where the predicted cost standard deviation is below
    min_std.  Even a trivially small spread (min_std = 0.12 in [0,1]) stops
    the model from converging to a constant-prediction degenerate solution.

        L_div = mean_b( relu(min_std − std(pred_b))² )
    """

    def __init__(self, min_std: float = 0.12):
        super().__init__()
        self.min_std = min_std

    def forward(self, predictions):
        total, n = 0.0, 0
        for pred in predictions:
            if pred.numel() < 2:
                continue
            penalty = F.relu(self.min_std - pred.std()) ** 2
            total  += penalty
            n      += 1
        if n == 0:
            return _zero(predictions)
        return total / n


# ── Backward-compatible classes (not used by default) ─────────────────────────

class OrdinalRankingLoss(nn.Module):
    """Kept for API compatibility — use BradleyTerryRankingLoss instead."""

    def __init__(self, margin=0.05, hard_fraction=0.5):
        super().__init__()
        self.margin = margin
        self.hard_fraction = hard_fraction

    def forward(self, predictions, targets):
        total, n = 0.0, 0
        for pred, gt in zip(predictions, targets):
            K = pred.shape[0]
            if K < 2:
                continue
            pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
            gt_diff   = gt.unsqueeze(1)   - gt.unsqueeze(0)
            valid     = gt_diff < -1e-6
            if not valid.any():
                continue
            pair_losses = F.relu(pred_diff + self.margin)[valid]
            if pair_losses.numel() == 0:
                continue
            n_hard = max(1, int(pair_losses.numel() * self.hard_fraction))
            if pair_losses.numel() > n_hard:
                pair_losses, _ = pair_losses.topk(n_hard)
            total += pair_losses.sum()
            n     += pair_losses.numel()
        if n == 0:
            return _zero(predictions)
        return total / n


class ScaleInvariantLogLoss(nn.Module):
    """Kept for API compatibility — no longer used due to numeric instability."""

    def __init__(self, variance_weight=0.5, eps=1e-6):
        super().__init__()
        self.lam = variance_weight
        self.eps = eps

    def forward(self, predictions, targets):
        total, count = 0.0, 0
        for pred, gt in zip(predictions, targets):
            if pred.numel() == 0:
                continue
            log_pred = torch.log(pred.clamp(min=self.eps))
            log_gt   = torch.log(gt.clamp(min=self.eps))
            diff     = log_pred - log_gt
            total   += (diff ** 2).mean() - self.lam * (diff.mean()) ** 2
            count   += 1
        if count == 0:
            return _zero(predictions)
        return total / count


# ── Combined loss ──────────────────────────────────────────────────────────────

class LangGeoNetLoss(nn.Module):
    """
    L = L_regression  +  λ_rank · L_bt_ranking  +  λ_div · L_diversity

    λ_si is accepted for backward API compatibility but ignored.
    Recommended defaults: λ_rank=2.0, λ_div=1.0
    """

    def __init__(self,
                 lambda_rank: float = 2.0,
                 lambda_si: float = 0.0,      # kept for API compat, ignored
                 lambda_div: float = 1.0):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.lambda_div  = lambda_div
        self.regression  = GeodesicRegressionLoss(beta=0.05)
        self.ranking     = BradleyTerryRankingLoss(gt_threshold=1e-4)
        self.diversity   = DiversityPenaltyLoss(min_std=0.12)

    def forward(self, predictions, targets):
        l_reg  = self.regression(predictions, targets)
        l_rank = self.ranking(predictions, targets)
        l_div  = self.diversity(predictions)

        total  = l_reg + self.lambda_rank * l_rank + self.lambda_div * l_div

        return total, {
            "loss_total":           total.item(),
            "loss_regression":      l_reg.item(),
            "loss_ranking":         l_rank.item(),
            "loss_scale_invariant": 0.0,          # API compat key
            "loss_diversity":       l_div.item(),
        }


# ── Helper ─────────────────────────────────────────────────────────────────────

def _zero(predictions):
    """Return a zero tensor on the correct device with requires_grad=True."""
    for p in predictions:
        if p.numel() > 0:
            return p.sum() * 0.0
    return torch.tensor(0.0, requires_grad=True)
