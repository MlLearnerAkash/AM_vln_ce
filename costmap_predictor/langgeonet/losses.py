"""
Loss functions for LangGeoNet.

Combined loss:
    L_total = L_regression + lambda_rank * L_ranking + lambda_si * L_scale_invariant

    1. Geodesic Regression Loss   (SmoothL1) - primary distance supervision
    2. Ordinal Ranking Loss       (margin)   - correct ordering of objects
    3. Scale-Invariant Log Loss   (Eigen)    - handles varying scene scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicRegressionLoss(nn.Module):
    """SmoothL1 between predicted and GT normalized geodesic distances."""

    def __init__(self, beta=0.05):
        super().__init__()
        self.beta = beta

    def forward(self, predictions, targets):
        total_loss = 0.0
        total_objects = 0

        for pred, gt in zip(predictions, targets):
            if pred.numel() == 0:
                continue
            total_loss += F.smooth_l1_loss(pred, gt, beta=self.beta, reduction='sum')
            total_objects += pred.numel()

        if total_objects == 0:
            return torch.tensor(0.0, device=predictions[0].device, requires_grad=True)
        return total_loss / total_objects


class OrdinalRankingLoss(nn.Module):
    """
    For all pairs (i,j) where GT d_i < d_j, enforce pred_i < pred_j with margin.
    Uses hard negative mining for efficiency.
    """

    def __init__(self, margin=0.05, hard_fraction=0.5):
        super().__init__()
        self.margin = margin
        self.hard_fraction = hard_fraction

    def forward(self, predictions, targets):
        total_loss = 0.0
        total_pairs = 0

        for pred, gt in zip(predictions, targets):
            K = pred.shape[0]
            if K < 2:
                continue

            # Pairwise differences
            pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)   # [K,K]: pred_i - pred_j
            gt_diff = gt.unsqueeze(1) - gt.unsqueeze(0)         # [K,K]: gt_i - gt_j
            valid = (gt_diff < -1e-6)  # gt_i < gt_j -> pred_i should be < pred_j

            if valid.sum() == 0:
                continue

            # Margin loss: max(0, pred_i - pred_j + margin) where gt_i < gt_j
            pair_losses = F.relu(pred_diff + self.margin)
            valid_losses = pair_losses[valid]

            violated = valid_losses[valid_losses > 0]
            if violated.numel() == 0:
                continue
            # Hard negative mining: topk over violated pairs only
            n_hard = max(1, int(violated.numel() * self.hard_fraction))
            if violated.numel() > n_hard:
                topk, _ = violated.topk(n_hard)
                total_loss += topk.sum()
                total_pairs += n_hard
            else:
                total_loss += violated.sum()
                total_pairs += violated.numel()

        if total_pairs == 0:
            return torch.tensor(0.0, device=predictions[0].device, requires_grad=True)
        return total_loss / total_pairs


class ScaleInvariantLogLoss(nn.Module):
    """
    Scale-invariant loss in log space (Eigen et al., 2014).
    Allows uniform scale shift without penalty.
    """

    def __init__(self, variance_weight=0.5, eps=1e-6):
        super().__init__()
        self.lam = variance_weight
        self.eps = eps

    def forward(self, predictions, targets):
        total_loss = 0.0
        count = 0

        for pred, gt in zip(predictions, targets):
            if pred.numel() == 0:
                continue

            log_pred = torch.log(pred.clamp(min=self.eps))
            log_gt = torch.log(gt.clamp(min=self.eps))
            diff = log_pred - log_gt

            total_loss += (diff ** 2).mean() - self.lam * (diff.mean()) ** 2
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=predictions[0].device, requires_grad=True)
        return total_loss / count


class LangGeoNetLoss(nn.Module):
    """
    Combined loss:
        L = L_reg + lambda_rank * L_rank + lambda_si * L_si
    """

    def __init__(self, lambda_rank=0.5, lambda_si=0.3):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.lambda_si = lambda_si
        self.regression = GeodesicRegressionLoss()
        self.ranking = OrdinalRankingLoss()
        self.scale_inv = ScaleInvariantLogLoss()

    def forward(self, predictions, targets):
        l_reg = self.regression(predictions, targets)
        l_rank = self.ranking(predictions, targets)
        l_si = self.scale_inv(predictions, targets)

        total = l_reg + self.lambda_rank * l_rank + self.lambda_si * l_si

        loss_dict = {
            "loss_total": total.item(),
            "loss_regression": l_reg.item(),
            "loss_ranking": l_rank.item(),
            "loss_scale_invariant": l_si.item(),
        }
        return total, loss_dict
