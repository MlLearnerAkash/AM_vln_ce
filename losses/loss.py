import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geodesic distance losses
# ---------------------------------------------------------------------------

def berhu_loss(pred, target, valid_mask=None, threshold_ratio=0.2):
    """
    BerHu (reverse Huber) loss — standard for dense depth/distance regression.
    Uses L1 for small residuals and L2 for large ones.

    Args:
        pred:   (B, 1, H, W) predicted geodesic distance map in [0, 1]
        target: (B, 1, H, W) ground truth geodesic distance map in [0, 1]
        valid_mask: (B, 1, H, W) bool/float, 1 = pixel belongs to a valid object
                    (i.e. not background).  If None, all pixels are used.
        threshold_ratio: c = threshold_ratio * max(|residual|) — controls the
                         L1/L2 cross-over point.  0.2 is the standard choice.

    Returns:
        scalar loss
    """
    residual = torch.abs(pred - target)  # (B, 1, H, W)

    if valid_mask is not None:
        residual = residual * valid_mask.float()
        denom = valid_mask.float().sum().clamp(min=1.0)
    else:
        denom = torch.tensor(residual.numel(), dtype=residual.dtype, device=residual.device)

    # Adaptive threshold
    with torch.no_grad():
        c = threshold_ratio * residual.detach().max()

    # BerHu: L1 where |r| <= c, L2 otherwise
    l1_part = residual
    l2_part = (residual ** 2 + c ** 2) / (2.0 * c.clamp(min=1e-8))

    loss_map = torch.where(residual <= c, l1_part, l2_part)

    if valid_mask is not None:
        loss_map = loss_map * valid_mask.float()

    return loss_map.sum() / denom


def ordinal_ranking_loss(pred, target, semantic_map, background_id=0, num_sample_pairs=2048):
    """
    Pairwise ordinal ranking loss: if object A is geodesically closer than
    object B in the GT, the model should also predict a lower distance for A.

    Operates per image in the batch by sampling random pixel pairs from
    *different* object regions and enforcing the correct ordering.

    Args:
        pred:         (B, 1, H, W) predicted distance map in [0, 1]
        target:       (B, 1, H, W) ground truth distance map in [0, 1]
        semantic_map: (B, 1, H, W) integer object-ID map (background == background_id)
        background_id: ID value used for background pixels (default 0 after normalisation)
        num_sample_pairs: number of pixel pairs to sample per image in the batch

    Returns:
        scalar loss
    """
    B, _, H, W = pred.shape
    total_loss = torch.tensor(0.0, device=pred.device)
    count = 0

    for b in range(B):
        sem = semantic_map[b, 0]           # (H, W)  integer ids
        p   = pred[b, 0]                   # (H, W)
        t   = target[b, 0]                 # (H, W)

        # Valid foreground pixels only
        valid = (sem != background_id)
        valid_idx = valid.nonzero(as_tuple=False)  # (N, 2)

        if valid_idx.shape[0] < 2:
            continue

        # Sample pairs — clamp to available pixels
        n_pairs = min(num_sample_pairs, valid_idx.shape[0] // 2)
        idx = torch.randperm(valid_idx.shape[0], device=pred.device)[:n_pairs * 2]
        idx_a, idx_b = idx[:n_pairs], idx[n_pairs:]

        pos_a = valid_idx[idx_a]  # (P, 2)
        pos_b = valid_idx[idx_b]  # (P, 2)

        t_a = t[pos_a[:, 0], pos_a[:, 1]]  # GT distances at A
        t_b = t[pos_b[:, 0], pos_b[:, 1]]  # GT distances at B
        p_a = p[pos_a[:, 0], pos_a[:, 1]]  # Pred distances at A
        p_b = p[pos_b[:, 0], pos_b[:, 1]]  # Pred distances at B

        # Only penalise pairs where the GT ordering is clear (margin > eps)
        delta_t = t_a - t_b  # positive → A is farther than B in GT
        margin = 0.05
        mask_ab = (delta_t >  margin).float()  # A farther → pred should have p_a > p_b
        mask_ba = (delta_t < -margin).float()  # B farther → pred should have p_b > p_a

        # Hinge loss: max(0, margin - (p_far - p_near))
        loss_ab = torch.clamp(margin - (p_a - p_b), min=0.0) * mask_ab
        loss_ba = torch.clamp(margin - (p_b - p_a), min=0.0) * mask_ba

        n = (mask_ab + mask_ba).sum().clamp(min=1.0)
        total_loss = total_loss + (loss_ab + loss_ba).sum() / n
        count += 1

    return total_loss / max(count, 1)


def geodesic_distance_loss(pred, target, semantic_map,
                           background_id=0,
                           berhu_weight=1.0,
                           ordinal_weight=0.3,
                           num_sample_pairs=2048):
    """
    Combined loss for geodesic distance map prediction.

    Args:
        pred:         (B, 1, H, W) predicted distance map in [0, 1]
        target:       (B, 1, H, W) ground truth geodesic distance map in [0, 1]
        semantic_map: (B, 1, H, W) integer semantic IDs — used to build the
                      valid-pixel mask and for ordinal sampling.
        background_id: background pixel value in semantic_map (default 0)
        berhu_weight: weight for BerHu loss
        ordinal_weight: weight for ordinal ranking loss
        num_sample_pairs: pairs to sample for ordinal loss

    Returns:
        total_loss, berhu_l, ordinal_l
    """
    # Valid foreground mask
    valid_mask = (semantic_map != background_id).float()  # (B, 1, H, W)

    berhu_l   = berhu_loss(pred, target, valid_mask=valid_mask)
    ordinal_l = ordinal_ranking_loss(pred, target, semantic_map,
                                     background_id=background_id,
                                     num_sample_pairs=num_sample_pairs)

    total_loss = berhu_weight * berhu_l + ordinal_weight * ordinal_l
    return total_loss, berhu_l, ordinal_l


# ---------------------------------------------------------------------------
# Original navigation cost-map losses (kept for backward compatibility)
# ---------------------------------------------------------------------------

def compute_gradients(cost_map):
    """
    Compute spatial gradients (∂C/∂x, ∂C/∂y)
    cost_map: (B, 1, H, W)
    Returns: (B, 2, H, W) where channel 0 is ∂x, channel 1 is ∂y
    """
    # Sobel-like gradient computation
    grad_x = cost_map[:, :, :, 1:] - cost_map[:, :, :, :-1]  # (B, 1, H, W-1)
    grad_y = cost_map[:, :, 1:, :] - cost_map[:, :, :-1, :]  # (B, 1, H-1, W)
    
    # Pad to match original size
    grad_x = F.pad(grad_x, (0, 1, 0, 0))  # (B, 1, H, W)
    grad_y = F.pad(grad_y, (0, 0, 0, 1))  # (B, 1, H, W)
    
    # Stack to (B, 2, H, W)
    gradients = torch.cat([grad_x, grad_y], dim=1)
    return gradients


def cost_map_loss(pred, target, occupancy_mask=None):
    """
    L_cost: L1 loss over navigable area
    
    Args:
        pred: (B, 1, H, W) predicted cost map in [0, 1]
        target: (B, 1, H, W) ground truth cost map in [0, 1]
        occupancy_mask: (B, 1, H, W) where 1=occupied, 0=navigable
                        If None, use all pixels
    
    Returns:
        scalar loss
    """
    if occupancy_mask is None:
        occupancy_mask = torch.zeros_like(pred)
    
    # Navigable area mask (1 - occupied)
    navigable_mask = 1 - occupancy_mask
    
    # L1 loss weighted by navigable area
    diff = torch.abs(pred - target)
    weighted_diff = diff * navigable_mask
    
    # Average over valid pixels
    loss = weighted_diff.sum() / (navigable_mask.sum() + 1e-8)
    
    return loss


def gradient_direction_loss(pred, target, occupancy_mask=None):
    """
    L_dir: Gradient direction consistency loss
    
    Args:
        pred: (B, 1, H, W) predicted cost map
        target: (B, 1, H, W) ground truth cost map
        occupancy_mask: (B, 1, H, W) where 1=occupied, 0=navigable
    
    Returns:
        scalar loss
    """
    if occupancy_mask is None:
        occupancy_mask = torch.zeros_like(pred)
    
    navigable_mask = 1 - occupancy_mask
    
    # Compute gradients (B, 2, H, W)
    grad_pred = compute_gradients(pred)
    grad_target = compute_gradients(target)
    
    # Compute magnitudes
    mag_pred = torch.sqrt((grad_pred ** 2).sum(dim=1, keepdim=True) + 1e-8)  # (B, 1, H, W)
    mag_target = torch.sqrt((grad_target ** 2).sum(dim=1, keepdim=True) + 1e-8)
    
    # Normalize gradients
    grad_pred_norm = grad_pred / (mag_pred + 1e-8)
    grad_target_norm = grad_target / (mag_target + 1e-8)
    
    # Cosine similarity (dot product of normalized gradients)
    # grad_pred_norm: (B, 2, H, W), grad_target_norm: (B, 2, H, W)
    cosine_sim = (grad_pred_norm * grad_target_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Loss: 1 - cosine_similarity (range [0, 2], 0 is best)
    direction_loss = 1 - cosine_sim
    
    # Weight by navigable mask
    weighted_loss = direction_loss * navigable_mask
    
    # Average over valid pixels
    loss = weighted_loss.sum() / (navigable_mask.sum() + 1e-8)
    
    return loss


def combined_navigation_loss(pred, target, occupancy_mask=None, 
                              cost_weight=1.0, dir_weight=0.5):
    """
    Combined loss for navigation cost map prediction
    
    Args:
        pred: (B, 1, H, W) predicted cost map
        target: (B, 1, H, W) ground truth cost map
        occupancy_mask: (B, 1, H, W) optional occupancy mask
        cost_weight: weight for cost map loss
        dir_weight: weight for gradient direction loss
    
    Returns:
        total_loss, cost_loss, dir_loss
    """
    cost_loss = cost_map_loss(pred, target, occupancy_mask)
    dir_loss = gradient_direction_loss(pred, target, occupancy_mask)
    
    total_loss = cost_weight * cost_loss + dir_weight * dir_loss
    
    return total_loss, cost_loss, dir_loss