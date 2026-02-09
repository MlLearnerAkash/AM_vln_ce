import torch
import torch.nn.functional as F

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