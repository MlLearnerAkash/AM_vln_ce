#----------------------original code --------------------------------------------
"""
Training loop for LangGeoNet — H5EpisodePathLengthsDataset edition.
Supports both the CLIP+DINOv2 model (LangGeoNet) and the Qwen2-VL backbone
model (VLMLangGeoNet).  Pass --use_vlm to activate the VLM path.
"""
from __future__ import annotations
import os
import time
import json
import logging
import random as _random
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr

from model import build_langgeonet, build_vlm_langgeonet
from dataset import H5EpisodePathLengthsDataset, create_h5_episode_pathlengths_dataloader
from losses import LangGeoNetLoss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Batch adaptor  —  bridges dataloader → model interface
# -------------------------------------------------------

def _prepare_batch(batch: dict, device: torch.device):
    """
    Convert a raw H5EpisodePathLengthsDataset batch into the tensors the
    model and loss expect.
    """
    pixel_values = batch["pixel_values"].to(device)
    input_ids  = batch["input_ids"].to(device)
    attn_mask  = batch["attention_mask"].to(device)

    masks_list, gts_list = [], []
    for registry in batch["node_registries"]:
        if not registry:
            masks_list.append(torch.zeros(0, 1, 1, dtype=torch.bool, device=device))
            gts_list.append(torch.zeros(0, dtype=torch.float32, device=device))
            continue

        node_ids   = list(registry.keys())
        raw_costs  = []
        masks      = []

        for node_id in node_ids:
            entry = registry[node_id]

            pr   = np.asarray(entry.path_row, dtype=np.float64)
            cost = float(np.nanmean(pr)) if pr.size else np.nan
            raw_costs.append(cost)

            m = entry.mask
            try:
                m = m.cpu().numpy()
            except AttributeError:
                m = np.asarray(m)
            masks.append(m.astype(bool))

        # ── per-frame min-max normalise to [0, 1] ────────────────────────────
        finite  = [c for c in raw_costs if np.isfinite(c)]
        c_min, c_max = (min(finite), max(finite)) if len(finite) > 1 else (0.0, 1.0)

        costs = []
        for c in raw_costs:
            if not np.isfinite(c):
                costs.append(np.nan)
            elif c_max == c_min:
                costs.append(0.0)
            else:
                costs.append((c - c_min) / (c_max - c_min))

        masks_list.append(torch.from_numpy(np.stack(masks)).to(device))
        gts_list.append(
            torch.tensor(costs, dtype=torch.float32, device=device)
        )

    return pixel_values, input_ids, attn_mask, masks_list, gts_list


# -------------------------------------------------------
# Metrics / checkpointing helpers  (unchanged)
# -------------------------------------------------------

def compute_metrics(all_preds, all_gts):
    flat_p = np.concatenate(all_preds)
    flat_g = np.concatenate(all_gts)

    mae  = float(np.mean(np.abs(flat_p - flat_g)))
    mse  = float(np.mean((flat_p - flat_g) ** 2))
    rmse = float(np.sqrt(mse))

    valid = flat_g > 0.01
    mape  = float(np.mean(np.abs(flat_p[valid] - flat_g[valid]) / flat_g[valid])) if valid.sum() else 0.0

    rank_accs, spearman_corrs = [], []
    for p, g in zip(all_preds, all_gts):
        if len(p) < 2:
            continue
        correct = total = 0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if abs(g[i] - g[j]) < 1e-6:
                    continue
                total += 1
                correct += int((p[i] < p[j]) == (g[i] < g[j]))
        if total:
            rank_accs.append(correct / total)
        if len(p) >= 3:
            c, _ = spearmanr(p, g)
            if not np.isnan(c):
                spearman_corrs.append(c)

    thresh = {
        f"acc@{d}": float(np.mean(np.abs(flat_p - flat_g) < d))
        for d in [0.05, 0.10, 0.20]
    }
    return {
        "mae": mae, "mse": mse, "rmse": rmse, "mape": mape,
        "ranking_accuracy": float(np.mean(rank_accs))   if rank_accs   else 0.0,
        "spearman":         float(np.mean(spearman_corrs)) if spearman_corrs else 0.0,
        **thresh,
    }


def _extract_sample_masks_and_rgb(batch, sample_idx):
    """Return (frame_rgb, masks_arr, node_ids) for one sample in a collated batch.

    Handles both `node_registries` + `frame_node_ids` and legacy `masks_arrs` formats.
    """
    frame_rgb = batch.get("frame_rgbs", [None])[sample_idx]

    if "node_registries" in batch and "frame_node_ids" in batch:
        node_registry = batch["node_registries"][sample_idx]
        frame_node_ids = batch["frame_node_ids"][sample_idx]
        H, W = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
        if not frame_node_ids:
            return frame_rgb, np.zeros((0, H, W), dtype=bool), []
        masks = np.stack([node_registry[nid].mask for nid in frame_node_ids], axis=0)
        return frame_rgb, masks, frame_node_ids

    if "masks_arrs" in batch and "path_rows" in batch:
        masks = batch["masks_arrs"][sample_idx]
        # node ids may not be available in this legacy format
        return frame_rgb, masks, list(range(masks.shape[0]))

    # Fallback: empty
    H, W = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    return frame_rgb, np.zeros((0, H, W), dtype=bool), []


def _render_cost_comparison_image(frame_rgb, masks_arr, gt_costs, pred_costs, alpha=0.5):
    """Side-by-side GT | PRED overlay using overlay_costs style:
    green (low cost) → yellow → red (high cost), white contours, centroid labels."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    H, W = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    K = 0 if masks_arr is None else masks_arr.shape[0]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'green_red', ['#00cc44', '#ffff00', '#ff2200']   # green → yellow → red
    )

    def _make_overlay(costs):
        overlay      = frame_rgb.astype(np.float32) / 255.0
        cost_canvas  = np.full((H, W), np.nan, dtype=np.float32)
        segment_mask = np.zeros((H, W), dtype=bool)

        for k in range(K):
            m = masks_arr[k].astype(bool)
            if not m.any():
                continue
            c = float(costs[k])
            cost_canvas[m] = 0.5 if not np.isfinite(c) else c
            segment_mask  |= m

        if segment_mask.any():
            canvas_filled = np.where(np.isnan(cost_canvas), 0.5, cost_canvas)
            rgba_heat = np.zeros((H, W, 4), dtype=np.float32)
            # avoid boolean fancy-indexed in-place assignment (segfaults on old numpy)
            heat_vals = cmap(canvas_filled)                  # [H, W, 4] full array
            rgba_heat = np.where(segment_mask[:, :, None], heat_vals, rgba_heat)
            blended = (1 - alpha) * overlay + alpha * rgba_heat[:, :, :3]
            overlay = np.where(segment_mask[:, :, None], blended, overlay)

        # white contours
        contour_mask = np.zeros((H, W), dtype=bool)
        for k in range(K):
            m = masks_arr[k].astype(bool)
            if not m.any():
                continue
            pad   = np.pad(m.astype(np.uint8), 1, mode='constant')
            neigh = pad[:-2,1:-1] + pad[2:,1:-1] + pad[1:-1,:-2] + pad[1:-1,2:]
            contour_mask |= m & (neigh < 4)
        overlay = np.where(contour_mask[:, :, None], 1.0, overlay)

        return overlay

    gt_overlay = _make_overlay(gt_costs)
    pd_overlay = _make_overlay(pred_costs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, overlay, costs, title in [
        (axes[0], gt_overlay, gt_costs, "GT"),
        (axes[1], pd_overlay, pred_costs, "PRED"),
    ]:
        ax.imshow(np.clip(overlay, 0, 1))
        for k in range(K):
            m = masks_arr[k].astype(bool)
            if not m.any():
                continue
            ys, xs = np.where(m)
            cy, cx = int(ys.mean()), int(xs.mean())
            c = float(costs[k])
            label = f"{c:.2f}" if np.isfinite(c) else "∞"
            ax.text(cx, cy, label, fontsize=7, color='white', ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.4, lw=0))
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Normalised cost  (0 = low / green,  1 = high / red)', fontsize=8)
    fig.tight_layout()

    fig.canvas.draw()
    fig_w, fig_h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig_h, fig_w, 3).copy()
    plt.close(fig)
    return img


def _clip_lang_ln_norms(model: nn.Module) -> dict:
    """Return the L2 norm of every LayerNorm weight in the unfrozen CLIP text layers.

    Keys are prefixed with 'ln_norm/' so they group together in W&B.
    Covers layer_norm1 and layer_norm2 of the last 6 encoder layers plus
    the final_layer_norm of the text model.
    """
    norms = {}
    text = model.clip.text_model
    n_total = len(text.encoder.layers)
    for idx, layer in enumerate(text.encoder.layers[-6:]):
        abs_idx = n_total - 6 + idx
        for ln_attr in ("layer_norm1", "layer_norm2"):
            ln = getattr(layer, ln_attr)
            norms[f"ln_norm/text_l{abs_idx}_{ln_attr}"] = ln.weight.norm().item()
    norms["ln_norm/text_final"] = text.final_layer_norm.weight.norm().item()
    return norms


def _clip_lang_grad_norms(model: nn.Module) -> dict:
    """L2 norm of accumulated gradients on the unfrozen CLIP text LayerNorm weights.

    Call immediately after clip_grad_norm_ (before zero_grad) so gradients are intact.
    Returns an empty dict when no gradients exist yet.
    """
    norms = {}
    text = model.clip.text_model
    n_total = len(text.encoder.layers)
    for idx, layer in enumerate(text.encoder.layers[-6:]):
        abs_idx = n_total - 6 + idx
        for ln_attr in ("layer_norm1", "layer_norm2"):
            g = getattr(layer, ln_attr).weight.grad
            if g is not None:
                norms[f"grad_norm/text_l{abs_idx}_{ln_attr}"] = g.norm().item()
    g = text.final_layer_norm.weight.grad
    if g is not None:
        norms["grad_norm/text_final"] = g.norm().item()
    return norms


def _resume_from_checkpoint(resume, model, optimizer, scheduler, device):
    if not resume:
        return 0, float("inf")
    if not os.path.isfile(resume):
        raise FileNotFoundError(f"Checkpoint not found: {resume}")
    ckpt = torch.load(resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start = int(ckpt.get("epoch", 0))
    best  = float(ckpt.get("best_val_mae", float("inf")))
    logger.info(f"Resumed at epoch {start + 1}, best MAE={best:.4f}")
    return start, best


# -------------------------------------------------------
# VLM helpers
# -------------------------------------------------------

def _prepare_batch_vlm(batch: dict, device: torch.device):
    """Extract (frame_rgbs, instructions, masks_list_np, gts_list) for VLM training.

    Unlike _prepare_batch, masks_list contains numpy bool arrays [K,H,W]
    because VLMLangGeoNet._pool_objects expects numpy input.
    """
    frame_rgbs   = batch["frame_rgbs"]                          # list of np [H,W,3]
    instructions = batch.get("instructions", [""] * len(frame_rgbs))  # list of str

    masks_list, gts_list = [], []
    for registry in batch["node_registries"]:
        if not registry:
            masks_list.append(np.zeros((0, 1, 1), dtype=bool))
            gts_list.append(torch.zeros(0, dtype=torch.float32, device=device))
            continue

        node_ids  = list(registry.keys())
        raw_costs = []
        masks     = []
        for nid in node_ids:
            entry = registry[nid]
            pr   = np.asarray(entry.path_row, dtype=np.float64)
            cost = float(np.nanmean(pr)) if pr.size else np.nan
            raw_costs.append(cost)
            m = entry.mask
            try:    m = m.cpu().numpy()
            except: m = np.asarray(m)
            masks.append(m.astype(bool))

        finite   = [c for c in raw_costs if np.isfinite(c)]
        c_min, c_max = (min(finite), max(finite)) if len(finite) > 1 else (0.0, 1.0)
        costs = []
        for c in raw_costs:
            if   not np.isfinite(c):  costs.append(np.nan)
            elif c_max == c_min:       costs.append(0.0)
            else:                      costs.append((c - c_min) / (c_max - c_min))

        masks_list.append(np.stack(masks))
        gts_list.append(torch.tensor(costs, dtype=torch.float32, device=device))

    return frame_rgbs, instructions, masks_list, gts_list


def _load_all_instructions(h5_path: str) -> list[str]:
    """Load unique instruction strings from the HDF5 file."""
    instrs = []
    with h5py.File(h5_path, "r") as hf:
        for ep_key in hf.keys():
            ep = hf[ep_key]
            if "instruction" in ep:
                raw = ep["instruction"][()]
                instrs.append(raw.decode() if isinstance(raw, bytes) else str(raw))
    # Always have fallback negatives even for tiny subsets
    instrs += [
        "turn around and go back to where you started",
        "the weather is sunny and the birds are singing",
        "walk through the doorway and turn left at the hallway end",
        "proceed through the living room towards the kitchen",
        "exit the room through the door on the right side",
    ]
    return list(set(instrs))


def _vlm_ln_norms(model: nn.Module) -> dict:
    """L2 norms of trainable norm-layer weights in the VLM (for W&B logging)."""
    norms = {}
    vlm = model.vlm
    lm = vlm.model.language_model
    for idx, layer in enumerate(lm.layers):
        for pname, p in layer.named_parameters():
            if "norm" in pname.lower() and p.requires_grad:
                norms[f"ln_norm/lm_l{idx}_{pname}"] = p.norm().item()
    for pname, p in lm.norm.named_parameters():
        if p.requires_grad:
            norms[f"ln_norm/lm_outnorm_{pname}"] = p.norm().item()
    for pname, p in vlm.visual.merger.named_parameters():
        if p.requires_grad:
            norms[f"ln_norm/vis_merger_{pname}"] = p.norm().item()
    return norms


def _vlm_grad_norms(model: nn.Module) -> dict:
    """L2 norms of gradients on trainable norm weights (call before zero_grad)."""
    norms = {}
    vlm = model.vlm
    lm = vlm.model.language_model
    for idx, layer in enumerate(lm.layers):
        for pname, p in layer.named_parameters():
            if "norm" in pname.lower() and p.grad is not None:
                norms[f"grad_norm/lm_l{idx}_{pname}"] = p.grad.norm().item()
    for pname, p in lm.norm.named_parameters():
        if p.grad is not None:
            norms[f"grad_norm/lm_outnorm_{pname}"] = p.grad.norm().item()
    for pname, p in vlm.visual.merger.named_parameters():
        if p.grad is not None:
            norms[f"grad_norm/vis_merger_{pname}"] = p.grad.norm().item()
    return norms


def make_exp_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    n = 1
    while True:
        candidate = os.path.join(base_dir, f"exp{n}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        n += 1


# -------------------------------------------------------
# VLM train / validate
# -------------------------------------------------------

def train_one_epoch_vlm(model, loader, criterion, optimizer, device,
                        grad_accum=1, wandb_run=None, global_step=0,
                        all_instructions=None, lambda_contrast=2.0, margin=0.1):
    """One training epoch for VLMLangGeoNet.

    Processes each sample independently (one VLM forward per sample) to keep
    activation memory bounded.  Gradients accumulate across both the in-batch
    samples and `grad_accum` loader steps before each optimizer.step().

    Contrastive loss: for each sample, a negative instruction is drawn from
    other in-batch samples (shifted by 1) or from the global pool when
    batch_size=1.  neg_preds are always detached (no_grad) so gradients flow
    only through the positive prediction path.
    """
    model.train()
    losses = defaultdict(float)
    n       = 0
    optimizer.zero_grad()
    if all_instructions is None:
        all_instructions = []

    for i, batch in enumerate(loader):
        frame_rgbs, instructions, masks_list, gts_list = \
            _prepare_batch_vlm(batch, device)

        if all(m.shape[0] == 0 for m in masks_list):
            continue

        B = len(instructions)
        # Build per-sample negative instructions
        if B > 1:
            neg_instructions = [instructions[(j + 1) % B] for j in range(B)]
        else:
            neg_instructions = []
            for instr in instructions:
                neg = instr
                for _ in range(10):
                    cand = _random.choice(all_instructions) if all_instructions else ""
                    if cand != instr:
                        neg = cand
                        break
                neg_instructions.append(neg)

        batch_ld   = defaultdict(float)
        n_valid    = 0

        for j, (rgb, masks_np, instr, neg_instr, gt) in enumerate(
                zip(frame_rgbs, masks_list, instructions, neg_instructions, gts_list)):

            if masks_np.shape[0] == 0:
                continue
            valid = torch.isfinite(gt)
            if valid.sum() == 0:
                continue

            # Forward — correct instruction (with grad through unfrozen layers)
            preds_j, _ = model([rgb], [masks_np], [instr])
            loss_j, ld_j = criterion(preds_j, [gt])

            # Contrastive forward — different instruction, no grad
            with torch.no_grad():
                neg_preds_j, _ = model([rgb], [masks_np], [neg_instr])

            diff = (preds_j[0][valid] - neg_preds_j[0][valid]).abs().mean()
            contrast_j = F.relu(
                torch.tensor(margin, device=device) - diff
            )

            sample_loss = loss_j + lambda_contrast * contrast_j
            # Divide by grad_accum so effective gradient magnitude equals
            # one full un-accumulated step per grad_accum loader batches.
            (sample_loss / grad_accum).backward()

            for k, v in ld_j.items():
                batch_ld[k] += v
            batch_ld["loss_contrast"] += contrast_j.item()
            n_valid += 1
            del preds_j, neg_preds_j

        if n_valid == 0:
            continue

        # Normalize batch accumulation
        for k in batch_ld:
            batch_ld[k] /= n_valid
        batch_ld["loss_total"] = (
            batch_ld.get("loss_total", 0.0) + lambda_contrast * batch_ld["loss_contrast"]
        )

        if (i + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norms = _vlm_grad_norms(model)
            optimizer.step()
            optimizer.zero_grad()
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{k}": v for k, v in batch_ld.items()}
                    | grad_norms
                    | {"batch_step": global_step}
                )
            global_step += 1

        for k, v in batch_ld.items():
            losses[k] += v
        n += 1

        if (i + 1) % 50 == 0:
            logger.info(
                f"  batch {i+1}/{len(loader)}"
                f"  loss={losses.get('loss_total', 0)/n:.4f}"
                f"  contrast={losses.get('loss_contrast', 0)/n:.4f}"
            )

    return {k: v / max(n, 1) for k, v in losses.items()}, global_step


@torch.no_grad()
def validate_vlm(model, loader, criterion, device,
                 wandb_run=None, exp_dir=None, epoch=None, max_viz=8):
    """Validation epoch for VLMLangGeoNet."""
    model.eval()
    losses              = defaultdict(float)
    n                   = 0
    all_preds, all_gts  = [], []
    viz_count           = 0
    viz_images          = []

    for batch in loader:
        frame_rgbs, instructions, masks_list, gts_list = \
            _prepare_batch_vlm(batch, device)

        if all(m.shape[0] == 0 for m in masks_list):
            continue

        preds, _ = model(frame_rgbs, masks_list, instructions)
        _, ld    = criterion(preds, gts_list)

        for k, v in ld.items():
            losses[k] += v
        n += 1

        for p, g in zip(preds, gts_list):
            all_preds.append(p.cpu().numpy())
            all_gts.append(g.cpu().numpy())

        if viz_count < max_viz:
            for idx, (p, g) in enumerate(zip(preds, gts_list)):
                if viz_count >= max_viz:
                    break
                try:
                    frame_rgb  = batch["frame_rgbs"][idx]
                    registry   = batch["node_registries"][idx]
                    if not registry:
                        continue
                    node_ids   = list(registry.keys())
                    masks_arr  = np.stack([registry[nid].mask for nid in node_ids])
                    pred_np    = p.cpu().numpy()
                    gt_np      = g.cpu().numpy()
                    K = min(len(pred_np), len(gt_np), masks_arr.shape[0])
                    if K == 0:
                        continue
                    img = _render_cost_comparison_image(
                        frame_rgb, masks_arr[:K], gt_np[:K], pred_np[:K])
                    viz_images.append(img)
                    viz_count += 1
                except Exception as e:
                    logger.debug(f"Viz failed for sample {idx}: {e}")

    avg_losses = {k: v / max(n, 1) for k, v in losses.items()}
    metrics    = compute_metrics(all_preds, all_gts)
    metrics.update(avg_losses)

    if wandb_run is not None and viz_images:
        try:
            wb_imgs = [wandb.Image(img, caption=f"val_ep{epoch}_s{i}")
                       for i, img in enumerate(viz_images)]
            wandb_run.log({"viz/pred_vs_gt": wb_imgs, "epoch": epoch})
        except Exception:
            pass

    return metrics


# -------------------------------------------------------
# Train / validate  (CLIP+DINOv2 path — unchanged)
# -------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_accum=1, wandb_run=None, global_step=0,
                    lambda_lang=0.1, lambda_align=0.1):
    model.train()
    losses      = defaultdict(float)
    n           = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        pixel_values, input_ids, attn_mask, masks_list, gts_list = \
            _prepare_batch(batch, device)
        del batch 
        # skip frames where every registry was empty
        if all(m.shape[0] == 0 for m in masks_list):
            continue

        preds, aux   = model(pixel_values, masks_list, input_ids, attn_mask)
        lang_aux      = aux["lang_aux"]      # [B]
        align_scores  = aux["align_scores"]  # list[B] of [K_b]
        loss, ld = criterion(preds, gts_list)

        # Auxiliary language loss: Pearson correlation between lang_aux and the
        # per-sample frame-mean GT cost within the batch.
        #
        # Why Pearson instead of MSE:
        #   frame_mean_gt is a geometric quantity (same for any instruction given
        #   the same frame).  MSE lets the model predict the global mean ~0.5 for
        #   every instruction and achieve near-zero loss without ever distinguishing
        #   instructions.  Pearson correlation loss demands the model correctly
        #   RANK instructions by their mean cost — which requires discriminative
        #   instr_vecs and forces real gradient flow to the text encoder.
        frame_mean_gt = torch.stack([
            g[torch.isfinite(g)].mean() if torch.isfinite(g).any()
            else torch.tensor(0.5, device=device)
            for g in gts_list
        ])
        if frame_mean_gt.shape[0] > 1 and frame_mean_gt.std() > 1e-4:
            pred_c  = lang_aux     - lang_aux.mean()
            tgt_c   = frame_mean_gt - frame_mean_gt.mean()
            num     = (pred_c * tgt_c).sum()
            denom   = (pred_c.norm() * tgt_c.norm()).clamp(min=1e-8)
            aux_loss = 1.0 - num / denom          # 0 = perfect rank match
        else:
            aux_loss = F.mse_loss(lang_aux, frame_mean_gt)  # fallback for B=1
        # Alignment loss: Pearson(per-obj align_score, -GT_cost) per sample
        align_losses_list = []
        for b, (a_score, gt) in enumerate(zip(align_scores, gts_list)):
            if a_score.shape[0] < 2:
                continue
            valid = torch.isfinite(gt)
            if valid.sum() < 2:
                continue
            av = a_score[valid]; gv = -gt[valid]
            ac = av - av.mean(); gc = gv - gv.mean()
            pearson = (ac * gc).sum() / (ac.norm() * gc.norm() + 1e-8)
            align_losses_list.append(1.0 - pearson)
        align_loss = (torch.stack(align_losses_list).mean()
                      if align_losses_list
                      else torch.tensor(0.0, device=device))

        total_loss = loss + lambda_lang * aux_loss + lambda_align * align_loss
        ld["loss_lang_aux"]   = aux_loss.item()
        ld["loss_lang_align"] = align_loss.item()
        ld["loss_total"]      = total_loss.item()

        (total_loss / grad_accum).backward()

        if (i + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norms = _clip_lang_grad_norms(model)
            optimizer.step()
            optimizer.zero_grad()
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{k}": v for k, v in ld.items()}
                    | grad_norms
                    | {"batch_step": global_step}
                )
            global_step += 1

        for k, v in ld.items():
            losses[k] += v
        n += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  batch {i+1}/{len(loader)}  loss={losses['loss_total']/n:.4f}")



    return {k: v / max(n, 1) for k, v in losses.items()}, global_step


@torch.no_grad()
def validate(model, loader, criterion, device,
             wandb_run=None, exp_dir=None, epoch=None, max_viz=8):
    model.eval()
    losses               = defaultdict(float)
    n                    = 0
    all_preds, all_gts   = [], []
    viz_count=0
    viz_images= []
    first_batch_vizd = False
    for batch in loader:
        pixel_values, input_ids, attn_mask, masks_list, gts_list = \
            _prepare_batch(batch, device)

        if all(m.shape[0] == 0 for m in masks_list):
            del batch
            continue

        preds, _ = model(pixel_values, masks_list, input_ids, attn_mask)
        _, ld = criterion(preds, gts_list)

        for k, v in ld.items():
            losses[k] += v
        n += 1

        # accumulate metrics
        for p, g in zip(preds, gts_list):
            all_preds.append(p.cpu().numpy())
            all_gts.append(g.cpu().numpy())

        # collect visualization samples (small number)
        if not first_batch_vizd:
            for i, (p, g) in enumerate(zip(preds, gts_list)):
                if viz_count >= max_viz:
                    break
                try:
                    frame_rgb, masks_arr, node_ids = _extract_sample_masks_and_rgb(batch, i)
                    pred_np = p.cpu().numpy()
                    gt_np   = g.cpu().numpy()
                    K = min(len(pred_np), len(gt_np), masks_arr.shape[0])
                    if K == 0:
                        continue
                    img = _render_cost_comparison_image(frame_rgb, masks_arr[:K], gt_np[:K], pred_np[:K])
                    viz_images.append(img)
                    viz_count += 1
                except Exception as e:
                    print(f"Viz sample {i} failed: {e}")
            first_batch_vizd = True
    avg_losses = {k: v / max(n, 1) for k, v in losses.items()}
    metrics    = compute_metrics(all_preds, all_gts)
    metrics.update(avg_losses)

    if wandb_run is not None and viz_images:
        try:
            wb_imgs = [wandb.Image(img, caption=f"val_ep{epoch}_s{idx}") for idx, img in enumerate(viz_images)]
            wandb_run.log({"viz/pred_vs_gt": wb_imgs, "epoch": epoch})
        except Exception:
            pass

    return metrics


# -------------------------------------------------------
# Main entry-point
# -------------------------------------------------------

def train_h5(
    h5_path: str,
    val_h5_path: str | None = None,
    output_dir: str = "./checkpoints",
    run_dir: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = "langgeonet",
    wandb_run_name: str | None = None,
    # ── CLIP+DINOv2 model args ──────────────────────────────────────────────
    clip_model: str = "openai/clip-vit-base-patch16",
    bert_model: str = "bert-base-uncased",
    d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
    # ── VLM model args ──────────────────────────────────────────────────────
    use_vlm: bool = False,
    vlm_path: str = "/data/ws/VLN-CE/chkpt/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512",
    vlm_unfreeze_layers: int = 4,
    d_proj: int = 256,
    lambda_contrast: float = 2.0,
    margin: float = 0.1,
    # ── Training hypers (shared) ────────────────────────────────────────────
    epochs: int = 50, batch_size: int = 8,
    lr_head: float = 1e-4, lr_backbone: float = 1e-5, lr_text: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_epochs: int = 3, grad_accum: int = 1,
    lambda_rank: float = 0.5, lambda_si: float = 0.3,
    lambda_lang: float = 0.1, lambda_align: float = 0.1,
    num_workers: int = 0, seed: int = 42, patience: int = 10,
    device=None,
    resume: str | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    exp_dir  = make_exp_dir(run_dir) if run_dir else output_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    analysis_dir = os.path.join(exp_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # ── Build model ─────────────────────────────────────────────────────────
    logger.info("Building model...")
    if use_vlm:
        model = build_vlm_langgeonet(
            vlm_path=vlm_path, d_proj=d_proj, n_unfreeze=vlm_unfreeze_layers,
        ).to(device)
    else:
        model = build_langgeonet(d_model, n_heads, n_layers, clip_model=clip_model).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {total_p:,} total  {train_p:,} trainable")

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_loader, val_loader = create_h5_episode_pathlengths_dataloader(
        h5_path=h5_path, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        val_split=0.2, seed=seed,
    )
    logger.info(
        f"Loader sizes: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches (bs={batch_size})"
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        cfg = dict(
            h5_path=h5_path, epochs=epochs, batch_size=batch_size,
            lr_head=lr_head, lr_backbone=lr_backbone, lr_text=lr_text,
            lambda_rank=lambda_rank,
        )
        if use_vlm:
            cfg.update(dict(use_vlm=True, vlm_path=vlm_path,
                            vlm_unfreeze_layers=vlm_unfreeze_layers, d_proj=d_proj,
                            lambda_contrast=lambda_contrast, margin=margin))
        else:
            cfg.update(dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                            clip_model=clip_model))
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name or os.path.basename(exp_dir),
            config=cfg,
            dir=exp_dir,
        )
        wandb_run.define_metric("batch_step")
        wandb_run.define_metric("train/*", step_metric="batch_step")
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("epoch/*", step_metric="epoch")
        wandb_run.define_metric("val/*",   step_metric="epoch")
        logger.info(f"W&B: {wandb_run.name} | {wandb_run.url}")
    elif use_wandb:
        logger.warning("wandb not installed — skipping.")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if use_vlm:
        # 3-group: BilinearCostHead | VLM visual (encoder) | VLM LM decoder
        head_params, vlm_vis_params, vlm_txt_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "cost_head" in name:
                head_params.append(p)
            elif "vlm.visual" in name:
                vlm_vis_params.append(p)
            else:   # vlm.model.layers, vlm.model.norm
                vlm_txt_params.append(p)
        logger.info(
            f"  Params — head: {sum(p.numel() for p in head_params):,}  "
            f"vlm_vis: {sum(p.numel() for p in vlm_vis_params):,}  "
            f"vlm_txt: {sum(p.numel() for p in vlm_txt_params):,}  "
            f"(lr_head={lr_head:.0e}  lr_bb={lr_backbone:.0e}  lr_txt={lr_text:.0e})"
        )
        optimizer = optim.AdamW([
            {"params": head_params,    "lr": lr_head},
            {"params": vlm_vis_params, "lr": lr_backbone},
            # No weight decay for LM norm params — prevents LN shrinkage artefacts
            {"params": vlm_txt_params, "lr": lr_text, "weight_decay": 0.0},
        ], weight_decay=weight_decay)
        # Pre-load all instructions for contrastive negative sampling
        all_instructions = _load_all_instructions(h5_path)
        logger.info(
            f"  Contrastive pool: {len(all_instructions)} unique instructions"
        )
    else:
        text_enc_params, vis_enc_params, head_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "clip.text_model" in name:
                text_enc_params.append(p)
            elif "clip.vision_model" in name or "dino" in name:
                vis_enc_params.append(p)
            else:
                head_params.append(p)
        logger.info(
            f"  Params — head: {sum(p.numel() for p in head_params):,}  "
            f"vis_enc: {sum(p.numel() for p in vis_enc_params):,}  "
            f"text_enc: {sum(p.numel() for p in text_enc_params):,}  "
            f"(lr_head={lr_head:.0e}  lr_bb={lr_backbone:.0e}  lr_txt={lr_text:.0e})"
        )
        optimizer = optim.AdamW([
            {"params": head_params,     "lr": lr_head},
            {"params": vis_enc_params,  "lr": lr_backbone},
            {"params": text_enc_params, "lr": lr_text, "weight_decay": 0.0},
        ], weight_decay=weight_decay)
        all_instructions = []  # unused in CLIP path

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (
        (ep + 1) / warmup_epochs if ep < warmup_epochs
        else 0.5 * (1 + np.cos(
            np.pi * (ep - warmup_epochs) / max(1, epochs - warmup_epochs)))
    ))

    criterion   = LangGeoNetLoss(lambda_rank, lambda_si)
    start_epoch, best_mae = _resume_from_checkpoint(
        resume, model, optimizer, scheduler, device)
    wait        = 0
    history     = []
    global_step = start_epoch * len(train_loader)

    logger.info(
        f"Training: {epochs} epochs | bs={batch_size} | accum={grad_accum} | "
        f"device={device} | {'VLM' if use_vlm else 'CLIP+DINOv2'}"
    )

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{epochs}\n{'='*60}")

        if use_vlm:
            train_loss, global_step = train_one_epoch_vlm(
                model, train_loader, criterion, optimizer, device,
                grad_accum, wandb_run, global_step,
                all_instructions, lambda_contrast, margin,
            )
            val_m = validate_vlm(
                model, val_loader, criterion, device,
                wandb_run=wandb_run, exp_dir=analysis_dir,
                epoch=epoch + 1, max_viz=8,
            )
        else:
            train_loss, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                grad_accum, wandb_run, global_step, lambda_lang, lambda_align,
            )
            val_m = validate(
                model, val_loader, criterion, device,
                wandb_run=wandb_run, exp_dir=analysis_dir,
                epoch=epoch + 1, max_viz=8,
            )
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"  Train | total={train_loss.get('loss_total', 0):.4f}  "
            f"reg={train_loss.get('loss_regression', 0):.4f}  "
            f"rank={train_loss.get('loss_ranking', 0):.4f}  "
            + (f"contrast={train_loss.get('loss_contrast', 0):.4f}"
               if use_vlm else
               f"lang_aux={train_loss.get('loss_lang_aux', 0):.4f}  "
               f"lang_align={train_loss.get('loss_lang_align', 0):.4f}")
        )
        logger.info(
            f"  Val   | MAE={val_m['mae']:.4f}  RMSE={val_m['rmse']:.4f}  "
            f"MAPE={val_m['mape']:.4f}  "
            f"RankAcc={val_m['ranking_accuracy']:.4f}  Spearman={val_m['spearman']:.4f}"
        )
        logger.info(
            f"  Val   | acc@0.05={val_m.get('acc@0.05', 0):.4f}  "
            f"acc@0.10={val_m.get('acc@0.1', 0):.4f}  "
            f"acc@0.20={val_m.get('acc@0.2', 0):.4f}"
        )
        logger.info(f"  LR={lr:.2e} | {dt:.1f}s")

        # Norm / gradient monitoring
        if use_vlm:
            ln_norms = _vlm_ln_norms(model)
        else:
            ln_norms = _clip_lang_ln_norms(model)
        if ln_norms:
            logger.info(
                "  LN norms | " +
                "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in list(ln_norms.items())[:6])
            )

        # Modality contribution probe on first val batch
        contrib = {}
        try:
            probe_batch = next(iter(val_loader))
            if use_vlm:
                frgbs, instrs, mlist, _ = _prepare_batch_vlm(probe_batch, device)
                if not all(m.shape[0] == 0 for m in mlist):
                    vis_pct, lang_pct = model.modality_contributions(
                        frgbs, mlist, instrs)
                    contrib = {"contrib/visual_pct": vis_pct, "contrib/lang_pct": lang_pct}
            else:
                pv, ii, am, ml, _ = _prepare_batch(probe_batch, device)
                if not all(m.shape[0] == 0 for m in ml):
                    vis_pct, lang_pct = model.modality_contributions(pv, ml, ii, am)
                    contrib = {"contrib/visual_pct": vis_pct, "contrib/lang_pct": lang_pct}
            if contrib:
                logger.info(
                    f"  Modality contrib | visual={contrib['contrib/visual_pct']:.1f}%  "
                    f"language={contrib['contrib/lang_pct']:.1f}%"
                )
        except Exception as e:
            logger.warning(f"  Modality contrib probe failed: {e}")

        if wandb_run is not None:
            epoch_log = {f"epoch/train_{k}": v for k, v in train_loss.items()}
            epoch_log.update({f"epoch/val_{k}": v for k, v in val_m.items()})
            epoch_log.update(ln_norms)
            epoch_log.update(contrib)
            epoch_log["epoch/lr"]       = lr
            epoch_log["epoch/duration"] = dt
            epoch_log["epoch"]          = epoch + 1
            wandb_run.log(epoch_log)

        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_m, "lr": lr})

        ckpt_cfg = dict(use_vlm=use_vlm)
        if use_vlm:
            ckpt_cfg.update(dict(vlm_path=vlm_path,
                                 vlm_unfreeze_layers=vlm_unfreeze_layers, d_proj=d_proj))
        else:
            ckpt_cfg.update(dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                                 clip_model=clip_model))

        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            wait     = 0
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae":         best_mae,
                "val_metrics":          val_m,
                "config":               ckpt_cfg,
            }, os.path.join(ckpt_dir, "best_model.pt"))
            logger.info(f"  ★ Best model saved (MAE={best_mae:.4f})")
            if wandb_run is not None:
                wandb_run.summary["best_val_mae"]   = best_mae
                wandb_run.summary["best_val_epoch"] = epoch + 1
        else:
            wait += 1
            logger.info(f"  No improvement ({wait}/{patience})")

        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae":         best_mae,
            "config":               ckpt_cfg,
        }, os.path.join(ckpt_dir, "latest_model.pt"))

        if wait >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}.")
            break

    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    if wandb_run is not None:
        wandb_run.finish()

    logger.info(f"Done. Best val MAE: {best_mae:.4f} | {exp_dir}")
    return model, history


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--h5_path",       required=True)
    p.add_argument("--val_h5_path",   default=None)
    p.add_argument("--run_dir",       default=None)
    p.add_argument("--output_dir",    default="./checkpoints")
    p.add_argument("--wandb_project", default="langgeonet")
    p.add_argument("--wandb_run_name",default=None)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr_head",       type=float, default=1e-3)
    p.add_argument("--lr_backbone",   type=float, default=1e-5)
    p.add_argument("--lr_text",       type=float, default=5e-5)
    p.add_argument("--grad_accum",    type=int,   default=1)
    p.add_argument("--lambda_rank",   type=float, default=0.5)
    p.add_argument("--lambda_si",     type=float, default=0.0)
    p.add_argument("--patience",      type=int,   default=100)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--device",        default=None)
    p.add_argument("--resume",        default=None)
    # CLIP+DINOv2 args
    p.add_argument("--d_model",       type=int,   default=256)
    p.add_argument("--n_layers",      type=int,   default=1)
    p.add_argument("--lambda_lang",   type=float, default=0.5)
    p.add_argument("--lambda_align",  type=float, default=0.1)
    # VLM args
    p.add_argument("--use_vlm",       action="store_true",
                   help="Use Qwen2-VL backbone instead of CLIP+DINOv2")
    p.add_argument("--vlm_path",      default=(
        "/data/ws/VLN-CE/chkpt/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512"))
    p.add_argument("--vlm_unfreeze_layers", type=int, default=4,
                   help="How many last transformer layers to unfreeze in Qwen2-VL")
    p.add_argument("--d_proj",        type=int,   default=256,
                   help="BilinearCostHead projection dim")
    p.add_argument("--lambda_contrast",type=float, default=2.0,
                   help="Weight of contrastive instruction loss (VLM only)")
    p.add_argument("--margin",        type=float, default=0.1,
                   help="Minimum required mean prediction change (VLM only)")
    a = p.parse_args()

    train_h5(
        h5_path=a.h5_path,
        val_h5_path=a.val_h5_path,
        run_dir=a.run_dir,
        output_dir=a.output_dir,
        use_wandb=not a.no_wandb,
        wandb_project=a.wandb_project,
        wandb_run_name=a.wandb_run_name,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr_head=a.lr_head,
        lr_backbone=a.lr_backbone,
        lr_text=a.lr_text,
        grad_accum=a.grad_accum,
        lambda_rank=a.lambda_rank,
        lambda_si=a.lambda_si,
        patience=a.patience,
        num_workers=a.num_workers,
        device=a.device,
        resume=a.resume,
        # CLIP args
        d_model=a.d_model,
        n_layers=a.n_layers,
        lambda_lang=a.lambda_lang,
        lambda_align=a.lambda_align,
        # VLM args
        use_vlm=a.use_vlm,
        vlm_path=a.vlm_path,
        vlm_unfreeze_layers=a.vlm_unfreeze_layers,
        d_proj=a.d_proj,
        lambda_contrast=a.lambda_contrast,
        margin=a.margin,
    )

