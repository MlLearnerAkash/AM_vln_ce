"""
Training loop for LangGeoNet.

Features:
    - Differential learning rates (frozen backbone vs trainable head)
    - Cosine LR schedule with linear warmup
    - Mixed-precision training (AMP)
    - Gradient accumulation
    - Evaluation metrics: MAE, RMSE, ranking accuracy, Spearman rho
    - Checkpointing with early stopping
"""

import os
import time
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr

from model import build_langgeonet
from dataset import create_dataloaders
from losses import LangGeoNetLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------

def compute_metrics(all_preds, all_gts):
    """
    Compute comprehensive metrics.

    Args:
        all_preds: list of 1-D numpy arrays (per sample)
        all_gts:   list of 1-D numpy arrays (per sample)

    Returns:
        dict of metric_name -> value
    """
    flat_p = np.concatenate(all_preds)
    flat_g = np.concatenate(all_gts)

    mae = np.mean(np.abs(flat_p - flat_g))
    mse = np.mean((flat_p - flat_g) ** 2)
    rmse = np.sqrt(mse)

    valid = flat_g > 0.01
    mape = np.mean(np.abs(flat_p[valid] - flat_g[valid]) / flat_g[valid]) if valid.sum() else 0.0

    # Per-sample ranking accuracy
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

    # Threshold accuracy
    thresh = {}
    for d in [0.05, 0.10, 0.20]:
        thresh[f"acc@{d}"] = float(np.mean(np.abs(flat_p - flat_g) < d))

    return {
        "mae": mae, "mse": mse, "rmse": rmse, "mape": mape,
        "ranking_accuracy": np.mean(rank_accs) if rank_accs else 0.0,
        "spearman": np.mean(spearman_corrs) if spearman_corrs else 0.0,
        **thresh,
    }


# -------------------------------------------------------
# One Epoch
# -------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum=1):
    model.train()
    losses = defaultdict(float)
    n = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        pv = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        masks = [m.to(device) for m in batch["masks_list"]]
        cids = [c.to(device) for c in batch["class_ids_list"]]
        gts = [g.to(device) for g in batch["geodesic_dists_list"]]

        preds, _ = model(pv, masks, cids, ids, am)
        loss, ld = criterion(preds, gts)
        loss = loss / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        for k, v in ld.items():
            losses[k] += v
        n += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  batch {i+1}/{len(loader)} loss={losses['loss_total']/n:.4f}")

    return {k: v / max(n, 1) for k, v in losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = defaultdict(float)
    n = 0
    all_preds, all_gts = [], []

    for batch in loader:
        pv = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        masks = [m.to(device) for m in batch["masks_list"]]
        cids = [c.to(device) for c in batch["class_ids_list"]]
        gts = [g.to(device) for g in batch["geodesic_dists_list"]]

        preds, _ = model(pv, masks, cids, ids, am)
        _, ld = criterion(preds, gts)

        for k, v in ld.items():
            losses[k] += v
        n += 1

        for p, g in zip(preds, gts):
            all_preds.append(p.cpu().numpy())
            all_gts.append(g.cpu().numpy())

    avg_losses = {k: v / max(n, 1) for k, v in losses.items()}
    metrics = compute_metrics(all_preds, all_gts)
    metrics.update(avg_losses)
    return metrics


# -------------------------------------------------------
# Main Training Loop
# -------------------------------------------------------

def train(
    data_root,
    output_dir="./checkpoints",
    # Model
    d_model=256, n_heads=8, n_layers=6, num_classes=1550,
    clip_model="openai/clip-vit-base-patch16",
    bert_model="bert-base-uncased",
    # Training
    epochs=50, batch_size=8,
    lr_head=1e-4, lr_backbone=1e-5, weight_decay=0.01,
    warmup_epochs=3, grad_accum=1,
    # Loss
    lambda_rank=0.5, lambda_si=0.3,
    # Misc
    num_workers=4, seed=42, patience=10, device=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # --- Model ---
    logger.info("Building model...")
    model = build_langgeonet(d_model, n_heads, n_layers, num_classes, clip_model)
    model = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {total_p:,} total, {train_p:,} trainable")

    # --- Data ---
    logger.info("Loading data...")
    train_loader, val_loader = create_dataloaders(
        data_root, batch_size, num_workers, clip_model
    )

    # --- Optimizer (differential LR) ---
    backbone_names = {"clip", "bert"}
    head_params, bb_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if any(bn in name for bn in backbone_names) else head_params).append(p)

    optimizer = optim.AdamW([
        {"params": head_params, "lr": lr_head},
        {"params": bb_params,   "lr": lr_backbone},
    ], weight_decay=weight_decay)

    # --- Scheduler (cosine + warmup) ---
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup_epochs) / max(1, epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = LangGeoNetLoss(lambda_rank, lambda_si)

    # --- Loop ---
    best_mae = float("inf")
    wait = 0
    history = []

    logger.info(f"Training: {epochs} epochs, bs={batch_size}, accum={grad_accum}, device={device}")

    for epoch in range(epochs):
        t0 = time.time()
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{epochs}\n{'='*60}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_accum)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"  Train | total={train_loss['loss_total']:.4f} "
            f"reg={train_loss['loss_regression']:.4f} "
            f"rank={train_loss['loss_ranking']:.4f} "
            f"si={train_loss['loss_scale_invariant']:.4f}"
        )
        logger.info(
            f"  Val   | MAE={val_m['mae']:.4f} RMSE={val_m['rmse']:.4f} "
            f"RankAcc={val_m['ranking_accuracy']:.4f} "
            f"Spearman={val_m['spearman']:.4f}"
        )
        logger.info(
            f"  Val   | acc@0.05={val_m['acc@0.05']:.4f} "
            f"acc@0.10={val_m['acc@0.1']:.4f} "
            f"acc@0.20={val_m['acc@0.2']:.4f}"
        )
        logger.info(f"  LR={lr:.2e} | {dt:.1f}s")

        history.append({"epoch": epoch+1, "train": train_loss, "val": val_m, "lr": lr})

        # Checkpoint best
        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            wait = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae": best_mae,
                "val_metrics": val_m,
                "config": dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                               num_classes=num_classes, clip_model=clip_model, bert_model=bert_model),
            }, os.path.join(output_dir, "best_model.pt"))
            logger.info(f"  ★ Best model saved (MAE={best_mae:.4f})")
        else:
            wait += 1
            logger.info(f"  No improvement ({wait}/{patience})")

        # Latest checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae": best_mae,
        }, os.path.join(output_dir, "latest_model.pt"))

        if wait >= patience:
            logger.info(f"\nEarly stopping at epoch {epoch+1}.")
            break

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info(f"\nDone. Best val MAE: {best_mae:.4f}")
    return model, history


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train LangGeoNet")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lambda_rank", type=float, default=0.5)
    p.add_argument("--lambda_si", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    train(
        data_root=a.data_root, output_dir=a.output_dir,
        d_model=a.d_model, n_layers=a.n_layers,
        epochs=a.epochs, batch_size=a.batch_size,
        lr_head=a.lr_head, lr_backbone=a.lr_backbone,
        grad_accum=a.grad_accum,
        lambda_rank=a.lambda_rank, lambda_si=a.lambda_si,
        patience=a.patience, num_workers=a.num_workers, device=a.device,
    )
