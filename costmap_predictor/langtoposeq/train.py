"""
Training loop for LangTopoSeg.

Usage
-----
    python train.py                         # default config
    python train.py --data_root /my/data    # override any config field
    python train.py --ckpt_path ckpts/epoch_010.pt  # resume
"""

import os
import time
import logging
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config  import LangTopoSegConfig
from dataset import build_dataloaders
from model   import LangTopoSeg
from losses  import compute_total_loss

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _model_forward(model: LangTopoSeg, batch: dict, device: torch.device) -> dict:
    batch = _batch_to_device(batch, device)
    return model(
        rgb          = batch["rgb"],
        masks        = batch["masks"],
        centroids    = batch["centroids"],
        areas        = batch["areas"],
        k_valid      = batch["k_valid"],
        instructions = batch["instruction"],
    )


def save_checkpoint(model, optimizer, epoch, loss, cfg: LangTopoSegConfig):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch:03d}.pt")
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "loss":       loss,
    }, path)
    logger.info(f"Checkpoint saved → {path}")
    return path


def load_checkpoint(model, optimizer, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1
    logger.info(f"Resumed from {path}  (epoch {start_epoch})")
    return start_epoch


def _build_scheduler(optimizer, cfg: LangTopoSegConfig, steps_per_epoch: int):
    """
    Linear warmup for the first warmup_epochs, then cosine annealing.
    Operates on a per-step basis so the warmup finishes exactly at
    epoch=warmup_epochs regardless of loader length.
    """
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    cosine_steps = max((cfg.epochs - cfg.warmup_epochs) * steps_per_epoch, 1)

    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)

    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


class _CSVLogger:
    """Append one results row per epoch to a CSV file."""

    def __init__(self, path: str):
        self.path = path
        self._header_written = os.path.exists(path)

    def log(self, row: dict):
        write_header = not self._header_written
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
                self._header_written = True
            w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, epoch):
    model.train()
    total = {k: 0.0 for k in ["total", "obs", "rank", "node", "dir", "sym"]}
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()
        outputs = _model_forward(model, batch, device)
        losses  = compute_total_loss(outputs, batch, cfg, device)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        for k in total:
            total[k] += losses[k].item()
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in total.items()}
    logger.info(
        f"[Train ep{epoch:03d}]  total={avg['total']:.4f}  "
        f"obs={avg['obs']:.4f}  rank={avg['rank']:.4f}  "
        f"node={avg['node']:.4f}  dir={avg['dir']:.4f}  sym={avg['sym']:.4f}"
    )
    return avg


@torch.no_grad()
def eval_one_epoch(model, loader, cfg, device, epoch):
    model.eval()
    total = {k: 0.0 for k in ["total", "obs", "rank", "node", "dir", "sym"]}
    n_batches = 0

    for batch in loader:
        outputs = _model_forward(model, batch, device)
        losses  = compute_total_loss(outputs, batch, cfg, device)
        for k in total:
            total[k] += losses[k].item()
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in total.items()}
    logger.info(
        f"[Val   ep{epoch:03d}]  total={avg['total']:.4f}  "
        f"obs={avg['obs']:.4f}  rank={avg['rank']:.4f}  "
        f"node={avg['node']:.4f}  dir={avg['dir']:.4f}  sym={avg['sym']:.4f}"
    )
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: LangTopoSegConfig):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Model ───────────────────────────────────────────────────────────────
    model = LangTopoSeg(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # ── Optimiser ───────────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # ── Logging ─────────────────────────────────────────────────────────────
    csv_logger = _CSVLogger(os.path.join(cfg.log_dir, "metrics.csv"))
    tb_writer  = None
    if _TB_AVAILABLE:
        tb_writer = SummaryWriter(log_dir=cfg.log_dir)
        logger.info(f"TensorBoard logging → {cfg.log_dir}")

    # ── Optionally resume ────────────────────────────────────────────────────
    start_epoch = 0
    if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        start_epoch = load_checkpoint(model, optimizer, cfg.ckpt_path, device)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_ckpt = None

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, cfg, device, epoch
        )

        row = {"epoch": epoch, "split": "train", **train_metrics}
        csv_logger.log(row)
        if tb_writer is not None:
            for k, v in train_metrics.items():
                tb_writer.add_scalar(f"train/{k}", v, epoch)
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % cfg.eval_every == 0:
            val_metrics = eval_one_epoch(model, val_loader, cfg, device, epoch)
            val_row = {"epoch": epoch, "split": "val", **val_metrics}
            csv_logger.log(val_row)
            if tb_writer is not None:
                for k, v in val_metrics.items():
                    tb_writer.add_scalar(f"val/{k}", v, epoch)

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_ckpt = save_checkpoint(model, optimizer, epoch, best_val_loss, cfg)
                logger.info(f"  ↑ New best val_loss = {best_val_loss:.4f}")

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_metrics["total"], cfg)

        logger.info(f"  Epoch {epoch} done in {time.time() - t0:.1f}s")

    if tb_writer is not None:
        tb_writer.close()

    logger.info(f"Training complete. Best checkpoint: {best_ckpt}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Train LangTopoSeg")
    # Allow overriding any LangTopoSegConfig field from the command line.
    cfg_defaults = LangTopoSegConfig()
    for field_name, default in vars(cfg_defaults).items():
        if isinstance(default, bool):
            parser.add_argument(f"--{field_name}", type=lambda x: x.lower() != "false",
                                default=default)
        elif isinstance(default, (int, float, str)) or default is None:
            parser.add_argument(f"--{field_name}", type=type(default) if default is not None else str,
                                default=default)
        # dict / list fields (direction_priors) are skipped – use defaults
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = LangTopoSegConfig()
    # Apply CLI overrides to config
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    train(cfg)
