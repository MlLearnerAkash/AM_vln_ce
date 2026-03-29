# """
# Training loop for LangGeoNet.

# Features:
#     - Differential learning rates (frozen backbone vs trainable head)
#     - Cosine LR schedule with linear warmup
#     - Mixed-precision training (AMP)
#     - Gradient accumulation
#     - Evaluation metrics: MAE, RMSE, ranking accuracy, Spearman rho
#     - Checkpointing with early stopping
# """
# from __future__ import annotations
# import os
# import time
# import json
# import logging
# from collections import defaultdict


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from scipy.stats import spearmanr

# import h5py

# from model import build_langgeonet
# from dataset import create_dataloaders, create_h5_maskpls_dataloader, create_h5_episode_pathlengths_dataloader
# from losses import LangGeoNetLoss

# try:
#     import wandb
#     WANDB_AVAILABLE = True
# except ImportError:
#     wandb = None  # type: ignore[assignment]
#     WANDB_AVAILABLE = False

# # Batch key aliases so both dataloaders work transparently.
# # Old collate: geodesic_dists_list / class_ids_list
# # New H5 collate: pls_list  (no class_ids_list)
# _GT_KEYS   = ("pls_list", "geodesic_dists_list")
# _CID_KEY   = "class_ids_list"


# def _get_gt(batch):
#     """Return the ground-truth list whichever key is present."""
#     for k in _GT_KEYS:
#         if k in batch:
#             return batch[k]
#     raise KeyError(f"Batch has none of the expected GT keys: {_GT_KEYS}")


# def _get_cids(batch, device):
#     """Return class-id tensors list, or a list of None when absent."""
#     if _CID_KEY in batch:
#         return [c.to(device) for c in batch[_CID_KEY]]
#     # H5 dataset has no class ids; model.object_encoder accepts None per sample.
#     return [None] * len(batch["masks_list"])

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)


# def _resume_from_checkpoint(
#     resume: str | None,
#     model: nn.Module,
#     optimizer: optim.Optimizer,
#     scheduler,
#     device: torch.device,
# ) -> tuple[int, float]:
#     """
#     Load a checkpoint and restore model, optimizer, and scheduler state.

#     Returns
#     -------
#     start_epoch : int   – first epoch to run (0-based)
#     best_mae    : float – best validation MAE seen so far
#     """
#     if not resume:
#         return 0, float("inf")

#     if not os.path.isfile(resume):
#         raise FileNotFoundError(f"Resume checkpoint not found: {resume}")

#     logger.info(f"Resuming from checkpoint: {resume}")
#     ckpt = torch.load(resume, map_location=device, weights_only= False)

#     model.load_state_dict(ckpt["model_state_dict"])
#     logger.info("  ✔ model weights loaded")

#     if "optimizer_state_dict" in ckpt:
#         optimizer.load_state_dict(ckpt["optimizer_state_dict"])
#         logger.info("  ✔ optimizer state loaded")

#     if "scheduler_state_dict" in ckpt and scheduler is not None:
#         scheduler.load_state_dict(ckpt["scheduler_state_dict"])
#         logger.info("  ✔ scheduler state loaded")

#     start_epoch = int(ckpt.get("epoch", 0))          # checkpoint stores completed epoch count
#     best_mae    = float(ckpt.get("best_val_mae", float("inf")))
#     logger.info(f"  ↳ Resuming at epoch {start_epoch + 1}, best MAE so far = {best_mae:.4f}")
#     return start_epoch, best_mae


# # -------------------------------------------------------
# # Experiment Directories
# # -------------------------------------------------------

# def make_exp_dir(base_dir: str) -> str:
#     """
#     Create and return the next available ``expN`` sub-directory inside
#     *base_dir* (exp1, exp2, …), incrementing until an unused name is found.
#     """
#     os.makedirs(base_dir, exist_ok=True)
#     n = 1
#     while True:
#         candidate = os.path.join(base_dir, f"exp{n}")
#         if not os.path.exists(candidate):
#             os.makedirs(candidate)
#             logger.info(f"Experiment directory: {candidate}")
#             return candidate
#         n += 1


# def _flatten_dict(d: dict, prefix: str = "") -> dict:
#     """Recursively flatten a nested dict (used for W&B table rows)."""
#     out: dict = {}
#     for k, v in d.items():
#         key = f"{prefix}/{k}" if prefix else k
#         if isinstance(v, dict):
#             out.update(_flatten_dict(v, key))
#         else:
#             out[key] = v
#     return out


# # -------------------------------------------------------
# # Dataset Analysis
# # -------------------------------------------------------

# def analyze_h5_dataset(
#     ds: "H5MaskPLSDataset",
#     out_dir: str,
#     max_h5_samples: int = 5_000,
# ) -> dict:
#     """
#     Produce a statistical analysis of an H5MaskPLSDataset and write artefacts
#     to *out_dir*:

#     * ``summary.json``            – aggregate statistics
#     * ``pls_distribution.png``    – histogram of normalised PL scores
#     * ``masks_per_frame.png``     – distribution of objects per frame
#     * ``frames_per_episode.png``  – distribution of trajectory lengths
#     * ``instruction_lengths.png`` – instruction word-count histogram
#     """
#     try:
#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#     except ImportError:
#         logger.warning("matplotlib not installed — skipping dataset plots.")
#         plt = None

#     from collections import Counter

#     os.makedirs(out_dir, exist_ok=True)
#     samples = ds.samples
#     n_total = len(samples)

#     # ---- Episode / frame counts (in-memory index, no I/O) ----
#     ep_counts     = Counter(s["ep_folder"] for s in samples)
#     frames_per_ep = list(ep_counts.values())

#     # ---- Instruction word counts ----
#     instr_lengths = [len(s["instruction"].split()) for s in samples]

#     # ---- Subset of H5 for PL + mask-count stats ----
#     rng        = np.random.default_rng(42)
#     subset_idx = rng.choice(n_total, size=min(max_h5_samples, n_total), replace=False)
#     all_pls: list = []
#     masks_per_frame: list = []

#     with h5py.File(ds.h5_path, "r") as h5f:
#         for i in subset_idx:
#             key = samples[int(i)]["sample_key"]
#             grp = h5f[key]
#             pls = grp["img_pls"][()].astype(np.float32)
#             lo, hi = float(pls.min()), float(pls.max())
#             if hi - lo > 1e-6:
#                 pls = (pls - lo) / (hi - lo)
#             else:
#                 pls = np.zeros_like(pls)
#             all_pls.extend(pls.tolist())
#             masks_per_frame.append(int(len(grp["img_masks"])))

#     arr_pls = np.array(all_pls,         dtype=np.float32)
#     arr_mpm = np.array(masks_per_frame, dtype=np.int32)

#     # ---- Summary JSON ----
#     summary = {
#         "total_samples":  n_total,
#         "total_episodes": len(ep_counts),
#         "frames_per_episode": {
#             "min":    int(min(frames_per_ep)),
#             "max":    int(max(frames_per_ep)),
#             "mean":   float(np.mean(frames_per_ep)),
#             "median": float(np.median(frames_per_ep)),
#         },
#         "masks_per_frame": {
#             "min":    int(arr_mpm.min()),
#             "max":    int(arr_mpm.max()),
#             "mean":   float(arr_mpm.mean()),
#             "median": float(np.median(arr_mpm)),
#         },
#         "pls_scores_normalised": {
#             "min":  float(arr_pls.min()),
#             "max":  float(arr_pls.max()),
#             "mean": float(arr_pls.mean()),
#             "std":  float(arr_pls.std()),
#             "p25":  float(np.percentile(arr_pls, 25)),
#             "p50":  float(np.percentile(arr_pls, 50)),
#             "p75":  float(np.percentile(arr_pls, 75)),
#         },
#         "instruction_word_counts": {
#             "min":  int(min(instr_lengths)),
#             "max":  int(max(instr_lengths)),
#             "mean": float(np.mean(instr_lengths)),
#         },
#         "h5_subset_size": int(len(subset_idx)),
#     }

#     with open(os.path.join(out_dir, "summary.json"), "w") as f:
#         json.dump(summary, f, indent=2)

#     logger.info(
#         f"[analyze_h5_dataset] {n_total:,} samples | "
#         f"{len(ep_counts):,} episodes | "
#         f"masks/frame mean={summary['masks_per_frame']['mean']:.1f} | "
#         f"PL mean={summary['pls_scores_normalised']['mean']:.3f}"
#     )

#     if plt is None:
#         return summary

#     # ---- Plots ----
#     def _save(fig, name: str) -> None:
#         fig.tight_layout()
#         fig.savefig(os.path.join(out_dir, name), dpi=120)
#         plt.close(fig)

#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.hist(arr_pls, bins=50, color="steelblue", edgecolor="white", lw=0.4)
#     ax.set_xlabel("Normalised PL score"); ax.set_ylabel("Count")
#     ax.set_title(f"PL Score Distribution  (subset n={len(arr_pls):,})")
#     _save(fig, "pls_distribution.png")

#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.hist(arr_mpm, bins=range(int(arr_mpm.min()), int(arr_mpm.max()) + 2),
#             color="darkorange", edgecolor="white", lw=0.4)
#     ax.set_xlabel("Objects per frame"); ax.set_ylabel("Count")
#     ax.set_title(f"Masks per Frame  (subset n={len(arr_mpm):,})")
#     _save(fig, "masks_per_frame.png")

#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.hist(frames_per_ep, bins=40, color="seagreen", edgecolor="white", lw=0.4)
#     ax.set_xlabel("Frames per episode"); ax.set_ylabel("Count")
#     ax.set_title(f"Trajectory Length Distribution  ({len(frames_per_ep):,} episodes)")
#     _save(fig, "frames_per_episode.png")

#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.hist(instr_lengths, bins=40, color="mediumpurple", edgecolor="white", lw=0.4)
#     ax.set_xlabel("Instruction length (words)"); ax.set_ylabel("Count")
#     ax.set_title(f"Instruction Word-Count Distribution  (n={n_total:,})")
#     _save(fig, "instruction_lengths.png")

#     logger.info(f"[analyze_h5_dataset] Plots saved to {out_dir}")
#     return summary


# # -------------------------------------------------------
# # Evaluation Metrics
# # -------------------------------------------------------


# # -------------------------------------------------------
# # Sample Visualisation
# # -------------------------------------------------------

# def _render_pls_overlay(
#     image_np: np.ndarray,
#     masks: np.ndarray,
#     pls_gt: np.ndarray,
#     pls_pred: np.ndarray | None = None,
#     label_fontsize: int = 9,
# ) -> np.ndarray:
#     """
#     Render one frame: image with each mask contour coloured by its PL score.

#     Parameters
#     ----------
#     image_np  : (H, W, 3) uint8  – original RGB image resized to display size
#     masks     : (K, H, W) bool   – instance masks
#     pls_gt    : (K,) float32     – ground-truth (normalised) PL scores
#     pls_pred  : (K,) float32 or None – model predictions (shown alongside GT)
#     """
#     try:
#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#         import matplotlib.patches as mpatches
#         from matplotlib.colors import Normalize
#         from matplotlib.cm import ScalarMappable
#     except ImportError:
#         return image_np

#     n_cols = 2 if pls_pred is not None else 1
#     fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4),
#                              squeeze=False)

#     cmap  = plt.get_cmap("RdYlGn")
#     norm  = Normalize(vmin=0.0, vmax=1.0)
#     sm    = ScalarMappable(cmap=cmap, norm=norm)

#     def _draw(ax, title, scores):
#         ax.imshow(image_np)
#         ax.set_title(title, fontsize=label_fontsize + 1)
#         ax.axis("off")
#         H, W = image_np.shape[:2]
#         for k, (mask, score) in enumerate(zip(masks, scores)):
#             if not mask.any():
#                 continue
#             colour = cmap(norm(float(score)))
#             overlay = np.zeros((H, W, 4), dtype=np.float32)
#             overlay[mask] = (*colour[:3], 0.45)
#             ax.imshow(overlay)
#             # centroid label
#             ys, xs = np.where(mask)
#             cy, cx = ys.mean(), xs.mean()
#             ax.text(
#                 cx, cy, f"{score:.2f}",
#                 ha="center", va="center",
#                 fontsize=label_fontsize - 1,
#                 color="white",
#                 fontweight="bold",
#             )
#         fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="PL score")

#     _draw(axes[0, 0], "GT PL", pls_gt)
#     if pls_pred is not None:
#         _draw(axes[0, 1], "Pred PL", pls_pred)

#     fig.tight_layout()
#     fig.canvas.draw()
#     buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return buf


# def collect_viz_samples(
#     ds,
#     n: int = 8,
#     seed: int = 0,
# ) -> list[dict]:
#     """
#     Draw *n* random samples from *ds* (H5MaskPLSDataset) and return a list of
#     dicts with the raw data needed for visualisation.

#     Each dict contains:
#         pixel_values   – (3, H_clip, W_clip) float tensor  (CLIP-preprocessed)
#         masks          – (K, H, W) bool tensor
#         pls_gt         – (K,) float32 tensor (normalised)
#         input_ids      – (L,) long tensor
#         attention_mask – (L,) long tensor
#         image_np       – (H_orig, W_orig, 3) uint8 numpy array
#         instruction    – str
#         sample_key     – str
#     """
#     from PIL import Image as PILImage

#     rng      = np.random.default_rng(seed)
#     indices  = rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist()
#     samples  = []
#     for idx in indices:
#         item = ds[idx]   # uses __getitem__; H5 opened lazily
#         meta = ds.samples[idx]
#         img_path = os.path.join(
#             ds.traj_root,
#             meta["ep_folder"],
#             "images",
#             f"{meta['frame_idx']:05d}.png",
#         )
#         image_np = np.array(PILImage.open(img_path).convert("RGB"))
#         samples.append({
#             **item,
#             "image_np":    image_np,
#             "instruction": meta["instruction"],
#             "sample_key":  meta["sample_key"],
#         })
#     return samples


# @torch.no_grad()
# def build_sample_images(
#     model,
#     viz_samples: list[dict],
#     device,
#     epoch: int,
# ) -> list:
#     """
#     Run *model* on *viz_samples*, render GT-vs-Pred overlays, and return a
#     list of ``wandb.Image`` objects ready to be inserted into any log dict.

#     Pass ``model=None`` for GT-only reference panels (before training).
#     Returns an empty list when wandb is unavailable.
#     """
#     if not WANDB_AVAILABLE:
#         return []

#     model_was_training = model is not None and model.training
#     if model is not None:
#         model.eval()

#     wandb_images = []
#     for s in viz_samples:
#         pv  = s["pixel_values"].unsqueeze(0).to(device)
#         ids = s["input_ids"].unsqueeze(0).to(device)
#         am  = s["attention_mask"].unsqueeze(0).to(device)
#         msk = [s["masks"].to(device)]
#         cid = [None]

#         if model is not None:
#             preds, _ = model(pv, msk, cid, ids, am)
#             pls_pred = preds[0].cpu().numpy().astype(np.float32)
#         else:
#             pls_pred = None

#         msk_np = s["masks"].numpy()          # (K, H, W)
#         H, W   = msk_np.shape[1], msk_np.shape[2]
#         from PIL import Image as PILImage
#         img_disp = np.array(
#             PILImage.fromarray(s["image_np"]).resize((W, H), PILImage.BILINEAR)
#         )

#         panel = _render_pls_overlay(
#             img_disp,
#             msk_np,
#             s["pls"].numpy().astype(np.float32),
#             pls_pred,
#         )
#         caption = (
#             f"{s['sample_key']}\n"
#             f"epoch {epoch} | "
#             f"{s['instruction'][:80]}"
#         )
#         wandb_images.append(wandb.Image(panel, caption=caption))

#     if model_was_training:
#         model.train()

#     return wandb_images


# def log_sample_panel(
#     wandb_run,
#     model,
#     viz_samples: list[dict],
#     device,
#     epoch: int,
#     panel_key: str = "viz/samples",
# ) -> None:
#     """
#     Convenience wrapper: build panels and log them in a standalone W&B call.
#     Use this only when you need a one-off log (e.g. the GT reference at step 0).
#     For per-epoch logging prefer ``build_sample_images`` so the images are
#     merged into the same ``wandb_run.log`` call as the epoch metrics.
#     """
#     if wandb_run is None:
#         return
#     images = build_sample_images(model, viz_samples, device, epoch)
#     if images:
#         wandb_run.log({panel_key: images, "epoch": epoch})

# def compute_metrics(all_preds, all_gts):
#     """
#     Compute comprehensive metrics.

#     Args:
#         all_preds: list of 1-D numpy arrays (per sample)
#         all_gts:   list of 1-D numpy arrays (per sample)

#     Returns:
#         dict of metric_name -> value
#     """
#     flat_p = np.concatenate(all_preds)
#     flat_g = np.concatenate(all_gts)

#     mae = np.mean(np.abs(flat_p - flat_g))
#     mse = np.mean((flat_p - flat_g) ** 2)
#     rmse = np.sqrt(mse)

#     valid = flat_g > 0.01
#     mape = np.mean(np.abs(flat_p[valid] - flat_g[valid]) / flat_g[valid]) if valid.sum() else 0.0

#     # Per-sample ranking accuracy
#     rank_accs, spearman_corrs = [], []
#     for p, g in zip(all_preds, all_gts):
#         if len(p) < 2:
#             continue
#         correct = total = 0
#         for i in range(len(p)):
#             for j in range(i + 1, len(p)):
#                 if abs(g[i] - g[j]) < 1e-6:
#                     continue
#                 total += 1
#                 correct += int((p[i] < p[j]) == (g[i] < g[j]))
#         if total:
#             rank_accs.append(correct / total)
#         if len(p) >= 3:
#             c, _ = spearmanr(p, g)
#             if not np.isnan(c):
#                 spearman_corrs.append(c)

#     # Threshold accuracy
#     thresh = {}
#     for d in [0.05, 0.10, 0.20]:
#         thresh[f"acc@{d}"] = float(np.mean(np.abs(flat_p - flat_g) < d))

#     return {
#         "mae": mae, "mse": mse, "rmse": rmse, "mape": mape,
#         "ranking_accuracy": np.mean(rank_accs) if rank_accs else 0.0,
#         "spearman": np.mean(spearman_corrs) if spearman_corrs else 0.0,
#         **thresh,
#     }


# # -------------------------------------------------------
# # One Epoch
# # -------------------------------------------------------

# def train_one_epoch(model, loader, criterion, optimizer, device,
#                     grad_accum=1, wandb_run=None, global_step: int = 0):
#     model.train()
#     losses = defaultdict(float)
#     n = 0
#     optimizer.zero_grad()

#     for i, batch in enumerate(loader):
#         pv = batch["pixel_values"].to(device)
#         ids = batch["input_ids"].to(device)
#         am = batch["attention_mask"].to(device)
#         masks = [m.to(device) for m in batch["masks_list"]]
#         cids = _get_cids(batch, device)
#         gts = [g.to(device) for g in _get_gt(batch)]

#         preds, _ = model(pv, masks, cids, ids, am)
#         loss, ld = criterion(preds, gts)
#         loss = loss / grad_accum
#         loss.backward()

#         if (i + 1) % grad_accum == 0:
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             optimizer.zero_grad()

#             if wandb_run is not None:
#                 step_log = {f"train/{k}": v for k, v in ld.items()}
#                 step_log["batch_step"] = global_step
#                 wandb_run.log(step_log)
#             global_step += 1
            

#         for k, v in ld.items():
#             losses[k] += v
#         n += 1

#         if (i + 1) % 50 == 0:
#             logger.info(f"  batch {i+1}/{len(loader)} loss={losses['loss_total']/n:.4f}")

#     return {k: v / max(n, 1) for k, v in losses.items()}, global_step


# @torch.no_grad()
# def validate(model, loader, criterion, device, wandb_run=None, epoch: int = 0):
#     model.eval()
#     losses = defaultdict(float)
#     n = 0
#     all_preds, all_gts = [], []

#     for batch in loader:
#         pv = batch["pixel_values"].to(device)
#         ids = batch["input_ids"].to(device)
#         am = batch["attention_mask"].to(device)
#         masks = [m.to(device) for m in batch["masks_list"]]
#         cids = _get_cids(batch, device)
#         gts = [g.to(device) for g in _get_gt(batch)]

#         preds, _ = model(pv, masks, cids, ids, am)
#         _, ld = criterion(preds, gts)

#         for k, v in ld.items():
#             losses[k] += v
#         n += 1

#         for p, g in zip(preds, gts):
#             all_preds.append(p.cpu().numpy())
#             all_gts.append(g.cpu().numpy())

#     avg_losses = {k: v / max(n, 1) for k, v in losses.items()}
#     metrics = compute_metrics(all_preds, all_gts)
#     metrics.update(avg_losses)

#     if wandb_run is not None:
#         wandb_run.log({f"val/{k}": v for k, v in metrics.items()}, step=epoch)

#     return metrics


# # -------------------------------------------------------
# # Main Training Loop
# # -------------------------------------------------------

# def train(
#     data_root,
#     output_dir="./checkpoints",
#     # Model
#     d_model=256, n_heads=8, n_layers=6, num_classes=1550,
#     clip_model="openai/clip-vit-base-patch16",
#     bert_model="bert-base-uncased",
#     # Training
#     epochs=50, batch_size=8,
#     lr_head=1e-4, lr_backbone=1e-5, weight_decay=0.01,
#     warmup_epochs=3, grad_accum=1,
#     # Loss
#     lambda_rank=0.5, lambda_si=0.3,
#     # Misc
#     num_workers=4, seed=42, patience=10, device=None,
#     resume: str | None = None,
# ):
#     wandb.init(
#         project="langgeonet",
#         name= "vlnce",
#         config={
#             "d_model": d_model,
#             "n_heads": n_heads,
#             "n_layers": n_layers,
#             "epochs": epochs
#         }
#     )
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(device)
#     os.makedirs(output_dir, exist_ok=True)

#     # --- Model ---
#     logger.info("Building model...")
#     model = build_langgeonet(d_model, n_heads, n_layers, num_classes, clip_model)
#     model = model.to(device)

#     total_p = sum(p.numel() for p in model.parameters())
#     train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"Params: {total_p:,} total, {train_p:,} trainable")

#     # --- Data ---
#     logger.info("Loading data...")
#     train_loader, val_loader = create_dataloaders(
#         data_root, batch_size, num_workers, clip_model
#     )

#     # --- Optimizer (differential LR) ---
#     backbone_names = {"clip", "bert"}
#     head_params, bb_params = [], []
#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         (bb_params if any(bn in name for bn in backbone_names) else head_params).append(p)

#     optimizer = optim.AdamW([
#         {"params": head_params, "lr": lr_head},
#         {"params": bb_params,   "lr": lr_backbone},
#     ], weight_decay=weight_decay)

#     # --- Scheduler (cosine + warmup) ---
#     def lr_lambda(ep):
#         if ep < warmup_epochs:
#             return (ep + 1) / warmup_epochs
#         return 0.5 * (1 + np.cos(np.pi * (ep - warmup_epochs) / max(1, epochs - warmup_epochs)))

#     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     criterion = LangGeoNetLoss(lambda_rank, lambda_si)

#     # --- Resume ---
#     start_epoch, best_mae = _resume_from_checkpoint(resume, model, optimizer, scheduler, device)
#     wait = 0
#     history = []

#     logger.info(f"Training: {epochs} epochs, bs={batch_size}, accum={grad_accum}, device={device}")

#     for epoch in range(start_epoch, epochs):
#         t0 = time.time()
#         logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{epochs}\n{'='*60}")

#         train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_accum)
#         val_m = validate(model, val_loader, criterion, device)
#         scheduler.step()

#         dt = time.time() - t0
#         lr = optimizer.param_groups[0]["lr"]

#         logger.info(
#             f"  Train | total={train_loss['loss_total']:.4f} "
#             f"reg={train_loss['loss_regression']:.4f} "
#             f"rank={train_loss['loss_ranking']:.4f} "
#             f"si={train_loss['loss_scale_invariant']:.4f}"
#         )
#         logger.info(
#             f"  Val   | MAE={val_m['mae']:.4f} RMSE={val_m['rmse']:.4f} "
#             f"RankAcc={val_m['ranking_accuracy']:.4f} "
#             f"Spearman={val_m['spearman']:.4f}"
#         )
#         logger.info(
#             f"  Val   | acc@0.05={val_m['acc@0.05']:.4f} "
#             f"acc@0.10={val_m['acc@0.1']:.4f} "
#             f"acc@0.20={val_m['acc@0.2']:.4f}"
#         )
#         logger.info(f"  LR={lr:.2e} | {dt:.1f}s")

#         history.append({"epoch": epoch+1, "train": train_loss, "val": val_m, "lr": lr})

#         # Checkpoint best
#         if val_m["mae"] < best_mae:
#             best_mae = val_m["mae"]
#             wait = 0
#             torch.save({
#                 "epoch": epoch + 1,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "scheduler_state_dict": scheduler.state_dict(),
#                 "best_val_mae": best_mae,
#                 "val_metrics": val_m,
#                 "config": dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
#                                num_classes=num_classes, clip_model=clip_model, bert_model=bert_model),
#             }, os.path.join(output_dir, "best_model.pt"))
#             logger.info(f"  ★ Best model saved (MAE={best_mae:.4f})")
#         else:
#             wait += 1
#             logger.info(f"  No improvement ({wait}/{patience})")

#         # Latest checkpoint
#         torch.save({
#             "epoch": epoch + 1,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "best_val_mae": best_mae,
#         }, os.path.join(output_dir, "latest_model.pt"))
    
#         if wait >= patience:
#             logger.info(f"\nEarly stopping at epoch {epoch+1}.")
#             break
#         wandb.log({
#                 "train/total_loss": train_loss['loss_total'],
#                 "train/regression_loss": train_loss['loss_regression'],
#                 "train/ranking_loss": train_loss['loss_ranking'],
#                 "train/scale_invariant_loss": train_loss['loss_scale_invariant'],
#                 "epoch": epoch
#             })
#         wandb.log({
#                 "val/MAE": val_m['mae'],
#                 "val/RMSE": val_m['rmse'],
#                 "val/RankAcc": val_m['ranking_accuracy'],
#                 "val/Spearman": val_m['spearman'],
#                 "val/acc@0.05": val_m['acc@0.05'],
#                 "val/acc@0.10": val_m['acc@0.1'],
#                 "val/acc@0.20": val_m['acc@0.2'],
#                 "epoch": epoch
#             })
        

#     with open(os.path.join(output_dir, "history.json"), "w") as f:
#         json.dump(history, f, indent=2, default=str)

#     logger.info(f"\nDone. Best val MAE: {best_mae:.4f}")
#     return model, history


# # -------------------------------------------------------
# # H5-backed training entry-point
# # -------------------------------------------------------

# def train_h5(
#     h5_path: str,
#     jsonl_path: str,
#     base_dir: str,
#     val_h5_path: str | None = None,
#     val_jsonl_path: str | None = None,
#     output_dir: str = "./checkpoints",
#     run_dir: str | None = None,
#     # W&B
#     use_wandb: bool = True,
#     wandb_project: str = "langgeonet",
#     wandb_run_name: str | None = None,
#     n_viz_samples: int = 8,
#     # Model
#     d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
#     num_classes: int = 1550,
#     clip_model: str = "openai/clip-vit-base-patch16",
#     bert_model: str = "bert-base-uncased",
#     # Training
#     epochs: int = 50, batch_size: int = 8,
#     lr_head: float = 1e-4, lr_backbone: float = 1e-5,
#     weight_decay: float = 0.01,
#     warmup_epochs: int = 3, grad_accum: int = 1,
#     # Loss
#     lambda_rank: float = 0.5, lambda_si: float = 0.3,
#     # Misc
#     num_workers: int = 4, seed: int = 42, patience: int = 10,
#     device=None,
#     resume: str | None = None,
# ):
#     """
#     Training loop backed by H5MaskPLSDataset.

#     *run_dir* (e.g. ``./runs``) activates auto-numbered experiment directories
#     (``run_dir/exp1``, ``run_dir/exp2``, …).  Each expN contains:
#         checkpoints/   – model snapshots
#         analysis/      – dataset statistics and plots

#     When *run_dir* is None, *output_dir* is used directly (no sub-folder).

#     val_h5_path / val_jsonl_path are optional; when omitted the training loader
#     is reused for validation with a logged warning.
#     """
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(device)

#     # --- Experiment directory ---
#     if run_dir is not None:
#         exp_dir = make_exp_dir(run_dir)
#     else:
#         exp_dir = output_dir
#         os.makedirs(exp_dir, exist_ok=True)

#     ckpt_dir     = os.path.join(exp_dir, "checkpoints")
#     analysis_dir = os.path.join(exp_dir, "analysis")
#     os.makedirs(ckpt_dir, exist_ok=True)

#     # --- Model ---
#     logger.info("Building model...")
#     model = build_langgeonet(d_model, n_heads, n_layers, num_classes, clip_model)
#     model = model.to(device)

#     total_p = sum(p.numel() for p in model.parameters())
#     train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"Params: {total_p:,} total, {train_p:,} trainable")

#     # --- Data ---
#     logger.info("Loading H5 data...")
#     train_loader = create_h5_maskpls_dataloader(
#         h5_path=h5_path, jsonl_path=jsonl_path, base_dir=base_dir,
#         batch_size=batch_size, shuffle=True, num_workers=num_workers,
#         clip_model=clip_model,
#     )

#     if val_h5_path and val_jsonl_path:
#         val_loader = create_h5_maskpls_dataloader(
#             h5_path=val_h5_path, jsonl_path=val_jsonl_path, base_dir=base_dir,
#             batch_size=batch_size, shuffle=False, num_workers=num_workers,
#             clip_model=clip_model,
#         )
#     else:
#         logger.warning("No validation H5/JSONL provided — using training set as val.")
#         val_loader = create_h5_maskpls_dataloader(
#             h5_path=h5_path, jsonl_path=jsonl_path, base_dir=base_dir,
#             batch_size=batch_size, shuffle=False, num_workers=num_workers,
#             clip_model=clip_model,
#         )

#     # --- Dataset analysis ---
#     logger.info("Analysing dataset...")
#     ds_summary = analyze_h5_dataset(train_loader.dataset, analysis_dir)

#     # --- W&B ---
#     wandb_run = None
#     if use_wandb and WANDB_AVAILABLE:
#         run_cfg = dict(
#             h5_path=h5_path, jsonl_path=jsonl_path, base_dir=base_dir,
#             d_model=d_model, n_heads=n_heads, n_layers=n_layers,
#             num_classes=num_classes, clip_model=clip_model,
#             epochs=epochs, batch_size=batch_size,
#             lr_head=lr_head, lr_backbone=lr_backbone, weight_decay=weight_decay,
#             warmup_epochs=warmup_epochs, grad_accum=grad_accum,
#             lambda_rank=lambda_rank, lambda_si=lambda_si,
#             exp_dir=exp_dir,
#         )
#         wandb_run = wandb.init(
#             project=wandb_project,
#             name=wandb_run_name or os.path.basename(exp_dir),
#             config=run_cfg,
#             dir=exp_dir,
#         )
#         # Separate x-axes: train/* plots against batch_step,
#         # everything epoch-level plots against epoch.
#         # This prevents W&B from dropping logs when the two counters diverge.
#         wandb_run.define_metric("batch_step")
#         wandb_run.define_metric("train/*", step_metric="batch_step")
#         wandb_run.define_metric("epoch")
#         wandb_run.define_metric("epoch/*", step_metric="epoch")
#         wandb_run.define_metric("val/*",   step_metric="epoch")
#         wandb_run.define_metric("viz/*",   step_metric="epoch")
#         # Log dataset summary once at epoch 0
#         flat_summary = _flatten_dict(ds_summary)
#         wandb_run.log({
#             "dataset/summary": wandb.Table(
#                 columns=["metric", "value"],
#                 data=[[k, str(v)] for k, v in flat_summary.items()],
#             ),
#             "epoch": 0,
#         })
#         logger.info(f"W&B run: {wandb_run.name} | {wandb_run.url}")
#     elif use_wandb and not WANDB_AVAILABLE:
#         logger.warning("wandb not installed — skipping W&B logging. `pip install wandb` to enable.")

#     # --- Fixed visualisation samples (collected once before training) ---
#     viz_samples: list[dict] = []
#     if wandb_run is not None and n_viz_samples > 0:
#         logger.info(f"Collecting {n_viz_samples} visualisation samples...")
#         viz_samples = collect_viz_samples(train_loader.dataset, n=n_viz_samples, seed=seed)
#         # Log GT-only reference (epoch 0, model=None → pure GT overlays)
#         log_sample_panel(wandb_run, None, viz_samples, device,
#                          epoch=0, panel_key="viz/gt_reference")

#     # --- Optimizer (differential LR) ---
#     backbone_names = {"clip", "bert"}
#     head_params, bb_params = [], []
#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         (bb_params if any(bn in name for bn in backbone_names) else head_params).append(p)

#     optimizer = optim.AdamW([
#         {"params": head_params, "lr": lr_head},
#         {"params": bb_params,   "lr": lr_backbone},
#     ], weight_decay=weight_decay)

#     def lr_lambda(ep):
#         if ep < warmup_epochs:
#             return (ep + 1) / warmup_epochs
#         return 0.5 * (1 + np.cos(
#             np.pi * (ep - warmup_epochs) / max(1, epochs - warmup_epochs)))

#     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     criterion = LangGeoNetLoss(lambda_rank, lambda_si)

#     # --- Resume ---
#     start_epoch, best_mae = _resume_from_checkpoint(resume, model, optimizer, scheduler, device)
#     wait        = 0
#     history     = []
#     global_step = start_epoch * len(train_loader)   # approximate batch step offset

#     logger.info(
#         f"H5 training: {epochs} epochs, bs={batch_size}, "
#         f"accum={grad_accum}, device={device} | exp_dir={exp_dir}"
#     )

#     for epoch in range(start_epoch, epochs):
#         t0 = time.time()
#         logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{epochs}\n{'='*60}")

#         train_loss, global_step = train_one_epoch(
#             model, train_loader, criterion, optimizer, device, grad_accum,
#             wandb_run=wandb_run, global_step=global_step,
#         )
#         val_m = validate(
#             model, val_loader, criterion, device)
#         scheduler.step()

#         dt = time.time() - t0
#         lr = optimizer.param_groups[0]["lr"]

#         logger.info(
#             f"  Train | total={train_loss['loss_total']:.4f} "
#             f"reg={train_loss['loss_regression']:.4f} "
#             f"rank={train_loss['loss_ranking']:.4f} "
#             f"si={train_loss['loss_scale_invariant']:.4f}"
#         )
#         logger.info(
#             f"  Val   | MAE={val_m['mae']:.4f} RMSE={val_m['rmse']:.4f} "
#             f"RankAcc={val_m['ranking_accuracy']:.4f} "
#             f"Spearman={val_m['spearman']:.4f}"
#         )
#         logger.info(f"  LR={lr:.2e} | {dt:.1f}s")

#         # Epoch-level W&B log — metrics + image panels in ONE call so
#         # nothing is silently dropped by W&B's per-step deduplication.
#         if wandb_run is not None:
#             epoch_log = {f"epoch/train_{k}": v for k, v in train_loss.items()}
#             epoch_log.update({f"epoch/val_{k}": v for k, v in val_m.items()})
#             epoch_log["epoch/lr"]       = lr
#             epoch_log["epoch/duration"] = dt

#             # Build GT-vs-Pred panels and merge into the same dict.
#             if viz_samples:
#                 panel_images = build_sample_images(
#                     model, viz_samples, device, epoch=epoch + 1
#                 )
#                 if panel_images:
#                     epoch_log["viz/pred_vs_gt"] = panel_images

#             epoch_log["epoch"] = epoch + 1
#             wandb_run.log(epoch_log)

#         history.append({"epoch": epoch + 1, "train": train_loss, "val": val_m, "lr": lr})

#         if val_m["mae"] < best_mae:
#             best_mae = val_m["mae"]
#             wait = 0
#             torch.save({
#                 "epoch": epoch + 1,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "scheduler_state_dict": scheduler.state_dict(),
#                 "best_val_mae": best_mae,
#                 "val_metrics": val_m,
#                 "config": dict(
#                     d_model=d_model, n_heads=n_heads, n_layers=n_layers,
#                     num_classes=num_classes, clip_model=clip_model,
#                     bert_model=bert_model,
#                 ),
#             }, os.path.join(ckpt_dir, "best_model.pt"))
#             logger.info(f"  ★ Best model saved (MAE={best_mae:.4f})")
#             if wandb_run is not None:
#                 wandb_run.summary["best_val_mae"]   = best_mae
#                 wandb_run.summary["best_val_epoch"] = epoch + 1
#         else:
#             wait += 1
#             logger.info(f"  No improvement ({wait}/{patience})")

#         torch.save({
#             "epoch": epoch + 1,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "best_val_mae": best_mae,
#         }, os.path.join(ckpt_dir, "latest_model.pt"))

#         if wait >= patience:
#             logger.info(f"\nEarly stopping at epoch {epoch+1}.")
#             break

#     with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
#         json.dump(history, f, indent=2, default=str)

#     if wandb_run is not None:
#         wandb_run.finish()

#     logger.info(f"\nDone. Best val MAE: {best_mae:.4f}. Results in {exp_dir}")
#     return model, history


# # -------------------------------------------------------
# # CLI
# # -------------------------------------------------------

# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser(description="Train LangGeoNet")
#     p.add_argument("--mode", choices=["dir", "h5"], default="dir",
#                    help="'dir' uses the original directory dataset; "
#                         "'h5' uses H5MaskPLSDataset")
#     # --- dir-mode args ---
#     p.add_argument("--data_root", default=None,
#                    help="[dir mode] root directory with episode folders")
#     # --- h5-mode args ---
#     p.add_argument("--h5_path", default=None,
#                    help="[h5 mode] path to the HDF5 file")
#     p.add_argument("--jsonl_path", default=None,
#                    help="[h5 mode] path to train JSONL (filename + instruction)")
#     p.add_argument("--base_dir", default=None,
#                    help="[h5 mode] root dir that contains trajectories/")
#     p.add_argument("--val_h5_path", default=None,
#                    help="[h5 mode] optional validation HDF5 file")
#     p.add_argument("--val_jsonl_path", default=None,
#                    help="[h5 mode] optional validation JSONL file")
#     p.add_argument("--run_dir", default=None,
#                    help="[h5 mode] root dir for auto-numbered exp1/exp2/… runs; "
#                         "overrides --output_dir when set")
#     p.add_argument("--wandb_project", default="langgeonet",
#                    help="[h5 mode] W&B project name")
#     p.add_argument("--wandb_run_name", default=None,
#                    help="[h5 mode] W&B run display name (defaults to expN)")
#     p.add_argument("--no_wandb", action="store_true",
#                    help="[h5 mode] disable W&B logging")
#     p.add_argument("--n_viz_samples", type=int, default=8,
#                    help="[h5 mode] number of fixed samples visualised per epoch in W&B")
#     p.add_argument("--output_dir", default="./checkpoints")
#     p.add_argument("--epochs", type=int, default=100)
#     p.add_argument("--batch_size", type=int, default=1)
#     p.add_argument("--lr_head", type=float, default=1e-4)
#     p.add_argument("--lr_backbone", type=float, default=1e-5)
#     p.add_argument("--d_model", type=int, default=256)
#     p.add_argument("--n_layers", type=int, default=6)
#     p.add_argument("--grad_accum", type=int, default=1)
#     p.add_argument("--lambda_rank", type=float, default=0.5)
#     p.add_argument("--lambda_si", type=float, default=0.3)
#     p.add_argument("--patience", type=int, default=10)
#     p.add_argument("--num_workers", type=int, default=4)
#     p.add_argument("--device", default=None)
#     p.add_argument("--resume", default=None,
#                    help="Path to a checkpoint (.pt) to resume training from")
#     a = p.parse_args()

#     shared = dict(
#         output_dir=a.output_dir,
#         d_model=a.d_model, n_layers=a.n_layers,
#         epochs=a.epochs, batch_size=a.batch_size,
#         lr_head=a.lr_head, lr_backbone=a.lr_backbone,
#         grad_accum=a.grad_accum,
#         lambda_rank=a.lambda_rank, lambda_si=a.lambda_si,
#         patience=a.patience, num_workers=a.num_workers, device=a.device,
#         resume=a.resume,
#     )

#     if a.mode == "h5":
#         if not (a.h5_path and a.jsonl_path and a.base_dir):
#             p.error("--h5_path, --jsonl_path and --base_dir are required for h5 mode")
#         train_h5(
#             h5_path=a.h5_path,
#             jsonl_path=a.jsonl_path,
#             base_dir=a.base_dir,
#             val_h5_path=a.val_h5_path,
#             val_jsonl_path=a.val_jsonl_path,
#             run_dir=a.run_dir,
#             use_wandb=not a.no_wandb,
#             wandb_project=a.wandb_project,
#             wandb_run_name=a.wandb_run_name,
#             n_viz_samples=a.n_viz_samples,
#             **shared,
#         )
#     else:
#         if not a.data_root:
#             p.error("--data_root is required for dir mode")
#         train(data_root=a.data_root, **shared)

#----------------------original code --------------------------------------------
"""
Training loop for LangGeoNet — H5EpisodePathLengthsDataset edition.
"""
from __future__ import annotations
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
from PIL import Image as PILImage
from transformers import CLIPProcessor

from model import build_langgeonet
from dataset import create_h5_episode_pathlengths_dataloader
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

def _prepare_batch(batch: dict, clip_processor: CLIPProcessor, device: torch.device):
    """
    Convert a raw H5EpisodePathLengthsDataset batch into the tensors the
    model and loss expect.

    Returns
    -------
    pixel_values : (B, 3, H_clip, W_clip)  – CLIP image features
    input_ids    : (B, L)
    attn_mask    : (B, L)
    masks_list   : list[Tensor(K_i, H, W) bool]  – one per sample
    gts_list     : list[Tensor(K_i,) float32]     – mean path-length per node
    """
    pil_images = [PILImage.fromarray(rgb.astype(np.uint8)) for rgb in batch["frame_rgbs"]]
    clip_out   = clip_processor(images=pil_images, return_tensors="pt", padding=True)
    pixel_values = clip_out["pixel_values"].to(device)

    input_ids  = batch["input_ids"].to(device)
    attn_mask  = batch["attention_mask"].to(device)

    masks_list, gts_list = [], []
    for registry in batch["node_registries"]:
        if not registry:
            masks_list.append(torch.zeros(0, 1, 1, dtype=torch.bool, device=device))
            gts_list.append(torch.zeros(0, dtype=torch.float32, device=device))
            continue

        node_masks, node_gts = [], []
        for entry in registry.values():
            m = entry.mask
            if not isinstance(m, np.ndarray):
                m = np.asarray(m)
            node_masks.append(torch.from_numpy(m.astype(bool)))

            pr = np.asarray(entry.path_row, dtype=np.float64)
            cost = float(np.nanmean(pr)) if pr.size else 0.0
            node_gts.append(cost)

        masks_list.append(torch.stack(node_masks).to(device))

        gt_arr = np.array(node_gts, dtype=np.float32)
        lo, hi = gt_arr.min(), gt_arr.max()
        if hi - lo > 1e-6:
            gt_arr = (gt_arr - lo) / (hi - lo)
        else:
            gt_arr = np.zeros_like(gt_arr)
        gts_list.append(torch.from_numpy(gt_arr).to(device))

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
# Train / validate
# -------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    clip_processor, grad_accum=1, wandb_run=None, global_step=0):
    model.train()
    losses      = defaultdict(float)
    n           = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        pixel_values, input_ids, attn_mask, masks_list, gts_list = \
            _prepare_batch(batch, clip_processor, device)

        # skip frames where every registry was empty
        if all(m.shape[0] == 0 for m in masks_list):
            continue

        cids = [None] * len(masks_list)
        preds, _ = model(pixel_values, masks_list, cids, input_ids, attn_mask)
        loss, ld = criterion(preds, gts_list)
        (loss / grad_accum).backward()

        if (i + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in ld.items()} | {"batch_step": global_step})
            global_step += 1

        for k, v in ld.items():
            losses[k] += v
        n += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  batch {i+1}/{len(loader)}  loss={losses['loss_total']/n:.4f}")

    return {k: v / max(n, 1) for k, v in losses.items()}, global_step


@torch.no_grad()
def validate(model, loader, criterion, device, clip_processor):
    model.eval()
    losses               = defaultdict(float)
    n                    = 0
    all_preds, all_gts   = [], []

    for batch in loader:
        pixel_values, input_ids, attn_mask, masks_list, gts_list = \
            _prepare_batch(batch, clip_processor, device)

        if all(m.shape[0] == 0 for m in masks_list):
            continue

        cids     = [None] * len(masks_list)
        preds, _ = model(pixel_values, masks_list, cids, input_ids, attn_mask)
        _, ld    = criterion(preds, gts_list)

        for k, v in ld.items():
            losses[k] += v
        n += 1

        for p, g in zip(preds, gts_list):
            all_preds.append(p.cpu().numpy())
            all_gts.append(g.cpu().numpy())

    avg_losses = {k: v / max(n, 1) for k, v in losses.items()}
    metrics    = compute_metrics(all_preds, all_gts)
    metrics.update(avg_losses)
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
    clip_model: str = "openai/clip-vit-base-patch16",
    bert_model: str = "bert-base-uncased",
    d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
    num_classes: int = 1550,
    epochs: int = 50, batch_size: int = 8,
    lr_head: float = 1e-4, lr_backbone: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_epochs: int = 3, grad_accum: int = 1,
    lambda_rank: float = 0.5, lambda_si: float = 0.3,
    num_workers: int = 0, seed: int = 42, patience: int = 10,
    device=None,
    resume: str | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    exp_dir  = make_exp_dir(run_dir) if run_dir else output_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # CLIP processor lives here so we can call it in _prepare_batch
    clip_processor = CLIPProcessor.from_pretrained(clip_model)

    logger.info("Building model...")
    model   = build_langgeonet(d_model, n_heads, n_layers, num_classes, clip_model).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {total_p:,} total  {train_p:,} trainable")

    logger.info("Loading data...")
    train_loader, val_loader = create_h5_episode_pathlengths_dataloader(
        h5_path=h5_path, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    # val_loader = create_h5_episode_pathlengths_dataloader(
    #     h5_path=val_h5_path or h5_path,
    #     batch_size=batch_size, shuffle=False, num_workers=num_workers,
    # )
    if not val_h5_path:
        logger.warning("No val H5 provided — reusing train set for validation.")

    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name or os.path.basename(exp_dir),
            config=dict(
                h5_path=h5_path, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                epochs=epochs, batch_size=batch_size, lr_head=lr_head,
                lr_backbone=lr_backbone, lambda_rank=lambda_rank, lambda_si=lambda_si,
            ),
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

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (
        (ep + 1) / warmup_epochs if ep < warmup_epochs
        else 0.5 * (1 + np.cos(np.pi * (ep - warmup_epochs) / max(1, epochs - warmup_epochs)))
    ))

    criterion   = LangGeoNetLoss(lambda_rank, lambda_si)
    start_epoch, best_mae = _resume_from_checkpoint(resume, model, optimizer, scheduler, device)
    wait        = 0
    history     = []
    global_step = start_epoch * len(train_loader)

    logger.info(f"Training: {epochs} epochs | bs={batch_size} | accum={grad_accum} | device={device}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{epochs}\n{'='*60}")

        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            clip_processor, grad_accum, wandb_run, global_step,
        )
        val_m = validate(model, val_loader, criterion, device, clip_processor)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"  Train | total={train_loss['loss_total']:.4f}  "
            f"reg={train_loss['loss_regression']:.4f}  "
            f"rank={train_loss['loss_ranking']:.4f}  "
            f"si={train_loss['loss_scale_invariant']:.4f}"
        )
        logger.info(
            f"  Val   | MAE={val_m['mae']:.4f}  RMSE={val_m['rmse']:.4f}  "
            f"RankAcc={val_m['ranking_accuracy']:.4f}  Spearman={val_m['spearman']:.4f}"
        )
        logger.info(f"  LR={lr:.2e} | {dt:.1f}s")

        if wandb_run is not None:
            epoch_log = {f"epoch/train_{k}": v for k, v in train_loss.items()}
            epoch_log.update({f"epoch/val_{k}": v for k, v in val_m.items()})
            epoch_log["epoch/lr"]       = lr
            epoch_log["epoch/duration"] = dt
            epoch_log["epoch"]          = epoch + 1
            wandb_run.log(epoch_log)

        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_m, "lr": lr})

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
                "config": dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                               num_classes=num_classes, clip_model=clip_model),
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
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path",       required=True)
    p.add_argument("--val_h5_path",   default=None)
    p.add_argument("--run_dir",       default=None)
    p.add_argument("--output_dir",    default="./checkpoints")
    p.add_argument("--wandb_project", default="langgeonet")
    p.add_argument("--wandb_run_name",default=None)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr_head",       type=float, default=1e-4)
    p.add_argument("--lr_backbone",   type=float, default=1e-5)
    p.add_argument("--d_model",       type=int,   default=256)
    p.add_argument("--n_layers",      type=int,   default=6)
    p.add_argument("--grad_accum",    type=int,   default=1)
    p.add_argument("--lambda_rank",   type=float, default=1.0)
    p.add_argument("--lambda_si",     type=float, default=0.01)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--device",        default=None)
    p.add_argument("--resume",        default=None)
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
        d_model=a.d_model,
        n_layers=a.n_layers,
        grad_accum=a.grad_accum,
        lambda_rank=a.lambda_rank,
        lambda_si=a.lambda_si,
        patience=a.patience,
        num_workers=a.num_workers,
        device=a.device,
        resume=a.resume,
    )