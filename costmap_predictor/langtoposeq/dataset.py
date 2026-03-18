"""
Dataset for LangTopoSeg.

Expected on-disk layout (produced by lang_geonet_dataset_generator_e3d.py):

    data_root/
      train.txt                   # one episode_id per line
      val.txt
      episode_{id}/
        instruction.txt           # single line: navigation instruction
        episode_graph.pickle      # NetworkX graph (optional, used for virtual edges)
        frame_000/
          rgb.png                 # [H, W, 3]  uint8
          masks.npy               # [K, H, W]  uint8  (binary instance masks)
          class_ids.npy           # [K]        int64
          instance_ids.npy        # [K]        int64
          e3d_distances.npy       # [K]        float32  (normalised mean e3d per instance)
        frame_001/
        ...

Each Dataset item is a *temporal window* of n+1 consecutive frames (current frame
+ n past frames).  All tensors are padded to max_instances along the K dimension
so samples can be batched with the default collate_fn.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Return (x_norm, y_norm) centroid of a binary mask in [-0.5, 0.5]."""
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    cx = float(xs.mean()) / W - 0.5   # negative=left, positive=right
    cy = float(ys.mean()) / H - 0.5   # negative=top,  positive=bottom
    return cx, cy


def _mask_area(mask: np.ndarray) -> float:
    """Normalised area of a binary mask."""
    return float(mask.sum()) / mask.size


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame loader
# ─────────────────────────────────────────────────────────────────────────────

def load_frame(frame_dir: str, max_k: int, image_h: int, image_w: int) -> Dict:
    """
    Load one frame from disk and return a dict with padded tensors.

    Returns
    -------
    dict with keys:
        rgb          : [3, H, W]        float32  (normalised to [0,1])
        masks        : [max_k, H, W]    float32  (padded binary masks)
        class_ids    : [max_k]          int64
        instance_ids : [max_k]          int64
        e3d_gt       : [max_k]          float32  (normalised mean e3d)
        centroids    : [max_k, 2]       float32  (cx_norm, cy_norm)
        areas        : [max_k]          float32  (normalised pixel area)
        k_valid      : int              actual number of instances
    """
    # ── RGB ──────────────────────────────────────────────────────────────
    rgb_path = os.path.join(frame_dir, "rgb.png")
    rgb = Image.open(rgb_path).convert("RGB")
    rgb = rgb.resize((image_w, image_h), Image.BILINEAR)
    rgb_t = TF.to_tensor(rgb)                          # [3, H, W]  float32 [0,1]

    # ── Instance data ─────────────────────────────────────────────────────
    masks_raw    = np.load(os.path.join(frame_dir, "masks.npy"))         # [K, H, W]
    class_ids    = np.load(os.path.join(frame_dir, "class_ids.npy"))     # [K]
    instance_ids = np.load(os.path.join(frame_dir, "instance_ids.npy"))  # [K]
    e3d          = np.load(os.path.join(frame_dir, "e3d_distances.npy")) # [K]

    K = len(masks_raw)

    # Compute per-instance centroids and areas
    centroids = np.zeros((K, 2), dtype=np.float32)
    areas     = np.zeros(K,      dtype=np.float32)
    for i, m in enumerate(masks_raw):
        cx, cy = _mask_centroid(m)
        centroids[i] = [cx, cy]
        areas[i]     = _mask_area(m)

    # Resize masks to (image_h, image_w) if needed
    if masks_raw.shape[1:] != (image_h, image_w) and K > 0:
        resized = np.stack([
            np.array(Image.fromarray(masks_raw[i]).resize(
                (image_w, image_h), Image.NEAREST))
            for i in range(K)
        ], axis=0)
        masks_raw = resized

    # ── Pad to max_k ─────────────────────────────────────────────────────
    k_valid = min(K, max_k)
    pad = max_k - k_valid

    def pad_1d(arr, fill=0):
        arr = arr[:k_valid]
        return np.concatenate([arr, np.full(pad, fill, dtype=arr.dtype)]) if pad else arr

    def pad_2d(arr, fill=0.0):
        arr = arr[:k_valid]
        return np.concatenate([arr, np.zeros((pad, arr.shape[1]), dtype=arr.dtype)]) if pad else arr

    def pad_3d(arr, fill=0.0):
        arr = arr[:k_valid]
        return np.concatenate([arr, np.zeros((pad, *arr.shape[1:]), dtype=arr.dtype)]) if pad else arr

    masks_pad    = pad_3d(masks_raw.astype(np.float32))  # [max_k, H, W]
    class_pad    = pad_1d(class_ids,    fill=0)          # [max_k]
    instance_pad = pad_1d(instance_ids, fill=0)          # [max_k]
    e3d_pad      = pad_1d(e3d,          fill=0.0)        # [max_k]
    cent_pad     = pad_2d(centroids)                     # [max_k, 2]
    area_pad     = pad_1d(areas,        fill=0.0)        # [max_k]

    return {
        "rgb":          rgb_t,
        "masks":        torch.from_numpy(masks_pad),
        "class_ids":    torch.from_numpy(class_pad),
        "instance_ids": torch.from_numpy(instance_pad),
        "e3d_gt":       torch.from_numpy(e3d_pad),
        "centroids":    torch.from_numpy(cent_pad),
        "areas":        torch.from_numpy(area_pad),
        "k_valid":      k_valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Episode-level dataset
# ─────────────────────────────────────────────────────────────────────────────

class LangTopoDataset(Dataset):
    """
    Each item is a temporal window of (n_frames + 1) consecutive frames
    from a single episode, plus the navigation instruction.

    The *target* frame is the last frame in the window (index = n_frames).
    Past frames provide temporal context.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        n_frames: int = 4,
        max_instances: int = 32,
        image_h: int = 120,
        image_w: int = 160,
        augment: bool = False,
        min_frames_required: int = 2,
    ):
        super().__init__()
        self.data_root       = Path(data_root)
        self.n_frames        = n_frames          # number of past frames
        self.max_instances   = max_instances
        self.image_h         = image_h
        self.image_w         = image_w
        self.augment         = augment

        split_file = self.data_root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        episode_ids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

        # Build (episode_id, start_frame_idx) index over all valid windows
        self.samples: List[Tuple[str, int]] = []
        for ep_id in episode_ids:
            ep_dir = self.data_root / f"episode_{ep_id}"
            frame_dirs = sorted(ep_dir.glob("frame_*"))
            n_f = len(frame_dirs)
            if n_f < min_frames_required:
                continue
            window_size = n_frames + 1   # n past + 1 current
            for start in range(n_f - window_size + 1):
                self.samples.append((str(ep_id), start))

        print(f"[LangTopoDataset] split={split}  episodes={len(episode_ids)}  windows={len(self.samples)}")

    def _load_instruction(self, ep_dir: Path) -> str:
        p = ep_dir / "instruction.txt"
        return p.read_text().strip() if p.exists() else ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        ep_id, start = self.samples[idx]
        ep_dir = self.data_root / f"episode_{ep_id}"

        instruction = self._load_instruction(ep_dir)
        frame_dirs  = sorted(ep_dir.glob("frame_*"))
        window_size = self.n_frames + 1
        window_dirs = frame_dirs[start : start + window_size]

        frames_data = [
            load_frame(str(fd), self.max_instances, self.image_h, self.image_w)
            for fd in window_dirs
        ]

        # Stack along a new temporal dimension
        def stack(key):
            return torch.stack([fd[key] for fd in frames_data], dim=0)   # [T, ...]

        return {
            # Temporal tensors  [T=n_frames+1, ...]
            "rgb":          stack("rgb"),          # [T, 3, H, W]
            "masks":        stack("masks"),        # [T, K, H, W]
            "class_ids":    stack("class_ids"),    # [T, K]
            "instance_ids": stack("instance_ids"), # [T, K]
            "e3d_gt":       stack("e3d_gt"),       # [T, K]  — target: last frame
            "centroids":    stack("centroids"),    # [T, K, 2]
            "areas":        stack("areas"),        # [T, K]
            "k_valid":      torch.tensor([fd["k_valid"] for fd in frames_data]),  # [T]
            # Language
            "instruction":  instruction,
            # Meta
            "episode_id":   ep_id,
            "start_frame":  start,
        }


def build_dataloaders(cfg):
    """Convenience: returns (train_loader, val_loader)."""
    from torch.utils.data import DataLoader

    train_ds = LangTopoDataset(
        data_root=cfg.data_root,
        split=cfg.train_split,
        n_frames=cfg.n_frames,
        max_instances=cfg.max_instances,
        image_h=cfg.image_h,
        image_w=cfg.image_w,
        augment=True,
    )
    val_ds = LangTopoDataset(
        data_root=cfg.data_root,
        split=cfg.val_split,
        n_frames=cfg.n_frames,
        max_instances=cfg.max_instances,
        image_h=cfg.image_h,
        image_w=cfg.image_w,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader
