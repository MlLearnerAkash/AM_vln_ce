"""
Inference script for LangTopoSeg.

Given a trained checkpoint, a navigation instruction, and either a pre-built
frame directory (on-disk layout identical to the training data) or raw NumPy
arrays, this script:

  1. Loads the model and checkpoint.
  2. Runs the forward pass.
  3. Returns (and optionally saves) a structured prediction dict.
  4. Optionally renders a visualisation overlay on the current frame.

Programmatic usage
------------------
    from inference import LangTopoSegInference

    engine = LangTopoSegInference("checkpoints/langtoposeq/epoch_049.pt")
    result = engine.predict_from_dir(
        window_dirs=["/data/episode_0/frame_000", ..., "/data/episode_0/frame_004"],
        instruction="Turn left and walk towards the red sofa.",
    )
    print(result["pred_e3d"])   # [K]  numpy array
    print(result["node_mask"])  # [K]
    print(result["dir_2d"])     # [2]

CLI usage
---------
    python inference.py \\
        --ckpt_path checkpoints/langtoposeq/epoch_049.pt \\
        --episode_dir /data/episode_42 \\
        --instruction "Go past the table and stop at the window." \\
        --visualise

    python inference.py \\
        --ckpt_path checkpoints/langtoposeq/epoch_049.pt \\
        --episode_dir /data/episode_42 \\
        --current_frame 10 \\
        --instruction "Turn left" \\
        --output_dir results/episode_42
"""

import os
import argparse
import logging
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from config  import LangTopoSegConfig
from dataset import load_frame
from model   import LangTopoSeg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Inference engine
# ─────────────────────────────────────────────────────────────────────────────

class LangTopoSegInference:
    """
    High-level inference wrapper.

    Parameters
    ----------
    ckpt_path : str
        Path to a LangTopoSeg checkpoint (.pt file).
    cfg       : LangTopoSegConfig, optional
        If None, the default config is used.  Values stored in the checkpoint
        (if any) take precedence for model architecture fields.
    device    : str | torch.device, optional
        Defaults to CUDA if available.
    """

    def __init__(
        self,
        ckpt_path: str,
        cfg: Optional[LangTopoSegConfig] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        cfg = cfg or LangTopoSegConfig()
        cfg.ckpt_path = ckpt_path

        ckpt = torch.load(ckpt_path, map_location=self.device)
        # Override architecture params from checkpoint config if saved
        if "config" in ckpt:
            saved_cfg = ckpt["config"]
            for attr in ("embed_dim", "vision_model", "text_model", "n_attn_heads",
                         "gat_heads", "n_gat_layers", "tau", "temp", "dir_scale",
                         "image_h", "image_w", "max_instances", "n_frames"):
                if hasattr(saved_cfg, attr):
                    setattr(cfg, attr, getattr(saved_cfg, attr))

        self.cfg   = cfg
        self.model = LangTopoSeg(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        logger.info(
            f"Loaded LangTopoSeg from '{ckpt_path}'  "
            f"(epoch {ckpt.get('epoch', '?')})  device={self.device}"
        )

    # ── Core prediction ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_from_arrays(
        self,
        rgb_window:       np.ndarray,   # [T, H, W, 3]  uint8
        masks_window:     np.ndarray,   # [T, K, H, W]  uint8/float
        centroids_window: np.ndarray,   # [T, K, 2]     float32
        areas_window:     np.ndarray,   # [T, K]        float32
        k_valid_window:   List[int],    # [T]
        instruction:      str,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference given pre-processed NumPy arrays for a temporal window.

        Returns
        -------
        dict with keys:
            pred_e3d   : [K]      float32  per-node predicted e3d score
            pred_edges : [K, K]   float32  pairwise edge weights
            node_mask  : [K]      float32  relevance probabilities
            dir_2d     : [2]      float32  image-plane direction (unit vector)
        """
        T   = len(k_valid_window)
        H   = self.cfg.image_h
        W   = self.cfg.image_w
        K   = self.cfg.max_instances

        # Normalise RGB to [0,1] and resize if needed
        def _prep_rgb(img: np.ndarray) -> torch.Tensor:
            pil = Image.fromarray(img).convert("RGB").resize((W, H), Image.BILINEAR)
            return torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)

        def _prep_masks(masks: np.ndarray) -> torch.Tensor:
            # Ensure [K, H, W] with padding
            assert masks.ndim == 3
            kk = masks.shape[0]
            out = np.zeros((K, H, W), dtype=np.float32)
            for i in range(min(kk, K)):
                m = masks[i].astype(np.float32)
                if m.shape != (H, W):
                    m = np.array(Image.fromarray(
                        (m * 255).astype(np.uint8)).resize((W, H), Image.NEAREST),
                        dtype=np.float32) / 255.0
                out[i] = m
            return torch.from_numpy(out)

        def _pad_1d(arr: np.ndarray) -> torch.Tensor:
            kk = len(arr)
            out = np.zeros(K, dtype=np.float32)
            out[:min(kk, K)] = arr[:min(kk, K)]
            return torch.from_numpy(out)

        def _pad_2d(arr: np.ndarray) -> torch.Tensor:
            kk = arr.shape[0]
            out = np.zeros((K, arr.shape[1]), dtype=np.float32)
            out[:min(kk, K)] = arr[:min(kk, K)]
            return torch.from_numpy(out)

        rgb_t       = torch.stack([_prep_rgb(rgb_window[t])                for t in range(T)])         # [T, 3, H, W]
        masks_t     = torch.stack([_prep_masks(masks_window[t])            for t in range(T)])         # [T, K, H, W]
        centroids_t = torch.stack([_pad_2d(centroids_window[t])            for t in range(T)])         # [T, K, 2]
        areas_t     = torch.stack([_pad_1d(areas_window[t])                for t in range(T)])         # [T, K]
        k_valid_t   = torch.tensor(k_valid_window, dtype=torch.long)                                   # [T]

        # Add batch dimension (B=1)
        rgb_t       = rgb_t.unsqueeze(0).to(self.device)
        masks_t     = masks_t.unsqueeze(0).to(self.device)
        centroids_t = centroids_t.unsqueeze(0).to(self.device)
        areas_t     = areas_t.unsqueeze(0).to(self.device)
        k_valid_t   = k_valid_t.unsqueeze(0).to(self.device)

        outputs = self.model(
            rgb=rgb_t, masks=masks_t, centroids=centroids_t,
            areas=areas_t, k_valid=k_valid_t, instructions=[instruction],
        )

        return {
            "pred_e3d":   outputs["pred_e3d"][0].cpu().numpy(),     # [K]
            "pred_edges": outputs["pred_edges"][0].cpu().numpy(),   # [K, K]
            "node_mask":  outputs["node_mask"][0].cpu().numpy(),    # [K]
            "dir_2d":     outputs["dir_2d"][0].cpu().numpy(),       # [2]
        }

    @torch.no_grad()
    def predict_from_dir(
        self,
        window_dirs: List[str],
        instruction: str,
    ) -> Dict[str, np.ndarray]:
        """
        Load a temporal window of frame directories and run inference.

        Parameters
        ----------
        window_dirs : list of str
            List of frame directory paths in chronological order.
            Must contain exactly (n_frames + 1) entries.
        instruction : str
            Navigation instruction.
        """
        assert len(window_dirs) == self.cfg.n_frames + 1, (
            f"Expected {self.cfg.n_frames + 1} directories, got {len(window_dirs)}"
        )

        frame_data = [
            load_frame(d, self.cfg.max_instances, self.cfg.image_h, self.cfg.image_w)
            for d in window_dirs
        ]

        # Assemble into batch of T frames (no padding needed here — load_frame handles it)
        T  = len(frame_data)
        K  = self.cfg.max_instances
        H, W = self.cfg.image_h, self.cfg.image_w

        rgb_t       = torch.stack([fd["rgb"]       for fd in frame_data]).unsqueeze(0).to(self.device)
        masks_t     = torch.stack([fd["masks"]     for fd in frame_data]).unsqueeze(0).to(self.device)
        centroids_t = torch.stack([fd["centroids"] for fd in frame_data]).unsqueeze(0).to(self.device)
        areas_t     = torch.stack([fd["areas"]     for fd in frame_data]).unsqueeze(0).to(self.device)
        k_valid_t   = torch.tensor([[fd["k_valid"] for fd in frame_data]], dtype=torch.long).to(self.device)

        outputs = self.model(
            rgb=rgb_t, masks=masks_t, centroids=centroids_t,
            areas=areas_t, k_valid=k_valid_t, instructions=[instruction],
        )

        k_valid_last = int(frame_data[-1]["k_valid"])
        return {
            "pred_e3d":      outputs["pred_e3d"][0].cpu().numpy(),     # [K]
            "pred_edges":    outputs["pred_edges"][0].cpu().numpy(),   # [K, K]
            "node_mask":     outputs["node_mask"][0].cpu().numpy(),    # [K]
            "dir_2d":        outputs["dir_2d"][0].cpu().numpy(),       # [2]
            "k_valid":       k_valid_last,
            "frame_dir":     window_dirs[-1],
        }

    @torch.no_grad()
    def predict_episode(
        self,
        episode_dir: str,
        instruction: str,
        current_frame: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper: given an episode directory and an (optional)
        current frame index, pick the appropriate window and run inference.

        Parameters
        ----------
        episode_dir    : str   path to episode_* directory
        instruction    : str
        current_frame  : int, optional   0-based index of the target frame.
                         Defaults to the last available frame.
        """
        ep_dir     = Path(episode_dir)
        frame_dirs = sorted(ep_dir.glob("frame_*"))
        if not frame_dirs:
            raise FileNotFoundError(f"No frame_* directories found in {episode_dir}")

        n_frames   = self.cfg.n_frames
        win_size   = n_frames + 1

        if current_frame is None:
            current_frame = len(frame_dirs) - 1

        start = max(0, current_frame - n_frames)
        # Pad with the first available frame if window runs before frame_000
        window = []
        for t in range(win_size):
            idx = start + t
            idx = min(idx, len(frame_dirs) - 1)
            window.append(str(frame_dirs[idx]))

        return self.predict_from_dir(window, instruction)


# ─────────────────────────────────────────────────────────────────────────────
# Optional visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _colour_for_score(score: float) -> Tuple[int, int, int]:
    """Map a score in [0,1] to an RGB colour (blue→green→red gradient)."""
    r = int(min(255, max(0, score * 510 - 255)))
    g = int(min(255, max(0, 255 - abs(score * 510 - 255))))
    b = int(min(255, max(0, 255 - score * 510)))
    return (r, g, b)


def visualise(
    frame_dir:   str,
    result:      Dict[str, np.ndarray],
    cfg:         LangTopoSegConfig,
    instruction: str,
    out_path:    Optional[str] = None,
) -> Image.Image:
    """
    Overlay predicted e3d heat-map and node selection on the current RGB frame.

    Each valid instance mask is tinted by its pred_e3d score:
      blue  = low score (near)
      red   = high score (far / salient)

    Nodes that are selected (node_mask ≥ 0.5) get a white border.
    The predicted direction arrow is drawn at the image centre.

    Returns the PIL Image (also saved to out_path if provided).
    """
    rgb_path = os.path.join(frame_dir, "rgb.png")
    img  = Image.open(rgb_path).convert("RGBA").resize(
        (cfg.image_w, cfg.image_h), Image.BILINEAR
    )
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    masks_path = os.path.join(frame_dir, "masks.npy")
    if os.path.exists(masks_path):
        masks_np = np.load(masks_path)          # [K_real, H, W]
        K_real   = len(masks_np)
        k_valid  = result.get("k_valid", K_real)
        pred_e3d = result["pred_e3d"]
        node_mask = result["node_mask"]

        H_m, W_m = masks_np.shape[1], masks_np.shape[2]
        scale_x  = cfg.image_w  / W_m
        scale_y  = cfg.image_h  / H_m

        for i in range(min(k_valid, K_real, cfg.max_instances)):
            score = float(np.clip(pred_e3d[i], 0.0, 1.0))
            r, g, b = _colour_for_score(score)
            alpha   = 100  # semi-transparent

            mask = masks_np[i]  # [H_m, W_m]
            # Scale mask to display size
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (cfg.image_w, cfg.image_h), Image.NEAREST
            )
            colour_layer = Image.new("RGBA", img.size, (r, g, b, alpha))
            overlay.paste(colour_layer, mask=mask_img)

            # Draw border for selected nodes
            if node_mask[i] >= 0.5:
                ys, xs = np.where(mask > 0)
                if len(xs):
                    x0 = int(xs.min() * scale_x)
                    y0 = int(ys.min() * scale_y)
                    x1 = int(xs.max() * scale_x)
                    y1 = int(ys.max() * scale_y)
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 220), width=2)

    # Blend
    composite = Image.alpha_composite(img, overlay).convert("RGB")
    draw_rgb  = ImageDraw.Draw(composite)

    # Draw direction arrow from image centre
    dir_2d = result["dir_2d"]
    cx, cy = cfg.image_w // 2, cfg.image_h // 2
    scale  = min(cfg.image_w, cfg.image_h) * 0.2
    ex = int(cx + dir_2d[0] * scale)
    ey = int(cy + dir_2d[1] * scale)
    draw_rgb.line([(cx, cy), (ex, ey)], fill=(255, 215, 0), width=3)
    # Arrowhead
    angle = math.atan2(ey - cy, ex - cx)
    for delta in (math.pi * 0.8, -math.pi * 0.8):
        ax = int(ex + 10 * math.cos(angle + delta))
        ay = int(ey + 10 * math.sin(angle + delta))
        draw_rgb.line([(ex, ey), (ax, ay)], fill=(255, 215, 0), width=2)

    # Instruction text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    draw_rgb.text((4, 4), instruction[:80], fill=(255, 255, 255), font=font)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        composite.save(out_path)
        logger.info(f"Visualisation saved → {out_path}")

    return composite


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Run LangTopoSeg inference")
    p.add_argument("--ckpt_path",     required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--episode_dir",   required=True,
                   help="Path to episode_* directory")
    p.add_argument("--instruction",   required=True,
                   help="Navigation instruction string")
    p.add_argument("--current_frame", type=int, default=None,
                   help="0-based index of the target frame (default: last)")
    p.add_argument("--output_dir",    default=None,
                   help="Directory to save outputs (predictions .npz + optional vis)")
    p.add_argument("--visualise",     action="store_true",
                   help="Produce and save a visualisation overlay image")
    p.add_argument("--device",        default=None)
    return p.parse_args()


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()

    engine = LangTopoSegInference(
        ckpt_path=args.ckpt_path,
        device=args.device,
    )

    result = engine.predict_episode(
        episode_dir   = args.episode_dir,
        instruction   = args.instruction,
        current_frame = args.current_frame,
    )

    k = result["k_valid"]
    logger.info(f"Inference complete.  k_valid={k}")
    logger.info(f"  pred_e3d   (top-{min(5,k)}): "
                f"{result['pred_e3d'][:k][:5].round(3)}")
    logger.info(f"  node_mask  (top-{min(5,k)}): "
                f"{result['node_mask'][:k][:5].round(3)}")
    logger.info(f"  dir_2d: {result['dir_2d'].round(3)}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_npz = os.path.join(args.output_dir, "prediction.npz")
        np.savez(out_npz, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        logger.info(f"Predictions saved → {out_npz}")

        if args.visualise:
            vis_path = os.path.join(args.output_dir, "overlay.png")
            visualise(
                frame_dir   = result["frame_dir"],
                result      = result,
                cfg         = engine.cfg,
                instruction = args.instruction,
                out_path    = vis_path,
            )


if __name__ == "__main__":
    main()
