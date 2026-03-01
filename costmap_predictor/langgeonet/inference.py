"""
LangGeoNet Inference & Visualization.

Usage:
    # Single frame
    python inference.py --checkpoint best_model.pt \
        --image frame.png --masks masks.npy --class_ids class_ids.npy \
        --instruction "Go to the lamp on the nightstand" \
        --output ./output

    # Full dataset evaluation
    python inference.py --checkpoint best_model.pt \
        --data_root ./data --eval_split val

    # Episode inference (all frames, one instruction)
    python inference.py --checkpoint best_model.pt \
        --episode_dir ./data/episode_000 --output ./output
"""

import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, BertTokenizer

from model import LangGeoNet


class LangGeoNetPredictor:
    """
    Inference wrapper. Handles preprocessing, prediction, and costmap generation.
    """

    def __init__(self, checkpoint_path, device=None):
        """
        Args:
            checkpoint_path: path to saved .pt checkpoint
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint & config
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        # Build model from saved config
        self.model = LangGeoNet(
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            num_classes=cfg["num_classes"],
            clip_model_name=cfg["clip_model"],
            bert_model_name=cfg["bert_model"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Tokenizer / processor (same as training)
        self.clip_processor = CLIPProcessor.from_pretrained(cfg["clip_model"])
        self.bert_tokenizer = BertTokenizer.from_pretrained(cfg["bert_model"])

        print(f"[LangGeoNetPredictor] Loaded epoch {ckpt['epoch']} "
              f"(val MAE={ckpt.get('best_val_mae', '?'):.4f}) on {self.device}")

    # ----------------------------------------------------------
    # Core prediction
    # ----------------------------------------------------------

    @torch.no_grad()
    def predict_frame(self, image, masks, class_ids, instruction):
        """
        Predict geodesic distance for every object in ONE frame.

        Args:
            image:       PIL.Image  or  np.ndarray [H, W, 3] (uint8 RGB)
            masks:       np.ndarray [K, H, W] bool/uint8 instance masks
            class_ids:   np.ndarray [K] int class IDs
            instruction: str, the navigation instruction

        Returns:
            distances: np.ndarray [K]  predicted normalized geodesic ∈ [0,1]
            costmap:   np.ndarray [H, W] float32, each pixel = its object's distance
            attn_map:  np.ndarray [K, L_actual] cross-attention weights (last layer)
        """
        # --- preprocess image ---
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(self.device)  # [1, 3, 224, 224]

        # --- preprocess instruction ---
        enc = self.bert_tokenizer(
            instruction, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)            # [1, L]
        attention_mask = enc["attention_mask"].to(self.device)   # [1, L]

        # --- preprocess masks ---
        masks_t = torch.from_numpy(masks.astype(bool)).to(self.device)     # [K, H, W]
        class_ids_t = torch.from_numpy(class_ids.astype(np.int64)).to(self.device)  # [K]

        # --- forward ---
        predictions, attn_weights_all = self.model(
            images=pixel_values,
            masks_list=[masks_t],
            class_ids_list=[class_ids_t],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        distances = predictions[0].cpu().numpy()  # [K]

        # --- build WayObject Costmap ---
        K, H, W = masks.shape
        costmap = np.ones((H, W), dtype=np.float32)  # background = 1.0
        for k in range(K):
            costmap[masks[k] > 0] = distances[k]

        # --- extract attention (last layer) ---
        attn_map = None
        if attn_weights_all:
            last = attn_weights_all[-1]  # [1, K_padded, L]
            L_actual = int(attention_mask.sum().item())
            attn_map = last[0, :K, :L_actual].cpu().numpy()  # [K, L_actual]

        return distances, costmap, attn_map

    # ----------------------------------------------------------
    # Episode-level (all frames share one instruction)
    # ----------------------------------------------------------

    @torch.no_grad()
    def predict_episode(self, episode_dir):
        """
        Run inference on all frames of an episode.

        Args:
            episode_dir: path containing instruction.txt + frame_*/

        Returns:
            results: list of dicts, one per frame:
                { "distances": [K], "costmap": [H,W], "frame_dir": str }
        """
        # Read instruction
        with open(os.path.join(episode_dir, "instruction.txt")) as f:
            instruction = f.read().strip()

        frame_dirs = sorted(glob.glob(os.path.join(episode_dir, "frame_*")))
        results = []

        for frame_dir in frame_dirs:
            rgb_path = os.path.join(frame_dir, "rgb.png")
            masks_path = os.path.join(frame_dir, "masks.npy")
            cids_path = os.path.join(frame_dir, "class_ids.npy")

            if not all(os.path.exists(p) for p in [rgb_path, masks_path, cids_path]):
                continue

            image = Image.open(rgb_path).convert("RGB")
            masks = np.load(masks_path)
            class_ids = np.load(cids_path)

            distances, costmap, attn = self.predict_frame(
                image, masks, class_ids, instruction
            )

            results.append({
                "frame_dir": frame_dir,
                "distances": distances,
                "costmap": costmap,
                "attn_map": attn,
            })

        return instruction, results

    # ----------------------------------------------------------
    # Batch prediction (multiple frames, same instruction)
    # ----------------------------------------------------------

    @torch.no_grad()
    def predict_batch(self, images, masks_list, class_ids_list, instruction):
        """
        Predict geodesic distances for a BATCH of frames (shared instruction).

        This is more efficient than calling predict_frame() in a loop because
        the language encoder runs once and the visual encoder processes all
        frames in parallel.

        Args:
            images:         list of PIL.Image (len B)
            masks_list:     list of np.ndarray [K_b, H, W] (len B)
            class_ids_list: list of np.ndarray [K_b] (len B)
            instruction:    str (shared)

        Returns:
            all_distances: list of np.ndarray [K_b] (len B)
            all_costmaps:  list of np.ndarray [H, W] (len B)
        """
        B = len(images)

        # --- preprocess images ---
        pv_list = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            pv = self.clip_processor(images=img, return_tensors="pt")["pixel_values"]
            pv_list.append(pv.squeeze(0))
        pixel_values = torch.stack(pv_list).to(self.device)  # [B, 3, 224, 224]

        # --- preprocess instruction (same for all frames) ---
        enc = self.bert_tokenizer(
            [instruction] * B, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # --- preprocess masks ---
        masks_t = [torch.from_numpy(m.astype(bool)).to(self.device) for m in masks_list]
        cids_t = [torch.from_numpy(c.astype(np.int64)).to(self.device) for c in class_ids_list]

        # --- forward ---
        predictions, _ = self.model(
            images=pixel_values,
            masks_list=masks_t,
            class_ids_list=cids_t,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        all_distances = []
        all_costmaps = []

        for b in range(B):
            dists = predictions[b].cpu().numpy()
            K, H, W = masks_list[b].shape
            costmap = np.ones((H, W), dtype=np.float32)
            for k in range(K):
                costmap[masks_list[b][k] > 0] = dists[k]
            all_distances.append(dists)
            all_costmaps.append(costmap)

        return all_distances, all_costmaps


# ==============================================================
# Visualization
# ==============================================================

def visualize_costmap(image, costmap, save_path=None, title="WayObject Costmap"):
    """Overlay costmap on RGB. Green=close, Red=far."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if isinstance(image, Image.Image):
        image = np.array(image)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    cmap = cm.get_cmap("RdYlGn_r")
    colored = cmap(costmap)[:, :, :3]
    axes[1].imshow(colored)
    axes[1].set_title(title)
    axes[1].axis("off")
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=axes[1], fraction=0.046)

    overlay = 0.5 * (image / 255.0) + 0.5 * colored
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_per_object(image, masks, distances, class_names=None, save_path=None):
    """Bar chart of per-object predicted geodesic + overlay of top/bottom objects."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if isinstance(image, Image.Image):
        image = np.array(image)

    K = len(distances)
    labels = [class_names[k] if class_names else f"obj_{k}" for k in range(K)]
    order = np.argsort(distances)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart sorted by distance
    colors = cm.get_cmap("RdYlGn_r")(distances[order])
    axes[0].barh(range(K), distances[order], color=colors)
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels([labels[i] for i in order], fontsize=8)
    axes[0].set_xlabel("Predicted Geodesic Distance (normalized)")
    axes[0].set_title("Per-Object Distances")
    axes[0].set_xlim(0, 1)

    # Image with closest (green) and farthest (red) object highlighted
    overlay = image.copy().astype(np.float32) / 255.0
    closest_k = order[0]
    farthest_k = order[-1]

    green_overlay = np.zeros_like(overlay)
    green_overlay[:, :, 1] = 1.0
    red_overlay = np.zeros_like(overlay)
    red_overlay[:, :, 0] = 1.0

    mask_close = masks[closest_k] > 0
    mask_far = masks[farthest_k] > 0

    overlay[mask_close] = 0.5 * overlay[mask_close] + 0.5 * green_overlay[mask_close]
    overlay[mask_far] = 0.5 * overlay[mask_far] + 0.5 * red_overlay[mask_far]

    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title(f"Closest: {labels[closest_k]} ({distances[closest_k]:.3f})  |  "
                       f"Farthest: {labels[farthest_k]} ({distances[farthest_k]:.3f})")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_attention(instruction, attn_weights, object_labels, save_path=None):
    """Heatmap: which instruction tokens each object attends to."""
    import matplotlib.pyplot as plt

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = ["[CLS]"] + tokenizer.tokenize(instruction) + ["[SEP]"]

    K, L = attn_weights.shape
    L_show = min(L, len(tokens))

    fig, ax = plt.subplots(figsize=(max(12, L_show * 0.5), max(3, K * 0.4)))
    im = ax.imshow(attn_weights[:, :L_show], cmap="Blues", aspect="auto")
    ax.set_xticks(range(L_show))
    ax.set_xticklabels(tokens[:L_show], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels(object_labels, fontsize=9)
    ax.set_title("Cross-Attention: Object x Instruction Token")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ==============================================================
# Dataset-level Evaluation
# ==============================================================

def evaluate_dataset(predictor, data_root, split="val"):
    """
    Run evaluation on an entire dataset split.

    Returns:
        metrics dict
    """
    from dataset import LangGeoNetDataset
    from train import compute_metrics

    ds = LangGeoNetDataset(data_root, split=split)
    all_preds, all_gts = [], []

    for i in range(len(ds)):
        sample = ds[i]
        frame_dir = sample["frame_dir"]
        ep_dir = sample["episode_dir"]

        image = Image.open(os.path.join(frame_dir, "rgb.png")).convert("RGB")
        masks = sample["masks"].numpy()
        class_ids = sample["class_ids"].numpy()
        gt = sample["geodesic_distances"].numpy()

        with open(os.path.join(ep_dir, "instruction.txt")) as f:
            instruction = f.read().strip()

        dists, _, _ = predictor.predict_frame(image, masks, class_ids, instruction)

        all_preds.append(dists)
        all_gts.append(gt)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds)}")

    metrics = compute_metrics(all_preds, all_gts)

    print(f"\n{'='*40}")
    print(f"Evaluation on {split} ({len(ds)} frames)")
    print(f"{'='*40}")
    print(f"  MAE:              {metrics['mae']:.4f}")
    print(f"  RMSE:             {metrics['rmse']:.4f}")
    print(f"  Ranking Accuracy: {metrics['ranking_accuracy']:.4f}")
    print(f"  Spearman:         {metrics['spearman']:.4f}")
    print(f"  acc@0.05:         {metrics['acc@0.05']:.4f}")
    print(f"  acc@0.10:         {metrics['acc@0.10']:.4f}")
    print(f"  acc@0.20:         {metrics['acc@0.20']:.4f}")

    return metrics


# ==============================================================
# CLI
# ==============================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LangGeoNet Inference")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--device", default=None)

    # Mode 1: single frame
    p.add_argument("--image", default=None, help="Path to RGB image")
    p.add_argument("--masks", default=None, help="Path to masks.npy [K, H, W]")
    p.add_argument("--class_ids", default=None, help="Path to class_ids.npy [K]")
    p.add_argument("--instruction", default=None, help="Navigation instruction string")

    # Mode 2: episode
    p.add_argument("--episode_dir", default=None, help="Path to episode directory")

    # Mode 3: dataset eval
    p.add_argument("--data_root", default=None, help="Path to dataset root")
    p.add_argument("--eval_split", default="val")

    p.add_argument("--output", default="./output", help="Output directory")
    a = p.parse_args()

    predictor = LangGeoNetPredictor(a.checkpoint, device=a.device)
    os.makedirs(a.output, exist_ok=True)

    # ---- Mode 1: Single frame ----
    if a.image and a.masks and a.instruction:
        image = Image.open(a.image).convert("RGB")
        masks = np.load(a.masks)
        class_ids = np.load(a.class_ids) if a.class_ids else np.zeros(masks.shape[0], dtype=np.int64)

        dists, costmap, attn = predictor.predict_frame(image, masks, class_ids, a.instruction)

        print(f"\nInstruction: {a.instruction}")
        print(f"Objects: {masks.shape[0]}")
        print(f"\nPredicted geodesic distances (sorted by distance):")
        for k in np.argsort(dists):
            print(f"  obj_{k:2d} (class {class_ids[k]:3d}): {dists[k]:.4f}")

        np.save(os.path.join(a.output, "distances.npy"), dists)
        np.save(os.path.join(a.output, "costmap.npy"), costmap)

        visualize_costmap(image, costmap, os.path.join(a.output, "costmap.png"))
        visualize_per_object(image, masks, dists, save_path=os.path.join(a.output, "per_object.png"))
        if attn is not None:
            labels = [f"obj_{k}" for k in range(masks.shape[0])]
            visualize_attention(a.instruction, attn, labels,
                                save_path=os.path.join(a.output, "attention.png"))

    # ---- Mode 2: Episode ----
    elif a.episode_dir:
        instruction, results = predictor.predict_episode(a.episode_dir)
        print(f"\nInstruction: {instruction}")
        print(f"Frames: {len(results)}")

        for i, r in enumerate(results):
            print(f"\n  Frame {i}: {r['distances'].shape[0]} objects, "
                  f"min_dist={r['distances'].min():.4f}, max_dist={r['distances'].max():.4f}")

            # Save per-frame outputs
            frame_out = os.path.join(a.output, f"frame_{i:03d}")
            os.makedirs(frame_out, exist_ok=True)
            np.save(os.path.join(frame_out, "distances.npy"), r["distances"])
            np.save(os.path.join(frame_out, "costmap.npy"), r["costmap"])

            # Visualize
            rgb_path = os.path.join(r["frame_dir"], "rgb.png")
            if os.path.exists(rgb_path):
                img = Image.open(rgb_path).convert("RGB")
                visualize_costmap(img, r["costmap"],
                                  os.path.join(frame_out, "costmap.png"),
                                  title=f"Frame {i} Costmap")

    # ---- Mode 3: Dataset eval ----
    elif a.data_root:
        evaluate_dataset(predictor, a.data_root, a.eval_split)

    else:
        print("Provide one of:")
        print("  --image + --masks + --instruction  (single frame)")
        print("  --episode_dir                      (full episode)")
        print("  --data_root                        (dataset eval)")
