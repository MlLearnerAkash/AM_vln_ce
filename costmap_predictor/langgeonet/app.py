"""
LangGeoNet Gradio App

Interactive demo: upload an image, enter a navigation instruction,
and get a costmap visualization showing predicted geodesic distances.

Usage:
    python app.py --checkpoint checkpoints/best_model.pt
    python app.py --checkpoint checkpoints/best_model.pt --share
"""

import os
import sys
import argparse
import io

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Mask loading  (mirrors inference.py --masks behaviour)
# ---------------------------------------------------------------------------

def load_masks(masks_path: str) -> np.ndarray:
    """
    Load a masks .npy file [K, H, W] and filter out empty masks,
    exactly as predict_frame() does in inference.py.

    Args:
        masks_path: path to masks.npy with shape [K, H, W]

    Returns:
        masks: np.ndarray [K_valid, H, W] bool
    """
    masks = np.load(masks_path)          # [K, H, W]
    valid = masks.any(axis=(1, 2))       # remove all-zero masks
    masks = masks[valid]
    return masks


# ---------------------------------------------------------------------------
# Predictor (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_predictor = None

def get_predictor(checkpoint_path: str, device: str = None):
    global _predictor
    if _predictor is None:
        from inference import LangGeoNetPredictor
        _predictor = LangGeoNetPredictor(checkpoint_path, device=device)
    return _predictor


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def render_costmap_figure(
    image_np: np.ndarray,
    costmap: np.ndarray,
    instruction: str,
    n_objects: int,
) -> np.ndarray:
    """Render a 3-panel costmap figure and return as RGB numpy array."""
    colormap = cm.get_cmap("RdYlGn_r")
    normalized = (costmap - costmap.min()) / (costmap.max() - costmap.min() + 1e-8)
    colored = colormap(normalized)[:, :, :3]
    overlay = np.clip(0.5 * (image_np / 255.0) + 0.5 * colored, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Instruction: "{instruction}"\n{n_objects} objects detected',
        fontsize=11,
        wrap=True,
    )

    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(normalized, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("Predicted Costmap\n(green = near goal, red = far)")
    axes[1].axis("off")
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1)),
        ax=axes[1],
        fraction=0.046,
        label="Normalized distance",
    )

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_per_object_figure(
    image_np: np.ndarray,
    masks: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Render bar chart + closest/farthest overlay as RGB numpy array."""
    K = len(distances)
    order = np.argsort(distances)
    labels = [f"obj_{k}" for k in range(K)]

    colormap = cm.get_cmap("RdYlGn_r")
    norm_d = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bar_colors = colormap(norm_d[order])
    axes[0].barh(range(K), distances[order], color=bar_colors)
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels([labels[i] for i in order], fontsize=max(5, 9 - K // 10))
    axes[0].set_xlabel("Predicted Geodesic Distance")
    axes[0].set_title("Per-Object Distances (sorted)")
    axes[0].invert_yaxis()

    overlay = image_np.astype(np.float32) / 255.0
    closest_k = order[0]
    farthest_k = order[-1]
    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for k, color in [(closest_k, green), (farthest_k, red)]:
        mask = masks[k] > 0
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * color

    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title(
        f"Closest: {labels[closest_k]} ({distances[closest_k]:.3f})  |  "
        f"Farthest: {labels[farthest_k]} ({distances[farthest_k]:.3f})"
    )
    axes[1].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_attention_figure(
    instruction: str,
    attn_weights: np.ndarray,
    n_objects: int,
) -> np.ndarray:
    """
    Heatmap: which instruction tokens each object attends to.
    - Skips the CLIP SOT token (position 0) — handled upstream in inference.py.
    - Applies row-wise min-max normalization so per-object patterns are visible
      even when absolute attention values are small.
    """
    from transformers import CLIPTokenizer  # type: ignore[import]
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    # tokenize returns content tokens only (no SOT/EOS), matching the sliced attn_weights
    tokens = tokenizer.tokenize(instruction)

    K, L = attn_weights.shape
    L_show = min(L, len(tokens))
    labels = [f"obj_{k}" for k in range(K)]

    # Row-wise min-max normalization: makes per-object token preferences visible
    # even when all absolute values are similar (flat raw attention)
    row_min = attn_weights[:, :L_show].min(axis=1, keepdims=True)
    row_max = attn_weights[:, :L_show].max(axis=1, keepdims=True)
    attn_norm = (attn_weights[:, :L_show] - row_min) / (row_max - row_min + 1e-8)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(12, L_show * 0.55), max(3, K * 0.4)),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # Left: raw attention
    im0 = axes[0].imshow(attn_weights[:, :L_show], cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(L_show))
    axes[0].set_xticklabels(tokens[:L_show], rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_title("Raw Cross-Attention")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Right: row-normalised attention (highlights relative preference per object)
    im1 = axes[1].imshow(attn_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xticks(range(L_show))
    axes[1].set_xticklabels(tokens[:L_show], rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(K))
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_title("Row-Normalised Attention\n(per-object token preference)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.suptitle("Cross-Attention: Object × Instruction Token", fontsize=11)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# ---------------------------------------------------------------------------
# Main inference function called by Gradio
# ---------------------------------------------------------------------------

def run_inference(
    image_pil: Image.Image,
    masks_path: str,          # path to masks .npy file
    instruction: str,
    checkpoint_path: str,
    device: str,
):
    if image_pil is None:
        return None, None, None, "Please upload an image."
    if not masks_path or not masks_path.strip():
        return None, None, None, "Please enter the path to a masks .npy file ([K, H, W])."
    if not instruction or not instruction.strip():
        return None, None, None, "Please enter a navigation instruction."

    masks_path = masks_path.strip()
    if not os.path.isfile(masks_path):
        return None, None, None, f"Masks file not found: {masks_path}"

    # Load masks from the .npy file — same as inference.py CLI
    try:
        masks = load_masks(masks_path)   # [K_valid, H, W]
    except Exception as e:
        return None, None, None, f"Failed to load masks: {e}"

    if masks.shape[0] == 0:
        return None, None, None, "All masks are empty after filtering."

    predictor = get_predictor(checkpoint_path, device)
    image_np = np.array(image_pil.convert("RGB"))

    # predict_frame handles the rest identically to inference.py
    distances, costmap, attn = predictor.predict_frame(image_pil, masks, instruction)

    if len(distances) == 0:
        return None, None, None, "No valid objects detected in the image."

    costmap_fig = render_costmap_figure(image_np, costmap, instruction, len(distances))
    per_obj_fig = render_per_object_figure(image_np, masks, distances)
    attn_fig = render_attention_figure(instruction, attn, len(distances)) if attn is not None else None

    status = (
        f"Done.  {len(distances)} objects | "
        f"min dist={distances.min():.4f}  max dist={distances.max():.4f}"
    )
    return costmap_fig, per_obj_fig, attn_fig, status


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_app(checkpoint_path: str, device: str = None, share: bool = False):
    import gradio as gr  # type: ignore[import]
    import torch  # type: ignore[import]

    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-load model so first request is fast
    print(f"Pre-loading model from {checkpoint_path} on {device} …")
    get_predictor(checkpoint_path, device)

    with gr.Blocks(title="LangGeoNet — Costmap Predictor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🗺️ LangGeoNet — Language-Guided Geodesic Costmap
            Upload a scene image and describe where you want to go.
            The model will predict geodesic distances to every object and render a navigation costmap.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Scene Image (RGB)",
                    height=320,
                )
                masks_input = gr.Textbox(
                    label="Masks file path  (.npy, shape [K, H, W])",
                    placeholder="/path/to/masks.npy",
                    lines=1,
                )
                instruction_input = gr.Textbox(
                    label="Navigation Instruction",
                    placeholder='e.g. "Go to the lamp on the nightstand"',
                    lines=2,
                )
                run_btn = gr.Button("Run Inference", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False, lines=1)

            with gr.Column(scale=2):
                costmap_out = gr.Image(
                    label="Costmap Visualization",
                    type="numpy",
                )
                per_obj_out = gr.Image(
                    label="Per-Object Distance Chart",
                    type="numpy",
                )
                attn_out = gr.Image(
                    label="Cross-Attention Map",
                    type="numpy",
                )

        run_btn.click(
            fn=lambda img, masks, txt: run_inference(img, masks, txt, checkpoint_path, device),
            inputs=[image_input, masks_input, instruction_input],
            outputs=[costmap_out, per_obj_out, attn_out, status_box],
            api_name=False,
        )

        gr.Examples(
            examples=[
                [None, "masks.npy", "Go to the lamp on the nightstand"],
                [None, "masks.npy", "Navigate to the chair near the window"],
                [None, "masks.npy", "Move towards the door at the end of the hallway"],
            ],
            inputs=[image_input, masks_input, instruction_input],
            label="Example Instructions (supply your own image + masks path)",
        )

    demo.launch(share=share, show_api=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGeoNet Gradio App")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: cuda / cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port to serve on (default: 7860)",
    )
    args = parser.parse_args()

    # Make sure imports from the same package work when run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    build_app(
        checkpoint_path=args.checkpoint,
        device=args.device,
        share=args.share,
    )
