"""
LangTopoSeg configuration dataclass.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LangTopoSegConfig:
    # ── Data ─────────────────────────────────────────────────────────────
    data_root: str = "/media/opervu-user/Data2/ws/data_langgeonet_e3d"
    train_split: str = "train"
    val_split: str = "train"
    image_h: int = 120          # frame height (train split default)
    image_w: int = 160          # frame width
    n_frames: int = 4           # number of past frames in temporal window (+ current = n+1 total)
    max_instances: int = 32     # max K per frame (pad to this)

    # ── Model ─────────────────────────────────────────────────────────────
    embed_dim: int = 256        # shared embedding dimension D
    vision_model: str = "facebook/dinov2-small"   # HuggingFace model id
    text_model: str = "openai/clip-vit-base-patch32"
    n_attn_heads: int = 4       # cross-attention heads
    gat_heads: int = 4          # attention heads for intra-frame GAT
    n_gat_layers: int = 2       # rounds of intra-frame message passing
    tau: float = 0.5            # node selection threshold
    temp: float = 0.1           # sigmoid temperature for node gate
    dir_scale: float = 5.0      # sharpness of directional gate sigmoid

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip: float = 1.0

    # Loss weights
    lambda_obs: float = 1.0     # observed edge MSE
    lambda_node: float = 0.5    # node BCE
    lambda_dir: float = 0.3     # direction head regression
    lambda_rank: float = 0.2    # ranking margin
    lambda_sym: float = 0.1     # symmetry
    rank_margin: float = 0.05

    # ── Logging / Checkpoints ─────────────────────────────────────────────
    log_dir: str = "./logs/langtoposeq"
    ckpt_dir: str = "./checkpoints/langtoposeq"
    save_every: int = 5         # epochs between checkpoints
    eval_every: int = 2

    # ── Inference ─────────────────────────────────────────────────────────
    ckpt_path: Optional[str] = None

    # Direction word priors for L_dir supervision
    direction_priors: dict = field(default_factory=lambda: {
        "left":    [-1.0,  0.0],
        "right":   [ 1.0,  0.0],
        "up":      [ 0.0, -1.0],
        "down":    [ 0.0,  1.0],
    })
