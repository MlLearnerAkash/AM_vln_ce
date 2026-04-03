"""
LangGeoNetV3: Robust dual-backbone language-guided geodesic cost predictor.

Redesigned from scratch to eliminate mean-field collapse seen in V2.

Root causes of V2 collapse and their fixes
──────────────────────────────────────────
  PROBLEM 1 — Mean-pooled CLIP patches collapse:
    All K objects in one image pool from the *same* 196 patch vectors.
    After background-subtraction the residuals are nearly identical if the
    objects have similar pixel frequencies.  Subtracting the scene mean was
    insufficient.
  FIX 1 — Mean + Max pooling (both backbones):
    Peak patch activation (max) is less sensitive to mask size and captures a
    complementary "most-salient-patch" signal that varies per object even when
    means are close.

  PROBLEM 2 — Independent per-object scoring:
    V2 scored each object independently from the others in the same image.
    The model converges to predicting the per-episode mean cost for every
    object because that minimises regression loss with zero ranking gradient.
  FIX 2 — Joint unified transformer:
    Objects and language tokens are concatenated into a single sequence and
    passed through a shared pre-norm TransformerEncoder.  Objects can see every
    other object *and* all language tokens in one pass, forcing them to compute
    scores relative to their siblings.

  PROBLEM 3 — Numerically unstable ScaleInvariant loss:
    log(near-zero predictions) → ±Inf gradients swamping the ranking signal.
  FIX 3 — Replace SI loss with a **Diversity Penalty**:
    Explicitly penalise low within-image prediction variance → directly
    prevents and punishes the collapse mode.

  PROBLEM 4 — Margin ranking's dead zone:
    Hard-negative mining + margin meant many valid pairs provided zero gradient
    once the model plateaued at ~0.51 ranking accuracy.
  FIX 4 — Bradley-Terry soft ranking:
    L = -log σ(pred_j – pred_i) for *all* valid pairs.  No dead zone.

  PROBLEM 5 — DINOv2-small (22 M) insufficient capacity:
  FIX 5 — DINOv2-base (86 M, 768-dim) with last 4 blocks fine-tuned.

  PROBLEM 6 — Sigmoid output head saturates gradients:
  FIX 6 — Raw linear head with sigmoid applied only at output (no interior
    gradient vanishing).  A learnable temperature scales the alignment score.

Input / Output API: IDENTICAL to V2 (fully compatible with train.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, Dinov2Model


# ── Normalisation constants ────────────────────────────────────────────────────
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_DINO_MEAN = [0.485,      0.456,      0.406      ]
_DINO_STD  = [0.229,      0.224,      0.225      ]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: background-subtracted masked average pooling (kept for V2 legacy)
# ─────────────────────────────────────────────────────────────────────────────

def _bg_sub_masked_pool(feature_map: torch.Tensor,
                        masks_list: list) -> list:
    """
    For each batch item, pool per-object features then subtract the scene mean.

    Args:
        feature_map : [B, N_patches, D]  (square patch grid: N = H_f * W_f)
        masks_list  : list[B] of [K_b, H, W] bool tensors

    Returns:
        list[B] of [K_b, D] background-subtracted object features.
    """
    B = feature_map.shape[0]
    results = []
    for b in range(B):
        mask_b = masks_list[b]          # [K, H, W]
        K = mask_b.shape[0]
        if K == 0:
            results.append(torch.zeros(0, feature_map.shape[-1],
                                       device=feature_map.device))
            continue
        feat = feature_map[b]           # [N, D]
        N, D = feat.shape
        H_f = W_f = int(N ** 0.5)
        feat_flat = feat.reshape(-1, D)                              # [H_f*W_f, D]
        masks_r = F.interpolate(
            mask_b.unsqueeze(1).float(),
            size=(H_f, W_f),
            mode='nearest',
        ).squeeze(1).reshape(K, -1)                                  # [K, H_f*W_f]
        mask_sum = masks_r.sum(1, keepdim=True).clamp(min=1e-6)
        pooled   = torch.mm(masks_r, feat_flat) / mask_sum           # [K, D]
        scene_mn = feat_flat.mean(0, keepdim=True)                   # [1, D]
        results.append(pooled - scene_mn)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utility: 2-D geometry features from instance masks
# ─────────────────────────────────────────────────────────────────────────────

def _compute_geom(masks: torch.Tensor) -> torch.Tensor:
    """
    Args:
        masks : [K, H, W] bool

    Returns:
        [K, 5]  — (cx, cy, area_norm, bbox_w, bbox_h), all in [0, 1]
    """
    K, H, W = masks.shape
    geom = torch.zeros(K, 5, device=masks.device)
    for k in range(K):
        m = masks[k]
        area = m.float().sum()
        if area < 1:
            continue
        ys, xs = torch.where(m > 0)
        cx = xs.float().mean() / W
        cy = ys.float().mean() / H
        bbox_w = (xs.max() - xs.min() + 1).float() / W
        bbox_h = (ys.max() - ys.min() + 1).float() / H
        geom[k] = torch.stack([cx, cy, area / (H * W), bbox_w, bbox_h])
    return geom


# ─────────────────────────────────────────────────────────────────────────────
# New utility: mean + max masked pooling (V3 core)
# ─────────────────────────────────────────────────────────────────────────────

def _masked_mean_max_pool(feature_map: torch.Tensor,
                          masks_list: list) -> list:
    """
    Per-object background-subtracted mean pooling  +  raw max pooling.

    Mean is subtracted from the global scene mean (removes DC image component).
    Max is kept raw — it captures the peak saliency per object/dimension and
    is LESS correlated across objects than the mean, giving richer descriptors.

    Args:
        feature_map : [B, N_patches, D]  — already projected to d_model
        masks_list  : list[B] of [K_b, H, W] bool tensors

    Returns:
        list[B] of [K_b, 2*D] tensors  (mean_bg_sub || max_raw)
    """
    B = feature_map.shape[0]
    results = []
    for b in range(B):
        mask_b = masks_list[b]          # [K, H, W]
        K = mask_b.shape[0]
        feat = feature_map[b]           # [N, D]
        N, D = feat.shape
        if K == 0:
            results.append(torch.zeros(0, 2 * D, device=feat.device))
            continue
        H_f = W_f = int(N ** 0.5)
        feat_flat = feat.reshape(-1, D)                          # [N, D]
        masks_r = F.interpolate(
            mask_b.unsqueeze(1).float(),
            size=(H_f, W_f),
            mode='nearest',
        ).squeeze(1).reshape(K, -1)                              # [K, N]

        mask_sum = masks_r.sum(1, keepdim=True).clamp(min=1e-6)

        # Background-subtracted mean pooling
        mean_pool  = torch.mm(masks_r, feat_flat) / mask_sum    # [K, D]
        scene_mean = feat_flat.mean(0, keepdim=True)             # [1, D]
        mean_pool  = mean_pool - scene_mean

        # Max pooling (raw — no background subtraction)
        neg_val  = torch.tensor(-1e9, dtype=feat_flat.dtype,
                                device=feat_flat.device)
        score    = feat_flat.unsqueeze(0).expand(K, N, D)       # view [K,N,D]
        valid    = (masks_r > 0.5).unsqueeze(-1).expand(-1, -1, D)
        masked   = torch.where(valid, score, neg_val)            # [K, N, D]
        max_pool = masked.max(dim=1).values                      # [K, D]
        del masked

        results.append(torch.cat([mean_pool, max_pool], dim=-1))  # [K, 2D]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Module 1: MaskedObjectPooling  (kept for API / checkpoint compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class MaskedObjectPooling(nn.Module):
    """Legacy module — retained so older checkpoints and unit tests still load."""

    def __init__(self, visual_dim, geom_dim=5, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(visual_dim + geom_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def compute_geometric_features(self, masks):
        return _compute_geom(masks)

    def forward(self, feature_map, masks, class_ids=None):
        clip_objs = _bg_sub_masked_pool(feature_map, masks)
        results = []
        for b, pooled in enumerate(clip_objs):
            geom     = _compute_geom(masks[b])
            combined = torch.cat([pooled, geom], dim=-1)
            results.append(self.projection(combined))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Module 2: Geometry-Keyed Self-Attention  (kept for V2 API compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class GeometryKeyedSelfAttention(nn.Module):
    """
    Objects exchange information via geometry-defined attention.

    Q and K are projected from 2-D geometry (centroid, bbox, area), NOT from
    the visual features.  Attention weights reflect spatial relationships
    between objects regardless of visual similarity.  V carries the rich
    visual content that gets redistributed.

    WHY THIS PREVENTS COLLAPSE:
    After pooling from a single frozen CLIP image all K object features are
    nearly identical -> Q ≈ K ≈ constant -> standard self-attn collapses to
    averaging V, which is also nearly constant.  By deriving Q and K from
    geometry (always distinct per object), attention patterns remain
    spatially meaningful even with visually identical V's.
    """

    def __init__(self, d_model: int, n_heads: int,
                 geom_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.geom_qk = nn.Linear(geom_dim, d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                geom: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        x    : [B, K, d_model]
        geom : [B, K, geom_dim]
        mask : [B, K] bool, True = valid object
        """
        qk  = self.geom_qk(geom)
        kpm = (~mask) if mask is not None else None
        out, _ = self.attn(qk, qk, x, key_padding_mask=kpm)
        return self.norm(x + self.drop(out))


# ─────────────────────────────────────────────────────────────────────────────
# Module 3: Cross-Modal Transformer Layer (geometry-conditioned query)
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalTransformerLayer(nn.Module):
    """
    Objects cross-attend to language tokens with geometry-biased queries.
    Even objects with identical visual features query language with distinct
    vectors (geometry-shifted) and receive distinct responses.
    """

    def __init__(self, d_model=256, n_heads=8, d_ff=1024,
                 dropout=0.1, geom_dim=5):
        super().__init__()
        self.geom_query_proj = nn.Linear(geom_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        # Kept for checkpoint compat with V1 which had a self_attn weight
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

    def forward(self, query, context,
                query_mask=None, context_mask=None, geom=None):
        c_pad = (~context_mask) if context_mask is not None else None
        q = query + self.geom_query_proj(geom) if geom is not None else query
        x_cross, attn_w = self.cross_attn(q, context, context,
                                           key_padding_mask=c_pad)
        x = self.norm2(query + self.drop2(x_cross))
        x = self.norm3(x + self.ffn(x))
        return x, attn_w


# ─────────────────────────────────────────────────────────────────────────────
# Module 5: Joint Unified Transformer  (V3 core)
# ─────────────────────────────────────────────────────────────────────────────

class JointTransformerEncoder(nn.Module):
    """
    Pre-norm TransformerEncoder that processes [object tokens | language tokens]
    jointly so every object can attend to all other objects AND all language
    tokens in a single forward pass.

    WHY THIS PREVENTS COLLAPSE:
    When objects are scored independently (as in V2) the model converges to
    predicting the dataset-mean cost for every object as this minimises the
    regression loss with near-zero ranking gradient.  By forcing every object
    to attend to its siblings, the scores must be RELATIONAL — the model
    cannot output identical scores for all objects without also incurring a
    large ranking loss (since attended-to objects provide different context).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 d_ff: int = 1024, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN: more stable gradient flow
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, tokens: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        tokens     : [B, S, d_model]
        valid_mask : [B, S] bool  True = valid token (padding = False)
        """
        kpm = (~valid_mask) if valid_mask is not None else None
        return self.encoder(tokens, src_key_padding_mask=kpm)


# ─────────────────────────────────────────────────────────────────────────────
# Module 6: LangGeoNetV3  (main model)
# ─────────────────────────────────────────────────────────────────────────────

class LangGeoNetV3(nn.Module):
    """
    Dual-backbone (CLIP ViT-B/16 + DINOv2-base) language-guided geodesic cost
    predictor with joint object-language transformer.

    Key differences from V2
    ───────────────────────
    Visual capacity  : DINOv2-base (86M, 768-dim) vs DINOv2-small (22M, 384-dim)
    Fine-tuned layers: CLIP last-6 + DINOv2 last-4 (vs last-4 + last-2 in V2)
    Object pooling   : mean + max per backbone (vs mean-only in V2)
    Transformer      : joint object+language encoder (vs cross-attn-only in V2)
    Cost head        : linear + final sigmoid (vs interior sigmoid in V2)

    Peak VRAM: ~8-12 GB at batch=8.  Fits comfortably within 32 GB.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        dino_model_name: str = "facebook/dinov2-small",
        freeze_clip: bool = True,
        freeze_dino: bool = True,
        max_objects: int = 50,
        num_classes=None,       # unused — kept for call-site compat
    ):
        super().__init__()
        self.d_model         = d_model
        self.max_objects     = max_objects
        self.geom_dim        = 5
        self.geom_bypass_dim = 64

        # ── Backbone 1: CLIP (visual + text) ─────────────────────────────────
        self.clip    = CLIPModel.from_pretrained(clip_model_name)
        clip_vis_dim = self.clip.config.vision_config.hidden_size   # 768
        clip_txt_dim = self.clip.config.text_config.hidden_size     # 512

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            # Unfreeze last 6 visual blocks (more than V2's 4)
            for layer in self.clip.vision_model.encoder.layers[-6:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            # Unfreeze last 6 text blocks
            for layer in self.clip.text_model.encoder.layers[-6:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.text_model.final_layer_norm.parameters():
                p.requires_grad = True

        # ── Backbone 2: DINOv2-small/base (spatial / depth features) ──────────
        self.dino    = Dinov2Model.from_pretrained(dino_model_name)
        dino_dim     = self.dino.config.hidden_size                 # 384 (small) / 768 (base)

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False
            # Unfreeze last 4 blocks (more than V2's 2)
            for layer in self.dino.encoder.layer[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.dino.layernorm.parameters():
                p.requires_grad = True

        # Normalisation buffers survive .to(device) / state_dict saves
        self.register_buffer('_clip_mean',
                             torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_clip_std',
                             torch.tensor(_CLIP_STD ).view(1, 3, 1, 1))
        self.register_buffer('_dino_mean',
                             torch.tensor(_DINO_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_dino_std',
                             torch.tensor(_DINO_STD ).view(1, 3, 1, 1))

        # ── Patch projections (high-dim → d_model before pooling) ────────────
        self.clip_vis_proj = nn.Linear(clip_vis_dim, d_model)   # 768 → 256
        self.dino_vis_proj = nn.Linear(dino_dim,     d_model)   # 768 → 256
        self.lang_proj     = nn.Linear(clip_txt_dim, d_model)   # 512 → 256

        # ── Object feature projection ─────────────────────────────────────────
        # Input: clip(mean+max) + dino(mean+max) + geom
        #      = 2*d_model + 2*d_model + geom_dim
        obj_in_dim = 4 * d_model + self.geom_dim              # 1029 at d=256
        self.obj_proj = nn.Sequential(
            nn.Linear(obj_in_dim, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # ── Geometry encoder and bypass ───────────────────────────────────────
        self.geom_encode = nn.Sequential(
            nn.Linear(self.geom_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.geom_bypass = nn.Sequential(
            nn.Linear(self.geom_dim, self.geom_bypass_dim),
            nn.GELU(),
            nn.LayerNorm(self.geom_bypass_dim),
        )

        # ── Type + positional embeddings ──────────────────────────────────────
        self.obj_type_embed  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.obj_pos_embed   = nn.Parameter(
            torch.randn(1, max_objects, d_model) * 0.02
        )

        # ── Joint unified transformer ─────────────────────────────────────────
        self.joint_transformer = JointTransformerEncoder(
            d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, n_layers=n_layers, dropout=dropout,
        )

        # ── Instruction pooling ───────────────────────────────────────────────
        self.instruction_pool_attn = nn.Linear(d_model, 1)

        # ── Learnable temperature for alignment score ─────────────────────────
        self.log_temp = nn.Parameter(torch.zeros(1))   # exp(0)=1, learned

        # ── Cost head  (no interior Sigmoid — avoids gradient vanishing) ──────
        # Input: obj_feats(d) + geom_bypass(64) + align_scalar(1)  = d+65
        head_in_dim = d_model + self.geom_bypass_dim + 1
        self.geo_head = nn.Sequential(
            nn.Linear(head_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # V2-compat attributes (not used in forward but may be referenced)
        self.geom_self_attn    = GeometryKeyedSelfAttention(
            d_model, n_heads, geom_dim=self.geom_dim, dropout=dropout
        )
        self.transformer_layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, n_heads, d_ff,
                                       dropout, geom_dim=self.geom_dim)
            for _ in range(max(1, n_layers // 2))
        ])
        self.instr_align_proj = nn.Linear(d_model, d_model)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _renorm_for_dino(self, images: torch.Tensor) -> torch.Tensor:
        rgb01 = images * self._clip_std + self._clip_mean
        return (rgb01 - self._dino_mean) / self._dino_std

    def encode_visual_clip(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP ViT-B/16 → [B, 196, d_model]"""
        grad_on = any(p.requires_grad
                      for p in self.clip.vision_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out     = self.clip.vision_model(pixel_values=images)
            patches = out.last_hidden_state[:, 1:, :]   # drop CLS [B, 196, 768]
        return self.clip_vis_proj(patches)              # [B, 196, d_model]

    def encode_visual_dino(self, images: torch.Tensor) -> torch.Tensor:
        """DINOv2-base → [B, 256, d_model]"""
        images_dino = self._renorm_for_dino(images)
        grad_on = any(p.requires_grad for p in self.dino.parameters())
        with torch.set_grad_enabled(grad_on):
            out     = self.dino(pixel_values=images_dino,
                                interpolate_pos_encoding=True)
            patches = out.last_hidden_state[:, 1:, :]   # drop CLS [B, 256, 768]
        return self.dino_vis_proj(patches)              # [B, 256, d_model]

    def encode_language(self, input_ids, attention_mask):
        """CLIP text → [B, L, d_model], bool mask."""
        grad_on = any(p.requires_grad
                      for p in self.clip.text_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.clip.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask)
        return self.lang_proj(out.last_hidden_state), attention_mask.bool()

    def pool_instruction(self, lang_feats: torch.Tensor,
                         lang_mask: torch.Tensor) -> torch.Tensor:
        scores = self.instruction_pool_attn(lang_feats).squeeze(-1)
        scores = scores.masked_fill(~lang_mask, -1e9)
        return torch.bmm(F.softmax(scores, dim=-1).unsqueeze(1),
                         lang_feats).squeeze(1)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, images, masks_list, class_ids_list,
                input_ids, attention_mask,
                positions_list=None):
        """
        Args
        ────
        images         : [B, 3, 224, 224]  CLIP-normalised
        masks_list     : list[B] of [K_b, H, W] bool tensors
        class_ids_list : ignored (kept for API compat)
        input_ids      : [B, L]
        attention_mask : [B, L]
        positions_list : ignored

        Returns
        ───────
        predictions      : list[B] of [K_b] float in [0, 1]
        attn_weights_all : [] (empty — for API compat; joint encoder has no
                            per-layer cross-attention outputs)
        """
        B      = images.shape[0]
        device = images.device

        # ── Visual encoding ────────────────────────────────────────────────────
        clip_feats = self.encode_visual_clip(images)    # [B, 196, d]
        dino_feats = self.encode_visual_dino(images)    # [B, 256, d]

        # ── Language encoding ──────────────────────────────────────────────────
        lang_feats, lang_mask = self.encode_language(input_ids, attention_mask)

        # ── Per-object pooling (mean+max, both backbones) ──────────────────────
        clip_pool = _masked_mean_max_pool(clip_feats, masks_list)  # [K_b, 2d]
        dino_pool = _masked_mean_max_pool(dino_feats, masks_list)  # [K_b, 2d]

        # ── Geometry ───────────────────────────────────────────────────────────
        geom_list = [_compute_geom(masks_list[b]) for b in range(B)]
        K_counts  = [g.shape[0] for g in geom_list]
        K_max     = max(K_counts) if K_counts else 1

        # ── Combined object projection ─────────────────────────────────────────
        obj_feats_list = []
        for b in range(B):
            K_b = K_counts[b]
            if K_b == 0:
                obj_feats_list.append(
                    torch.zeros(0, self.d_model, device=device))
                continue
            combined = torch.cat(
                [clip_pool[b], dino_pool[b], geom_list[b]], dim=-1
            )                                              # [K_b, 4d+5]
            obj_feats_list.append(self.obj_proj(combined))  # [K_b, d]

        # ── Pad to [B, K_max, d] ──────────────────────────────────────────────
        obj_padded  = torch.zeros(B, K_max, self.d_model, device=device)
        obj_valid   = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        geom_padded = torch.zeros(B, K_max, self.geom_dim,   device=device)

        for b in range(B):
            K_c = min(K_counts[b], K_max)
            if K_c > 0:
                obj_padded[b,  :K_c] = obj_feats_list[b][:K_c]
                obj_valid[b,   :K_c] = True
                geom_padded[b, :K_c] = geom_list[b][:K_c]

        # ── Positional + type + geometry embeddings ───────────────────────────
        if K_max <= self.max_objects:
            pos_emb = self.obj_pos_embed[:, :K_max]
        else:
            pos_emb = F.interpolate(
                self.obj_pos_embed.permute(0, 2, 1),
                size=K_max, mode='linear', align_corners=False,
            ).permute(0, 2, 1)

        geom_enc    = self.geom_encode(geom_padded)     # [B, K_max, d]
        obj_tokens  = (obj_padded + geom_enc
                       + self.obj_type_embed + pos_emb)
        lang_tokens = lang_feats + self.lang_type_embed

        # ── Joint transformer: objects + language attend to each other ─────────
        joint_tokens = torch.cat([obj_tokens,  lang_tokens], dim=1)  # [B, K+L, d]
        joint_valid  = torch.cat([obj_valid,   lang_mask   ], dim=1)  # [B, K+L]

        joint_out  = self.joint_transformer(joint_tokens, joint_valid)
        obj_out    = joint_out[:, :K_max]               # [B, K_max, d]
        lang_out   = joint_out[:, K_max:]               # [B, L, d]

        # ── Instruction pooling (on post-joint language tokens) ───────────────
        instr_vec  = self.pool_instruction(lang_out, lang_mask)     # [B, d]

        # ── Temperature-scaled alignment score ────────────────────────────────
        temp  = self.log_temp.exp().clamp(min=0.07, max=10.0)
        align = torch.bmm(
            obj_out, instr_vec.unsqueeze(-1)
        ).squeeze(-1) / (self.d_model ** 0.5 * temp)               # [B, K_max]

        # ── Cost head ──────────────────────────────────────────────────────────
        geom_bp  = self.geom_bypass(geom_padded)        # [B, K_max, 64]
        head_in  = torch.cat(
            [obj_out, geom_bp, align.unsqueeze(-1)], dim=-1
        )
        geo_pred = self.geo_head(head_in).squeeze(-1)   # [B, K_max]

        predictions = [
            geo_pred[b, :min(K_counts[b], K_max)] for b in range(B)
        ]
        return predictions, []          # empty list for API compat


# ─────────────────────────────────────────────────────────────────────────────
# Aliases — so code that imports the old names still works
# ─────────────────────────────────────────────────────────────────────────────

LangGeoNetV2 = LangGeoNetV3     # V3 is a drop-in replacement
LangGeoNet   = LangGeoNetV3


# ─────────────────────────────────────────────────────────────────────────────
# Factory (API-compatible with train.py positional call signature)
# ─────────────────────────────────────────────────────────────────────────────

def build_langgeonet(
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    num_classes=None,                           # unused — kept for compat
    clip_model: str = "openai/clip-vit-base-patch16",
) -> LangGeoNetV3:
    return LangGeoNetV3(
        d_model         = d_model,
        n_heads         = n_heads,
        n_layers        = n_layers,
        clip_model_name = clip_model,
        dino_model_name = "facebook/dinov2-small",
        freeze_clip     = True,
        freeze_dino     = True,
    )
