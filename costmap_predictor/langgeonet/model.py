"""
LangGeoNetV2: Dual-backbone language-guided geodesic cost predictor.

Architecture improvements over V1 that prevent mean-field collapse:

  1. Dual visual backbone:
       • CLIP ViT-B/16  — semantic per-object features (last 4 layers trainable)
       • DINOv2-small   — spatial/depth per-object features (last 2 blocks trainable)
     DINOv2 captures 3-D scene structure (depth, layout) that single-image masked
     pooling from CLIP misses, giving objects genuinely distinct representations.

  2. Background-subtracted masked pooling on BOTH backbones.
     Subtracting the scene mean removes the shared "image DC component" that
     made all K objects collapse after pooling from a single frozen CLIP image.

  3. Geometry-keyed self-attention (new module):
     Q and K are derived from 2-D geometry, not visual features.
     Objects exchange information via spatially-defined attention weights,
     preventing gradient-cancellation collapse when Q=K=near-identical features.

  4. Geometry bypass directly into the cost head (retained from V1).
     2-D geometry (centroid, area, bbox) is always distinct per object and
     bypasses the transformer entirely, anchoring each object's prediction.

  5. Bradley-Terry ranking loss (in losses.py) replaces the margin/hinge loss.
     Produces non-zero gradients everywhere — no dead-zone around a margin.

Input API (unchanged from V1, compatible with train.py):
    forward(images, masks_list, class_ids_list, input_ids, attention_mask)

Output API (unchanged):
    (predictions: list[B] of [K_b] tensors,
     attn_weights_all: list of cross-attn weight tensors)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, Dinov2Model


# ── Normalisation constants ────────────────────────────────────────────────────
# Convert CLIP-preprocessed images → DINOv2-preprocessed images inline.
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_DINO_MEAN = [0.485,      0.456,      0.406      ]
_DINO_STD  = [0.229,      0.224,      0.225      ]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: background-subtracted masked average pooling
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
# Module 2: Geometry-Keyed Self-Attention
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
# Module 4: LangGeoNetV2
# ─────────────────────────────────────────────────────────────────────────────

class LangGeoNetV2(nn.Module):
    """
    Dual-backbone (CLIP ViT-B/16 + DINOv2-small) language-guided geodesic
    cost predictor.

    Peak VRAM: ~10-15 GB at batch=32 on a single GPU.  Fits within 32 GB.

    Trainable parameter groups:
        CLIP visual : last 4 transformer blocks + post_layernorm
        CLIP text   : last 4 transformer blocks + final_layer_norm
        DINOv2      : last 2 encoder blocks + layernorm
        All new heads (obj_proj, geom_self_attn, transformer_layers, geo_head)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
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
        self.clip        = CLIPModel.from_pretrained(clip_model_name)
        clip_vis_dim     = self.clip.config.vision_config.hidden_size   # 768
        clip_txt_dim     = self.clip.config.text_config.hidden_size     # 512

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            for layer in self.clip.vision_model.encoder.layers[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            for layer in self.clip.text_model.encoder.layers[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.text_model.final_layer_norm.parameters():
                p.requires_grad = True

        # ── Backbone 2: DINOv2-small (spatial / depth features) ──────────────
        self.dino    = Dinov2Model.from_pretrained(dino_model_name)
        dino_dim     = self.dino.config.hidden_size                     # 384

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False
            for layer in self.dino.encoder.layer[-2:]:
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

        # ── Projections ───────────────────────────────────────────────────────
        self.clip_vis_proj = nn.Linear(clip_vis_dim, d_model)
        self.dino_vis_proj = nn.Linear(dino_dim,      d_model)
        self.lang_proj     = nn.Linear(clip_txt_dim,  d_model)

        # Combined object projection: clip_obj(d) + dino_obj(d) + geom(5) -> d
        self.obj_proj = nn.Sequential(
            nn.Linear(2 * d_model + self.geom_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # ── Geometry-keyed object self-attention ──────────────────────────────
        self.geom_self_attn = GeometryKeyedSelfAttention(
            d_model, n_heads, geom_dim=self.geom_dim, dropout=dropout
        )

        # ── Type + positional embeddings ──────────────────────────────────────
        self.obj_type_embed  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.obj_pos_embed   = nn.Parameter(
            torch.randn(1, max_objects, d_model) * 0.02
        )

        # ── Cross-modal transformer ───────────────────────────────────────────
        self.transformer_layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, n_heads, d_ff,
                                       dropout, geom_dim=self.geom_dim)
            for _ in range(n_layers)
        ])

        # ── Instruction pooling ───────────────────────────────────────────────
        self.instruction_pool_attn = nn.Linear(d_model, 1)
        self.instr_align_proj      = nn.Linear(d_model, d_model)

        # ── Geometry bypass + cost head ───────────────────────────────────────
        self.geom_bypass = nn.Sequential(
            nn.Linear(self.geom_dim, self.geom_bypass_dim),
            nn.GELU(),
            nn.LayerNorm(self.geom_bypass_dim),
        )
        # head input: d_model + 1 (align scalar) + geom_bypass_dim = 321
        self.geo_head = nn.Sequential(
            nn.Linear(d_model + 1 + self.geom_bypass_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _renorm_for_dino(self, images: torch.Tensor) -> torch.Tensor:
        """Convert CLIP-normalised images to DINOv2-normalised images."""
        rgb01 = images * self._clip_std + self._clip_mean   # -> [0, 1]
        return (rgb01 - self._dino_mean) / self._dino_std

    def encode_visual_clip(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP ViT-B/16 -> [B, 196, d_model] patch features."""
        grad_on = any(p.requires_grad
                      for p in self.clip.vision_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.clip.vision_model(pixel_values=images)
            patches = out.last_hidden_state[:, 1:, :]   # drop CLS  [B, 196, 768]
        return self.clip_vis_proj(patches)

    def encode_visual_dino(self, images: torch.Tensor) -> torch.Tensor:
        """DINOv2-small -> [B, 256, d_model] patch features (224x224 input)."""
        images_dino = self._renorm_for_dino(images)
        grad_on = any(p.requires_grad for p in self.dino.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.dino(pixel_values=images_dino,
                            interpolate_pos_encoding=True)
            patches = out.last_hidden_state[:, 1:, :]   # drop CLS  [B, 256, 384]
        return self.dino_vis_proj(patches)

    def encode_language(self, input_ids, attention_mask):
        """CLIP text -> [B, L, d_model] + bool mask."""
        grad_on = any(p.requires_grad
                      for p in self.clip.text_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.clip.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask)
        return self.lang_proj(out.last_hidden_state), attention_mask.bool()

    def pool_instruction(self, lang_feats, lang_mask):
        """Attention-weighted pooling -> [B, d_model]."""
        scores = self.instruction_pool_attn(lang_feats).squeeze(-1)
        scores = scores.masked_fill(~lang_mask, -1e9)
        return torch.bmm(F.softmax(scores, -1).unsqueeze(1),
                         lang_feats).squeeze(1)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, images, masks_list, class_ids_list,
                input_ids, attention_mask,
                positions_list=None):
        """
        Args:
            images         : [B, 3, 224, 224]  CLIP-preprocessed
            masks_list     : list[B] of [K_b, H, W] bool tensors
            class_ids_list : ignored
            input_ids      : [B, L]
            attention_mask : [B, L]
            positions_list : ignored (no agent position used)

        Returns:
            predictions      : list[B] of [K_b] float tensors in [0, 1]
            attn_weights_all : list of [B, K_max, L] cross-attn weight tensors
        """
        B      = images.shape[0]
        device = images.device

        # ── Visual encoding ────────────────────────────────────────────────────
        clip_feats = self.encode_visual_clip(images)    # [B, 196, d]
        dino_feats = self.encode_visual_dino(images)    # [B, 256, d]

        # ── Language encoding ──────────────────────────────────────────────────
        lang_feats, lang_mask = self.encode_language(input_ids, attention_mask)
        instr_vec = self.pool_instruction(lang_feats, lang_mask)  # [B, d]

        # ── Per-object pooling (background-subtracted, both backbones) ─────────
        clip_obj_list = _bg_sub_masked_pool(clip_feats, masks_list)
        dino_obj_list = _bg_sub_masked_pool(dino_feats, masks_list)

        # ── Geometry + combined object projection ──────────────────────────────
        geom_list = [_compute_geom(masks_list[b]) for b in range(B)]
        K_counts  = [geom_list[b].shape[0]         for b in range(B)]
        K_max     = max(K_counts) if K_counts else 1

        obj_feats_list = []
        for b in range(B):
            K_b = K_counts[b]
            if K_b == 0:
                obj_feats_list.append(
                    torch.zeros(0, self.d_model, device=device))
                continue
            combined = torch.cat(
                [clip_obj_list[b], dino_obj_list[b], geom_list[b]], dim=-1
            )                                           # [K_b, 2d+5]
            obj_feats_list.append(self.obj_proj(combined))    # [K_b, d]

        # ── Pad to [B, K_max, d] ──────────────────────────────────────────────
        obj_padded  = torch.zeros(B, K_max, self.d_model, device=device)
        obj_mask    = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        geom_padded = torch.zeros(B, K_max, self.geom_dim,   device=device)

        for b in range(B):
            K_c = min(K_counts[b], K_max)
            if K_c > 0:
                obj_padded[b,  :K_c] = obj_feats_list[b][:K_c]
                obj_mask[b,    :K_c] = True
                geom_padded[b, :K_c] = geom_list[b][:K_c]

        # ── Type + positional embeddings ───────────────────────────────────────
        if K_max <= self.max_objects:
            pos_emb = self.obj_pos_embed[:, :K_max]
        else:
            pos_emb = F.interpolate(
                self.obj_pos_embed.permute(0, 2, 1),
                size=K_max, mode='linear', align_corners=False,
            ).permute(0, 2, 1)

        x           = obj_padded + self.obj_type_embed + pos_emb
        lang_tokens = lang_feats + self.lang_type_embed

        # ── Geometry-keyed self-attention (objects compare with each other) ────
        x = self.geom_self_attn(x, geom_padded, mask=obj_mask)

        # ── Cross-modal transformer (objects <-> language) ─────────────────────
        attn_weights_all = []
        for layer in self.transformer_layers:
            x, attn_w = layer(x, lang_tokens, obj_mask, lang_mask,
                              geom=geom_padded)
            attn_weights_all.append(attn_w)

        # ── Cost head ──────────────────────────────────────────────────────────
        instr_proj = self.instr_align_proj(instr_vec)
        align = torch.bmm(
            x, instr_proj.unsqueeze(-1)
        ).squeeze(-1) / (self.d_model ** 0.5)           # [B, K_max]

        geom_bp  = self.geom_bypass(geom_padded)         # [B, K_max, 64]
        head_in  = torch.cat([x, align.unsqueeze(-1), geom_bp], dim=-1)
        geo_pred = self.geo_head(head_in).squeeze(-1)    # [B, K_max]

        predictions = [
            geo_pred[b, :min(K_counts[b], K_max)] for b in range(B)
        ]
        return predictions, attn_weights_all


# Alias so any code that imports the old class name still works
LangGeoNet = LangGeoNetV2


# ─────────────────────────────────────────────────────────────────────────────
# Factory (API-compatible with train.py positional call signature)
# ─────────────────────────────────────────────────────────────────────────────

def build_langgeonet(
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 2,
    num_classes=None,                           # unused — kept for compat
    clip_model: str = "openai/clip-vit-base-patch16",
) -> LangGeoNetV2:
    return LangGeoNetV2(
        d_model         = d_model,
        n_heads         = n_heads,
        n_layers        = n_layers,
        clip_model_name = clip_model,
        dino_model_name = "facebook/dinov2-small",
        freeze_clip     = True,
        freeze_dino     = True,
    )
