"""
LangTopoSeg model.

All sub-modules in one file for clarity. Import surface:
    from model import LangTopoSeg
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vision backbone — lightweight DINOv2 wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SegmentEncoder(nn.Module):
    """
    Encode each padded instance segment into a D-dimensional token.

    For each segment:
      1. Mask-weighted mean-pool DINO patch tokens  → visual_feat [D_dino]
      2. Project to embed_dim                        → v_i
      3. Encode 2D centroid + area with MLP          → p_i
      4. h_i = LayerNorm(v_i + p_i)

    Args
    ----
    dino_model_id : HuggingFace model id for DINOv2
    embed_dim     : output embedding dimension D
    image_h, image_w : input image size (must match dataset config)
    """

    def __init__(self, dino_model_id: str, embed_dim: int, image_h: int, image_w: int):
        super().__init__()
        # Load DINOv2 (frozen)
        self.backbone = AutoModel.from_pretrained(dino_model_id)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        dino_dim = self.backbone.config.hidden_size   # e.g. 384 for dino-small
        self.patch_size = self.backbone.config.patch_size  # default 14

        self.proj = nn.Sequential(
            nn.Linear(dino_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Positional encoder: (cx_norm, cy_norm, area) → embed_dim
        self.pos_enc = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    def _extract_patch_tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb : [B, 3, H, W]
        Returns patch tokens : [B, n_patches, D_dino]
        """
        out = self.backbone(pixel_values=rgb)
        # DINOv2 returns last_hidden_state [B, 1+n_patches, D]
        return out.last_hidden_state[:, 1:, :]   # drop CLS

    def forward(
        self,
        rgb: torch.Tensor,      # [B, 3, H, W]
        masks: torch.Tensor,    # [B, K, H, W]
        centroids: torch.Tensor, # [B, K, 2]
        areas: torch.Tensor,    # [B, K]
    ) -> torch.Tensor:
        """
        Returns node_feats : [B, K, embed_dim]
        """
        B, K, H, W = masks.shape
        patch_tokens = self._extract_patch_tokens(rgb)  # [B, n_patches, D_dino]

        # Reshape mask to patch grid
        ph = H // self.patch_size
        pw = W // self.patch_size
        n_patches = ph * pw

        # Resize masks to patch grid for pooling weights
        mask_patches = F.adaptive_avg_pool2d(masks.float(), (ph, pw))  # [B, K, ph, pw]
        mask_patches = mask_patches.reshape(B, K, ph * pw)              # [B, K, n_patches]

        # Normalise weights (avoid zero-division for empty masks)
        weight_sum = mask_patches.sum(-1, keepdim=True).clamp(min=1e-6)
        weights    = mask_patches / weight_sum                           # [B, K, n_patches]

        # Mask-weighted mean pooling
        # patch_tokens [B, n_patches, D], weights [B, K, n_patches]
        visual_feats = torch.bmm(weights, patch_tokens)                  # [B, K, D_dino]

        # Project
        v_i = self.proj(visual_feats)                                    # [B, K, embed_dim]

        # Positional encoding: (cx, cy, area)  [B, K, 3]
        pos_input = torch.cat([centroids, areas.unsqueeze(-1)], dim=-1)  # [B, K, 3]
        p_i = self.pos_enc(pos_input)                                    # [B, K, embed_dim]

        node_feats = self.norm(v_i + p_i)
        return node_feats   # [B, K, embed_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Language encoder — CLIP text (frozen)
# ─────────────────────────────────────────────────────────────────────────────

class LanguageEncoder(nn.Module):
    """Wrap CLIP text encoder; project to embed_dim."""

    def __init__(self, clip_model_id: str, embed_dim: int):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)
        self.model     = CLIPTextModel.from_pretrained(clip_model_id)
        for p in self.model.parameters():
            p.requires_grad_(False)

        clip_dim = self.model.config.hidden_size
        self.proj = nn.Linear(clip_dim, embed_dim)

    def forward(self, instructions: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        seq_tokens : [B, T, embed_dim]   per-word tokens
        cls_token  : [B, embed_dim]      sentence-level summary
        """
        enc = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            out = self.model(**enc)

        seq_tokens = self.proj(out.last_hidden_state)   # [B, T, embed_dim]
        cls_token  = self.proj(out.pooler_output)       # [B, embed_dim]
        return seq_tokens, cls_token


# ─────────────────────────────────────────────────────────────────────────────
# 3. Language ↔ Segment cross-attention (Level 1 — WHAT)
# ─────────────────────────────────────────────────────────────────────────────

class LangSegCrossAttention(nn.Module):
    """
    Each segment node attends over language tokens.
    Output: language-enriched node features + per-node semantic relevance score.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn      = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm      = nn.LayerNorm(embed_dim)
        self.score_head = nn.Linear(embed_dim, 1)   # semantic relevance

    def forward(
        self,
        node_feats: torch.Tensor,    # [B, K, D]
        lang_tokens: torch.Tensor,   # [B, T, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_L       : [B, K, D]   language-enriched features
        score_sem : [B, K]      semantic relevance in [0,1]
        """
        attended, _ = self.attn(query=node_feats, key=lang_tokens, value=lang_tokens)
        h_L = self.norm(node_feats + attended)
        score_sem = torch.sigmoid(self.score_head(h_L).squeeze(-1))   # [B, K]
        return h_L, score_sem


# ─────────────────────────────────────────────────────────────────────────────
# 4. Direction head — maps language to 2-D image-space direction (Level 1b)
# ─────────────────────────────────────────────────────────────────────────────

class DirectionHead(nn.Module):
    """
    Predicts a 2D image-plane direction vector from the instruction.

    dir_2d ∈ R² (image plane):
      negative x → left,   positive x → right
      negative y → top,    positive y → bottom

    "forward" is handled separately via segment area.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2),
            nn.Tanh(),   # output in (-1, 1)
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        cls_token : [B, embed_dim]
        Returns   : [B, 2]  unit direction vector in image plane
        """
        raw = self.mlp(cls_token)                          # [B, 2]
        norm = raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return raw / norm                                  # unit vector


def compute_dir_gate(
    centroids:  torch.Tensor,   # [B, K, 2]  (cx_norm, cy_norm)
    areas:      torch.Tensor,   # [B, K]
    dir_2d:     torch.Tensor,   # [B, 2]
    scale:      float = 5.0,
) -> torch.Tensor:
    """
    Computes directional gate per node.

    For directional instructions: gate_i = σ(scale · cos_sim(centroid_i, dir_2d))
    For "forward" (dir_2d ≈ zero): gate_i = σ(scale · (area_i - median_area))
    Returns dir_gate : [B, K] in [0, 1]
    """
    # Centroid compatibility with direction
    cent_norm = F.normalize(centroids, dim=-1)             # [B, K, 2]
    dir_norm  = F.normalize(dir_2d, dim=-1).unsqueeze(1)  # [B, 1, 2]
    compat    = (cent_norm * dir_norm).sum(-1)             # [B, K]
    dir_gate  = torch.sigmoid(scale * compat)              # [B, K]

    # Forward proxy: segments with above-median area
    med_area  = areas.median(dim=-1, keepdim=True).values  # [B, 1]
    fwd_gate  = torch.sigmoid(scale * (areas - med_area))  # [B, K]

    # Blend: when dir_2d norm is small, rely on fwd_gate
    dir_strength = dir_2d.norm(dim=-1, keepdim=True).clamp(0, 1)  # [B, 1]
    gate = dir_strength * dir_gate + (1 - dir_strength) * fwd_gate  # [B, K]
    return gate


# ─────────────────────────────────────────────────────────────────────────────
# 5. Intra-frame attention GNN (simplified GATv2-style)
# ─────────────────────────────────────────────────────────────────────────────

class IntraFrameAttention(nn.Module):
    """
    Fully-connected within-frame attention between K segment nodes.
    Implements a simplified GATv2:
      α_ij = softmax(LeakyReLU(W_a [h_i || h_j]))
      h_i' = ELU(Σ_j α_ij W_msg h_j)

    Masks out ghost nodes (mask_i ≈ 0).
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        self.head_dim = embed_dim // n_heads

        self.W_q  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o  = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,         # [B, K, D]
        node_mask: torch.Tensor, # [B, K]  in [0,1] — 0 = ghost
    ) -> torch.Tensor:
        B, K, D = h.shape
        H = self.n_heads

        Q = self.W_q(h).reshape(B, K, H, self.head_dim).transpose(1, 2)  # [B, H, K, d]
        K_ = self.W_k(h).reshape(B, K, H, self.head_dim).transpose(1, 2)
        V  = self.W_v(h).reshape(B, K, H, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K_.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, K, K]

        # Mask ghost nodes: set their key scores to -inf so they're ignored
        # node_mask [B, K] → key mask [B, 1, 1, K]
        ghost_mask = (node_mask < 0.1).unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(ghost_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Guard against all-ghost rows where softmax(-inf, ...) = NaN
        attn = attn.nan_to_num(0.0)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)              # [B, H, K, d]
        out = out.transpose(1, 2).reshape(B, K, D)
        out = self.W_o(out)
        return self.norm(h + out)                # residual + norm: [B, K, D]


class IntraFrameGNN(nn.Module):
    """Stack of IntraFrameAttention layers."""

    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            IntraFrameAttention(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, h: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, node_mask)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# 6. Inter-frame temporal GRU aggregation
# ─────────────────────────────────────────────────────────────────────────────

class InterFrameGRU(nn.Module):
    """
    Aggregate temporal context across n past frames for each node position k.

    At each frame t:
      context_k = mean of h[:,k,:] from past frames (matched by position)
      h_t_k     = GRU(h_t_k, context_k)

    In practice (with padding), we match nodes by their position in the
    padded tensor.  For real tracking, segment_id matching should be used.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(embed_dim, embed_dim)

    def forward(
        self,
        h_window: torch.Tensor,  # [B, T, K, D]  — T frames, current frame last
    ) -> torch.Tensor:
        """Returns h_out : [B, K, D] for the last (current) frame."""
        B, T, K, D = h_window.shape

        # Temporal context = mean of all past frames per node position
        context = h_window[:, :-1, :, :].mean(dim=1)    # [B, K, D]

        # GRU update current frame nodes
        h_current = h_window[:, -1, :, :]               # [B, K, D]
        h_flat    = h_current.reshape(B * K, D)
        ctx_flat  = context.reshape(B * K, D)
        h_out_flat = self.gru(h_flat, ctx_flat)
        return h_out_flat.reshape(B, K, D)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Edge prediction head
# ─────────────────────────────────────────────────────────────────────────────

class EdgeRegressionHead(nn.Module):
    """
    Predict a scalar edge weight for every (i, j) pair.

    Input per pair: [h_i || h_j || h_i ⊙ h_j || Δcent_ij]
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        in_dim = 3 * embed_dim + 2   # h_i || h_j || (h_i ⊙ h_j) || Δcent_ij
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,          # [B, K, D]
        centroids: torch.Tensor,  # [B, K, 2]
        node_mask: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        """Returns W : [B, K, K]  edge weight matrix (symmetric)."""
        B, K, D = h.shape

        # Build pairwise features
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)   # [B, K, K, D]
        h_j = h.unsqueeze(1).expand(-1, K, -1, -1)   # [B, K, K, D]
        had = h_i * h_j
        dc  = (centroids.unsqueeze(2) - centroids.unsqueeze(1))   # [B, K, K, 2]

        pair_feat = torch.cat([h_i, h_j, had, dc], dim=-1)       # [B, K, K, 4D+2]
        W = self.mlp(pair_feat).squeeze(-1)                        # [B, K, K]

        # Symmetrise
        W = (W + W.transpose(-1, -2)) / 2.0

        # Zero out edges involving ghost nodes
        ghost = (node_mask < 0.1)                      # [B, K]
        W = W.masked_fill(ghost.unsqueeze(2), 0.0)
        W = W.masked_fill(ghost.unsqueeze(1), 0.0)
        # Zero self-loops out-of-place to avoid mutating an autograd tensor
        eye = torch.eye(K, device=W.device, dtype=torch.bool).unsqueeze(0)
        W = W.masked_fill(eye, 0.0)

        return W      # [B, K, K]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full LangTopoSeg model
# ─────────────────────────────────────────────────────────────────────────────

class LangTopoSeg(nn.Module):
    """
    End-to-end Language-guided Topological Segmentation map.

    Input (per batch)
    -----------------
    rgb          : [B, T, 3, H, W]   RGB frames (temporal window, T = n_frames+1)
    masks        : [B, T, K, H, W]   instance masks (padded)
    centroids    : [B, T, K, 2]      2D image-space centroids
    areas        : [B, T, K]         normalised mask areas
    k_valid      : [B, T]            actual number of valid instances per frame
    instructions : List[str]  length B

    Output (per batch)
    ------------------
    pred_e3d     : [B, K]      per-instance predicted e3d score (current frame)
    pred_edges   : [B, K, K]   pairwise edge weights (current frame)
    node_mask    : [B, K]      language-guided node selection probabilities
    dir_2d       : [B, 2]      predicted image-plane direction vector
    """

    def __init__(self, cfg):
        super().__init__()
        D = cfg.embed_dim

        self.cfg = cfg
        self.seg_enc   = SegmentEncoder(cfg.vision_model, D, cfg.image_h, cfg.image_w)
        self.lang_enc  = LanguageEncoder(cfg.text_model, D)
        self.cross_attn = LangSegCrossAttention(D, cfg.n_attn_heads)
        self.dir_head  = DirectionHead(D)
        self.intra_gnn = IntraFrameGNN(D, cfg.gat_heads, cfg.n_gat_layers)
        self.inter_gru = InterFrameGRU(D)
        self.edge_head = EdgeRegressionHead(D)

        # Per-instance e3d regression (from final node features)
        self.e3d_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(inplace=True),
            nn.Linear(D // 2, 1),
            nn.Sigmoid(),
        )

    def encode_frame(
        self,
        rgb:       torch.Tensor,   # [B, 3, H, W]
        masks:     torch.Tensor,   # [B, K, H, W]
        centroids: torch.Tensor,   # [B, K, 2]
        areas:     torch.Tensor,   # [B, K]
        lang_tokens: torch.Tensor, # [B, T, D]
        cls_token:   torch.Tensor, # [B, D]
        k_valid:   torch.Tensor,   # [B]
        dir_2d:    torch.Tensor,   # [B, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single frame given pre-computed language features.
        Returns (h_L, score_sem, node_mask) all [B, K, D] / [B, K]
        """
        B, K = masks.shape[:2]

        # Segment encoding
        h = self.seg_enc(rgb, masks, centroids, areas)          # [B, K, D]

        # Language cross-attention → semantic score
        h_L, score_sem = self.cross_attn(h, lang_tokens)        # [B, K, D], [B, K]

        # Directional gate
        dir_gate = compute_dir_gate(centroids, areas, dir_2d, self.cfg.dir_scale)  # [B, K]

        # Combined score → node mask
        score_i  = score_sem * dir_gate                         # [B, K]
        node_mask = torch.sigmoid((score_i - self.cfg.tau) / self.cfg.temp)

        # Zero out padded nodes beyond k_valid (out-of-place to preserve autograd graph)
        idx = torch.arange(K, device=node_mask.device).unsqueeze(0)  # [1, K]
        valid_mask = idx < k_valid.unsqueeze(1)                       # [B, K]
        node_mask = node_mask * valid_mask.float()

        # Intra-frame GNN
        h1 = self.intra_gnn(h_L, node_mask)                    # [B, K, D]
        return h1, score_sem, node_mask

    def forward(
        self,
        rgb:          torch.Tensor,   # [B, T, 3, H, W]
        masks:        torch.Tensor,   # [B, T, K, H, W]
        centroids:    torch.Tensor,   # [B, T, K, 2]
        areas:        torch.Tensor,   # [B, T, K]
        k_valid:      torch.Tensor,   # [B, T]
        instructions: list,           # list of B strings
    ):
        B, T, K = masks.shape[:3]
        device = rgb.device

        # ── Language encoding (shared across frames) ─────────────────
        lang_tokens, cls_token = self.lang_enc(instructions)    # [B, T_lang, D], [B, D]
        dir_2d = self.dir_head(cls_token)                        # [B, 2]

        # ── Encode each frame in the temporal window ─────────────────
        h_per_frame = []
        nm_per_frame = []

        for t in range(T):
            h_t, _, nm_t = self.encode_frame(
                rgb       = rgb[:, t],
                masks     = masks[:, t],
                centroids = centroids[:, t],
                areas     = areas[:, t],
                lang_tokens = lang_tokens,
                cls_token   = cls_token,
                k_valid   = k_valid[:, t],
                dir_2d    = dir_2d,
            )
            h_per_frame.append(h_t)
            nm_per_frame.append(nm_t)

        h_window = torch.stack(h_per_frame,  dim=1)  # [B, T, K, D]
        nm_window = torch.stack(nm_per_frame, dim=1)  # [B, T, K]

        # ── Inter-frame GRU (temporal aggregation) ────────────────────
        h2 = self.inter_gru(h_window)    # [B, K, D] — updated current-frame nodes

        # Final node mask = last frame's mask after GRU update
        node_mask_final = nm_window[:, -1, :]   # [B, K]

        # ── Per-instance e3d prediction ───────────────────────────────
        pred_e3d = self.e3d_head(h2).squeeze(-1)   # [B, K]

        # ── Pairwise edge prediction ──────────────────────────────────
        pred_edges = self.edge_head(h2, centroids[:, -1, :, :], node_mask_final)  # [B, K, K]

        return {
            "pred_e3d":   pred_e3d,        # [B, K]
            "pred_edges": pred_edges,      # [B, K, K]
            "node_mask":  node_mask_final, # [B, K]
            "dir_2d":     dir_2d,          # [B, 2]
        }
