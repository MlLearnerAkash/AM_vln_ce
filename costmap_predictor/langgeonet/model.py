"""LangGeoNet v4 — Multiplicative Fusion Geodesic Cost Predictor.

ROOT-CAUSE FIX: All previous architectures used ADDITIVE fusion:
   cost = sigmoid(vis_logit + lang_logit + align)
This lets the model learn vis_logit ≈ answer and lang_logit ≈ 0.
Language becomes a tiny additive correction easily ignored.

Fix: MULTIPLICATIVE (Hadamard) fusion.

  vis_tokens  = obj_proj(clip_pool || dino_pool)   [B, K, d]
  lang_tokens = lang_proj(clip_text)                [B, L, d]
  lang_obj    = CrossAttn(Q=vis, K/V=lang)          [B, K, d]  — NO visual residual
  fused       = vis_tokens * lang_obj               [B, K, d]  — Hadamard product
  cost        = sigmoid(cost_head(fused) + align_gate * CLIP_align)

  Why language CANNOT be silent:
  - fused = vis * lang: if lang_obj → 0, fused → 0, cost → 0.5 (random)
  - The model MUST produce informative lang_obj to predict costs
  - Any change in instruction → different cross-attn output → different fused → different cost
  - ElementWise multiply binds modalities: neither can be removed

  Three gradient paths to text encoder:
    (1) Main loss → cost_head → fused = vis * lang_obj → cross_attn K/V → text encoder
    (2) Main loss → align_gate * align → text_projection → text encoder
    (3) Aux loss on lang_obj alone → cross_attn → text encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, Dinov2Model


_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_DINO_MEAN = [0.485,      0.456,      0.406      ]
_DINO_STD  = [0.229,      0.224,      0.225      ]


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------

def _masked_mean_max_pool(feature_map: torch.Tensor, masks_list: list) -> list:
    """Background-subtracted mean + raw max pooling per object."""
    B = feature_map.shape[0]
    results = []
    for b in range(B):
        mask_b = masks_list[b]
        K = mask_b.shape[0]
        feat = feature_map[b]
        N, D = feat.shape
        if K == 0:
            results.append(torch.zeros(0, 2 * D, device=feat.device))
            continue
        H_f = W_f = int(N ** 0.5)
        feat_flat = feat.reshape(-1, D)
        masks_r = F.interpolate(
            mask_b.unsqueeze(1).float(), size=(H_f, W_f), mode='nearest',
        ).squeeze(1).reshape(K, -1)
        mask_sum  = masks_r.sum(1, keepdim=True).clamp(min=1e-6)
        mean_pool = torch.mm(masks_r, feat_flat) / mask_sum
        mean_pool = mean_pool - feat_flat.mean(0, keepdim=True)
        neg_val  = torch.tensor(-1e9, dtype=feat_flat.dtype, device=feat_flat.device)
        valid    = (masks_r > 0.5).unsqueeze(-1).expand(-1, -1, D)
        masked   = torch.where(valid, feat_flat.unsqueeze(0).expand(K, N, D), neg_val)
        max_pool = masked.max(dim=1).values
        del masked
        results.append(torch.cat([mean_pool, max_pool], dim=-1))
    return results


def _masked_simple_mean_pool(feature_map: torch.Tensor, masks_list: list) -> list:
    """Simple mean pool per object for CLIP alignment."""
    B = feature_map.shape[0]
    results = []
    for b in range(B):
        mask_b = masks_list[b]
        K = mask_b.shape[0]
        feat = feature_map[b]
        N, D = feat.shape
        if K == 0:
            results.append(torch.zeros(0, D, device=feat.device))
            continue
        H_f = W_f = int(N ** 0.5)
        feat_flat = feat.reshape(-1, D)
        masks_r = F.interpolate(
            mask_b.unsqueeze(1).float(), size=(H_f, W_f), mode='nearest',
        ).squeeze(1).reshape(K, -1)
        mask_sum = masks_r.sum(1, keepdim=True).clamp(min=1)
        results.append(torch.mm(masks_r, feat_flat) / mask_sum)
    return results


# ---------------------------------------------------------------------------
# Cross-Attention Layer — NO visual residual
# ---------------------------------------------------------------------------

class PureCrossAttnLayer(nn.Module):
    """Cross-attention: Q=visual, K/V=language.  NO visual residual.
    Output is a pure weighted sum of language VALUE tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, obj, lang, obj_mask=None, lang_mask=None):
        q  = self.norm_q(obj)
        kv = self.norm_kv(lang)
        kpm = (~lang_mask) if lang_mask is not None else None
        ca_out, _ = self.cross_attn(query=q, key=kv, value=kv, key_padding_mask=kpm)
        ca_out = torch.nan_to_num(ca_out, nan=0.0)
        # NO residual from obj — output is purely language-derived
        out = ca_out + self.ffn(self.norm_ff(ca_out))
        return out


# ---------------------------------------------------------------------------
# Main model — MULTIPLICATIVE FUSION
# ---------------------------------------------------------------------------

class LangGeoNet(nn.Module):
    """Multiplicative Fusion Geodesic Cost Predictor.

    cost = sigmoid(cost_head(vis_tokens * lang_obj) + align_gate * CLIP_align)

    The Hadamard product vis * lang means:
      - If lang_obj → 0, fused → 0, all info lost → model cannot predict
      - If vis_tokens → 0, fused → 0 → model cannot predict
      - BOTH modalities must carry meaningful signal
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        dino_model_name: str = "facebook/dinov2-small",
        freeze_clip: bool = True,
        freeze_dino: bool = True,
        max_objects: int = 50,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_objects  = max_objects

        # ── Backbones ──────────────────────────────────────────────────────
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.dino = Dinov2Model.from_pretrained(dino_model_name)

        clip_vis_dim = self.clip.config.vision_config.hidden_size   # 768
        clip_txt_dim = self.clip.config.text_config.hidden_size     # 512
        dino_dim     = self.dino.config.hidden_size                 # 384

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            for layer in self.clip.vision_model.encoder.layers[-6:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = True
            for layer in self.clip.text_model.encoder.layers[-6:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.text_model.final_layer_norm.parameters():
                p.requires_grad = True
            for p in self.clip.text_projection.parameters():
                p.requires_grad = True

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False
            for layer in self.dino.encoder.layer[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.dino.layernorm.parameters():
                p.requires_grad = True

        # ── Normalisation buffers ──────────────────────────────────────────
        self.register_buffer('_clip_mean', torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_clip_std',  torch.tensor(_CLIP_STD ).view(1, 3, 1, 1))
        self.register_buffer('_dino_mean', torch.tensor(_DINO_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_dino_std',  torch.tensor(_DINO_STD ).view(1, 3, 1, 1))

        # ── Projections → d_model ──────────────────────────────────────────
        self.clip_vis_proj = nn.Linear(clip_vis_dim, d_model)
        self.dino_vis_proj = nn.Linear(dino_dim,     d_model)
        self.lang_proj     = nn.Linear(clip_txt_dim, d_model)

        # Object projection: clip(mean+max=2d) + dino(mean+max=2d) = 4d → d
        self.obj_proj = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model), nn.GELU(), nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),      nn.GELU(), nn.LayerNorm(d_model),
        )

        # ── Positional / type embeddings ───────────────────────────────────
        self.obj_type_embed  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.obj_pos_embed   = nn.Parameter(torch.randn(1, max_objects, d_model) * 0.02)

        # ── Pure cross-attention (NO visual residual) ──────────────────────
        self.cross_layers = nn.ModuleList([
            PureCrossAttnLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.lang_out_norm = nn.LayerNorm(d_model)

        # ── Fusion normalization ───────────────────────────────────────────
        # After Hadamard product, normalize to stabilize magnitudes
        self.fused_norm = nn.LayerNorm(d_model)

        # ── Cost head (operates on multiplicatively fused features) ────────
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.LayerNorm(d_model // 2), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # ── Language-only auxiliary head ───────────────────────────────────
        # Directly on cross-attn output → forces language pathway to learn costs
        self.lang_aux_cost_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # ── CLIP alignment gate (floored ≥ 0.1) ──────────────────────────
        self._align_gate_raw = nn.Parameter(torch.tensor(0.5))

        # ── Sentence-level auxiliary (EOS → scalar) ───────────────────────
        self.lang_aux_head = nn.Linear(d_model, 1)

    @property
    def align_gate(self):
        return F.softplus(self._align_gate_raw) + 0.1

    # ── Encoding helpers ───────────────────────────────────────────────────

    def _renorm_for_dino(self, images):
        rgb01 = images * self._clip_std + self._clip_mean
        return (rgb01 - self._dino_mean) / self._dino_std

    def _clip_forward_visual(self, images):
        grad_on = any(p.requires_grad for p in self.clip.vision_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.clip.vision_model(pixel_values=images)
            raw = out.last_hidden_state[:, 1:]
        return raw, self.clip_vis_proj(raw)

    def _encode_visual_dino(self, images):
        grad_on = any(p.requires_grad for p in self.dino.parameters())
        with torch.set_grad_enabled(grad_on):
            raw = self.dino(
                pixel_values=self._renorm_for_dino(images),
                interpolate_pos_encoding=True,
            ).last_hidden_state[:, 1:]
        return self.dino_vis_proj(raw)

    def _encode_language_full(self, input_ids, attention_mask):
        grad_on = any(p.requires_grad for p in self.clip.text_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        tokens = self.lang_proj(out.last_hidden_state)
        grad_proj = any(p.requires_grad for p in self.clip.text_projection.parameters())
        with torch.set_grad_enabled(grad_on or grad_proj):
            txt_embed = F.normalize(
                self.clip.text_projection(out.pooler_output), dim=-1
            )
        return tokens, attention_mask.bool(), txt_embed

    def _clip_object_align(self, clip_raw, masks_list, txt_embed):
        clip_obj_raw = _masked_simple_mean_pool(clip_raw, masks_list)
        grad_vp = any(p.requires_grad for p in self.clip.visual_projection.parameters())
        scores = []
        for b, obj_raw in enumerate(clip_obj_raw):
            K = obj_raw.shape[0]
            if K == 0:
                scores.append(torch.zeros(0, device=txt_embed.device))
                continue
            with torch.set_grad_enabled(grad_vp or obj_raw.requires_grad):
                obj_embed = F.normalize(self.clip.visual_projection(obj_raw), dim=-1)
            scores.append((obj_embed * txt_embed[b].unsqueeze(0)).sum(-1))
        return scores

    def _pool_eos(self, lang_feats, lang_mask):
        lengths = lang_mask.long().sum(dim=1) - 1
        lengths = lengths.clamp(min=0, max=lang_feats.shape[1] - 1)
        B = lang_feats.shape[0]
        return lang_feats[torch.arange(B, device=lang_feats.device), lengths]

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, images, masks_list, input_ids, attention_mask):
        B      = images.shape[0]
        device = images.device

        # Encode (each backbone runs once)
        clip_raw, clip_proj = self._clip_forward_visual(images)
        dino_proj           = self._encode_visual_dino(images)
        lang_feats, lang_mask, txt_embed = self._encode_language_full(input_ids, attention_mask)

        # Per-object visual pooling
        clip_pool = _masked_mean_max_pool(clip_proj, masks_list)
        dino_pool = _masked_mean_max_pool(dino_proj, masks_list)

        K_counts = [masks_list[b].shape[0] for b in range(B)]
        K_max    = max(K_counts) if K_counts else 1

        obj_feats_list = []
        for b in range(B):
            if K_counts[b] == 0:
                obj_feats_list.append(torch.zeros(0, self.d_model, device=device))
            else:
                combined = torch.cat([clip_pool[b], dino_pool[b]], dim=-1)
                obj_feats_list.append(self.obj_proj(combined))

        obj_padded = torch.zeros(B, K_max, self.d_model, device=device)
        obj_valid  = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        for b in range(B):
            K_c = min(K_counts[b], K_max)
            if K_c > 0:
                obj_padded[b, :K_c] = obj_feats_list[b][:K_c]
                obj_valid[b,  :K_c] = True

        if K_max <= self.max_objects:
            pos_emb = self.obj_pos_embed[:, :K_max]
        else:
            pos_emb = F.interpolate(
                self.obj_pos_embed.permute(0, 2, 1),
                size=K_max, mode='linear', align_corners=False,
            ).permute(0, 2, 1)

        vis_tokens  = obj_padded + self.obj_type_embed + pos_emb   # [B, K, d]
        lang_tokens = lang_feats + self.lang_type_embed             # [B, L, d]

        # ── Cross-attention: Q=vis, K/V=lang (no visual residual) ──────────
        lang_obj = vis_tokens  # initial queries
        for layer in self.cross_layers:
            lang_obj = layer(lang_obj, lang_tokens, obj_valid, lang_mask)
        lang_obj = self.lang_out_norm(lang_obj)                     # [B, K, d]

        # ── MULTIPLICATIVE FUSION ──────────────────────────────────────────
        # vis_tokens: what the object looks like + where it is
        # lang_obj:   what the instruction says about this object (purely from language)
        # fused = vis * lang:  both must be non-zero for signal to flow
        fused = self.fused_norm(vis_tokens * lang_obj)              # [B, K, d]

        # ── Cost prediction ────────────────────────────────────────────────
        cost_logit = self.cost_head(fused).squeeze(-1)              # [B, K]

        # ── CLIP alignment (additive fine-tuning) ──────────────────────────
        align_scores_list = self._clip_object_align(clip_raw, masks_list, txt_embed)
        align_padded = torch.zeros(B, K_max, device=device)
        for b in range(B):
            K_c = min(K_counts[b], K_max)
            if K_c > 0:
                align_padded[b, :K_c] = align_scores_list[b][:K_c]

        gate = self.align_gate
        cost_pred = torch.sigmoid(cost_logit + gate * align_padded)

        # ── Auxiliary: language-only cost prediction (from lang_obj alone) ──
        lang_aux_logit = self.lang_aux_cost_head(lang_obj).squeeze(-1)  # [B, K]

        # ── Auxiliary: sentence-level prediction ───────────────────────────
        instr_vec       = self._pool_eos(lang_feats, lang_mask)         # [B, d]
        lang_sent_pred  = torch.sigmoid(self.lang_aux_head(instr_vec)).squeeze(-1)

        return (
            [cost_pred[b, :min(K_counts[b], K_max)] for b in range(B)],
            {
                "lang_aux":      lang_sent_pred,                          # [B]
                "align_scores":  [align_scores_list[b][:min(K_counts[b], K_max)]
                                  for b in range(B)],
                "lang_aux_logit": [lang_aux_logit[b, :min(K_counts[b], K_max)]
                                   for b in range(B)],
                # For instrumented probing:
                "vis_tokens":    vis_tokens,       # [B, K, d]
                "lang_obj":      lang_obj,         # [B, K, d]
                "fused":         fused,             # [B, K, d]
                "cost_logit":    cost_logit,        # [B, K]
                "obj_valid":     obj_valid,         # [B, K]
            },
        )

    # ── Modality contribution probe ────────────────────────────────────────

    @torch.no_grad()
    def modality_contributions(self, images, masks_list, input_ids, attention_mask):
        was_training = self.training
        self.eval()
        try:
            preds_real, _ = self.forward(images, masks_list, input_ids, attention_mask)
            B, L   = input_ids.shape
            device = input_ids.device
            blank_ids   = torch.zeros(B, L, dtype=torch.long, device=device)
            blank_ids[:, 0] = 49406; blank_ids[:, 1] = 49407
            blank_amask = torch.zeros(B, L, dtype=torch.long, device=device)
            blank_amask[:, :2] = 1
            preds_blank, _ = self.forward(images, masks_list, blank_ids, blank_amask)
            lang_eff = sum((r - b).abs().sum().item() for r, b in zip(preds_real, preds_blank))
            vis_base = sum(b.abs().sum().item() for b in preds_blank)
            total = lang_eff + vis_base + 1e-9
            return 100.0 * vis_base / total, 100.0 * lang_eff / total
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_langgeonet(
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    clip_model: str = "openai/clip-vit-base-patch16",
    dino_model: str = "facebook/dinov2-small",
) -> LangGeoNet:
    return LangGeoNet(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        clip_model_name=clip_model, dino_model_name=dino_model,
        freeze_clip=True, freeze_dino=True,
    )


# ---------------------------------------------------------------------------
# Bilinear Cost Head — used by VLMLangGeoNet
# ---------------------------------------------------------------------------

class BilinearCostHead(nn.Module):
    """
    cost_k = sigmoid( post_MLP( W_v(obj_k) ⊙ W_l(lang) ) )

    Neither pathway can be zeroed without killing predictions because the
    output is a Hadamard product: if either factor → 0, fused → 0, cost → 0.5.

    W_l is initialised near-zero (std=0.01) so the model must explicitly learn
    to use the language pathway — it cannot start with a visual shortcut.
    """

    def __init__(self, obj_dim: int, lang_dim: int, d_proj: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.obj_norm  = nn.LayerNorm(obj_dim)
        self.lang_norm = nn.LayerNorm(lang_dim)

        self.W_v = nn.Linear(obj_dim,  d_proj, bias=False)
        self.W_l = nn.Linear(lang_dim, d_proj, bias=False)

        self.fused_norm = nn.LayerNorm(d_proj)
        self.post = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, d_proj // 2),
            nn.GELU(),
            nn.LayerNorm(d_proj // 2),
            nn.Linear(d_proj // 2, 1),
        )

        # W_l near-zero init: forces the model to learn to USE language
        nn.init.normal_(self.W_l.weight, std=0.01)
        nn.init.normal_(self.W_v.weight, std=0.02)

    def forward(self, obj_feats: torch.Tensor, lang_feat: torch.Tensor):
        """
        obj_feats : [K, obj_dim]
        lang_feat : [lang_dim]
        Returns   : (preds [K], logits [K])
        """
        v = self.W_v(self.obj_norm(obj_feats))           # [K, d_proj]
        l = self.W_l(self.lang_norm(lang_feat))           # [d_proj]
        fused  = v * l.unsqueeze(0)                       # [K, d_proj]
        logits = self.post(self.fused_norm(fused)).squeeze(-1)  # [K]
        return torch.sigmoid(logits), logits


# ---------------------------------------------------------------------------
# VLM-backed geodesic cost predictor
# ---------------------------------------------------------------------------

class VLMLangGeoNet(nn.Module):
    """Qwen2-VL backbone + BilinearCostHead geodesic cost predictor.

    Root-cause fixes vs. v1–v4:
      [1] Instruction BEFORE image in prompt → causal attention lets image
          tokens attend to the instruction.  Per-object features are
          conditioned on the instruction rather than being purely visual.
      [2] BilinearCostHead: W_v(obj) ⊙ W_l(lang) — neither can be zero.
      [3] Training unfreezes the last n_unfreeze transformer layers of both
          the visual encoder and the causal language decoder so the internal
          VLM representations can be fine-tuned.

    Optimizer groups (3-group like the CLIP model):
      head_params    — BilinearCostHead            (lr_head)
      vlm_vis_params — visual encoder unfrozen      (lr_backbone)
      vlm_txt_params — LM decoder unfrozen + norms  (lr_text, wd=0)
    """

    def __init__(self, vlm_path: str, d_proj: int = 256,
                 n_unfreeze: int = 4, dropout: float = 0.1):
        super().__init__()
        import math as _math
        self._math = _math

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=None,
        )
        self.processor = AutoProcessor.from_pretrained(vlm_path)
        self.processor.tokenizer.padding_side = "left"

        self.hidden_dim  = self.vlm.config.hidden_size       # 3584
        self._img_tok    = self.vlm.config.image_token_id    # 151655
        self._vis_tok    = getattr(self.vlm.config, 'vision_token_id', 151654)
        self._vis_start  = self.vlm.config.vision_start_token_id
        self._vis_end    = self.vlm.config.vision_end_token_id
        self._n_unfreeze = n_unfreeze

        # ── Freeze all VLM params first ───────────────────────────────────
        for p in self.vlm.parameters():
            p.requires_grad = False

        # ── Unfreeze last n_unfreeze LM decoder layers ────────────────────
        lm_layers = self.vlm.model.language_model.layers
        for layer in lm_layers[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True

        # ── Always unfreeze LM output norm (RMSNorm, ~3K params) ─────────
        for p in self.vlm.model.language_model.norm.parameters():
            p.requires_grad = True

        # ── Unfreeze last n_unfreeze visual encoder blocks ────────────────
        vis_blocks = self.vlm.visual.blocks
        for block in vis_blocks[-n_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

        # ── Always unfreeze visual merger (projects patches → LM space) ───
        for p in self.vlm.visual.merger.parameters():
            p.requires_grad = True

        # ── Cost head  (float32 for numerical stability) ──────────────────
        self.cost_head = BilinearCostHead(
            obj_dim=self.hidden_dim,
            lang_dim=self.hidden_dim,
            d_proj=d_proj,
            dropout=dropout,
        ).float()

        total_p    = sum(p.numel() for p in self.parameters())
        trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen     = total_p - trainable
        import logging as _log
        _log.getLogger(__name__).info(
            f"VLMLangGeoNet: {total_p:,} total | "
            f"{trainable:,} trainable ({100*trainable/total_p:.1f}%) | "
            f"{frozen:,} frozen"
        )

    # ── Prompt builder ────────────────────────────────────────────────────

    def _build_input(self, rgb_np, instruction: str):
        """Build VLM inputs. INSTRUCTION BEFORE IMAGE is critical for causal attn."""
        from PIL import Image as _PIL
        from qwen_vl_utils import process_vision_info

        pil = _PIL.fromarray(rgb_np.astype("uint8"))
        max_side = max(pil.size)
        if max_side > 448:
            s = 448 / max_side
            pil = pil.resize(
                (int(pil.width * s), int(pil.height * s)), _PIL.BILINEAR)

        messages = [{"role": "user", "content": [
            {"type": "text",
             "text": f"Navigation instruction: {instruction}\n"},
            {"type": "image", "image": pil},
            {"type": "text",
             "text": "\nFor each object predict its geodesic cost relevance."},
        ]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=img_inputs, padding=True, return_tensors="pt")
        return inputs, pil.size  # (W, H)

    # ── Feature extraction ────────────────────────────────────────────────

    def _extract_features(self, inputs, img_size, masks_np, device):
        """Returns (obj_feats [K,D], lang_feat [D]) in float32."""
        ids = inputs["input_ids"][0]

        # Run VLM — with gradients when some params are trainable
        if self.training and self._n_unfreeze > 0:
            out = self.vlm(**inputs, output_hidden_states=True)
        else:
            with torch.no_grad():
                out = self.vlm(**inputs, output_hidden_states=True)

        last = out.hidden_states[-1][0].float()  # [N, D]  fp32

        # Image token positions (merged visual patches)
        img_mask  = (ids == self._img_tok)
        img_feats = last[img_mask]                # [N_img, D]  — sees instruction ✓

        # Text token positions
        skip = {0, 151643, 151644, 151645,
                self._img_tok, self._vis_tok, self._vis_start, self._vis_end}
        text_mask = torch.tensor(
            [tok.item() not in skip for tok in ids],
            dtype=torch.bool, device=ids.device)
        text_feats = last[text_mask]

        lang_feat = (text_feats.mean(0) if text_feats.shape[0] > 0
                     else torch.zeros(self.hidden_dim, device=device))

        obj_feats = self._pool_objects(img_feats, img_size, masks_np, device)
        return obj_feats, lang_feat

    def _pool_objects(self, img_feats, img_size, masks_np, device):
        """Bilinearly resize masks to the image-token grid and weighted-mean pool."""
        import math as _math
        N_img = img_feats.shape[0]
        if N_img == 0 or masks_np.shape[0] == 0:
            return torch.zeros(masks_np.shape[0], self.hidden_dim, device=device)

        W_img, H_img = img_size
        grid_h = max(1, _math.ceil(H_img / 14 / 2))
        grid_w = max(1, _math.ceil(W_img / 14 / 2))
        if grid_h * grid_w > N_img:
            grid_h = max(1, int(_math.sqrt(N_img * H_img / max(W_img, 1))))
            grid_w = N_img // max(grid_h, 1)

        used      = grid_h * grid_w
        feat_grid = img_feats[:used].view(grid_h, grid_w, -1)
        feat_grid = feat_grid.permute(2, 0, 1).unsqueeze(0)  # [1, D, gh, gw]

        obj_feats_list = []
        for k in range(masks_np.shape[0]):
            m_t  = torch.from_numpy(masks_np[k].astype("float32")).to(device)
            m_rs = F.interpolate(m_t.unsqueeze(0).unsqueeze(0),
                                 size=(grid_h, grid_w),
                                 mode='bilinear', align_corners=False).squeeze()
            wsum   = m_rs.sum().clamp(min=1e-6)
            pooled = (feat_grid[0] * m_rs.unsqueeze(0)).sum(dim=(1, 2)) / wsum
            obj_feats_list.append(pooled)
        return torch.stack(obj_feats_list)  # [K, D]  fp32

    # ── Single-sample forward ─────────────────────────────────────────────

    def forward_single(self, rgb_np, instruction: str, masks_np, device):
        """Returns (preds [K], obj_feats [K,D], lang_feat [D])."""
        if masks_np.shape[0] == 0:
            empty = torch.zeros(0, device=device)
            return empty, empty, empty
        inputs, img_size = self._build_input(rgb_np, instruction)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        obj_feats, lang_feat = self._extract_features(inputs, img_size, masks_np, device)
        preds, _ = self.cost_head(obj_feats.to(device), lang_feat.to(device))
        return preds, obj_feats, lang_feat

    # ── Batch forward ─────────────────────────────────────────────────────

    def forward(self, frame_rgbs, masks_list, instructions):
        """
        frame_rgbs   : list of numpy [H,W,3] arrays (or uint8 tensors)
        masks_list   : list of numpy [K,H,W] bool arrays
        instructions : list of str

        Returns: (preds_list, aux)
          preds_list — list of [K] float32 sigmoid tensors
          aux        — {"lang_feats": list[D tensor]}
        """
        device = next(self.cost_head.parameters()).device
        preds_list  = []
        lang_feats  = []

        for rgb, masks_np, instr in zip(frame_rgbs, masks_list, instructions):
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            if masks_np.shape[0] == 0:
                preds_list.append(torch.zeros(0, device=device))
                lang_feats.append(torch.zeros(self.hidden_dim, device=device))
                continue
            preds, _, lang_feat = self.forward_single(rgb, instr, masks_np, device)
            preds_list.append(preds)
            lang_feats.append(lang_feat)

        return preds_list, {"lang_feats": lang_feats}

    # ── Modality contribution probe ───────────────────────────────────────

    @torch.no_grad()
    def modality_contributions(self, frame_rgbs, masks_list, instructions):
        """Compare real-instruction preds vs blank-instruction preds."""
        was_training = self.training
        self.eval()
        try:
            preds_real, _  = self.forward(frame_rgbs, masks_list, instructions)
            blank_instrs   = [""] * len(instructions)
            preds_blank, _ = self.forward(frame_rgbs, masks_list, blank_instrs)
            lang_eff  = sum((r - b).abs().sum().item()
                            for r, b in zip(preds_real, preds_blank))
            vis_base  = sum(b.abs().sum().item() for b in preds_blank)
            total = lang_eff + vis_base + 1e-9
            return 100.0 * vis_base / total, 100.0 * lang_eff / total
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# VLM factory
# ---------------------------------------------------------------------------

def build_vlm_langgeonet(
    vlm_path: str,
    d_proj: int = 256,
    n_unfreeze: int = 4,
) -> VLMLangGeoNet:
    return VLMLangGeoNet(vlm_path=vlm_path, d_proj=d_proj, n_unfreeze=n_unfreeze)
