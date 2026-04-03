"""LangGeoNet: Dual-backbone language-guided geodesic cost predictor.

Objects and language tokens are processed jointly through a shared pre-norm
transformer so every object attends to all siblings and all language tokens
in a single pass, producing relational cost scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, Dinov2Model


_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_DINO_MEAN = [0.485,      0.456,      0.406      ]
_DINO_STD  = [0.229,      0.224,      0.225      ]


def _masked_mean_max_pool(feature_map: torch.Tensor, masks_list: list) -> list:
    """Per-object background-subtracted mean + raw max pooling.

    Args:
        feature_map : [B, N_patches, D]
        masks_list  : list[B] of [K_b, H, W] bool tensors

    Returns:
        list[B] of [K_b, 2*D] tensors  (mean_bg_sub || max_raw)
    """
    B = feature_map.shape[0]
    results = []
    for b in range(B):
        mask_b = masks_list[b]      # [K, H, W]
        K = mask_b.shape[0]
        feat = feature_map[b]       # [N, D]
        N, D = feat.shape
        if K == 0:
            results.append(torch.zeros(0, 2 * D, device=feat.device))
            continue
        H_f = W_f = int(N ** 0.5)
        feat_flat = feat.reshape(-1, D)                      # [N, D]
        masks_r = F.interpolate(
            mask_b.unsqueeze(1).float(), size=(H_f, W_f), mode='nearest',
        ).squeeze(1).reshape(K, -1)                          # [K, N]

        mask_sum  = masks_r.sum(1, keepdim=True).clamp(min=1e-6)
        mean_pool = torch.mm(masks_r, feat_flat) / mask_sum  # [K, D]
        mean_pool = mean_pool - feat_flat.mean(0, keepdim=True)

        neg_val  = torch.tensor(-1e9, dtype=feat_flat.dtype, device=feat_flat.device)
        valid    = (masks_r > 0.5).unsqueeze(-1).expand(-1, -1, D)
        masked   = torch.where(valid, feat_flat.unsqueeze(0).expand(K, N, D), neg_val)
        max_pool = masked.max(dim=1).values                  # [K, D]
        del masked

        results.append(torch.cat([mean_pool, max_pool], dim=-1))  # [K, 2D]
    return results


class JointTransformerEncoder(nn.Module):
    """Pre-norm transformer encoder for joint [object | language] sequences."""

    def __init__(self, d_model=256, n_heads=8, d_ff=1024, n_layers=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        tokens     : [B, S, d_model]
        valid_mask : [B, S] bool  (True = valid, False = padding)
        """
        kpm = (~valid_mask) if valid_mask is not None else None
        return self.encoder(tokens, src_key_padding_mask=kpm)


class LangGeoNet(nn.Module):
    """Dual-backbone (CLIP ViT-B/16 + DINOv2) language-guided geodesic cost predictor.

    Per-object features are pooled from both backbones (mean + max), projected to
    d_model, then processed jointly with language tokens through a shared transformer.
    The final cost is predicted from object representations and their alignment with
    the instruction.
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
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_objects = max_objects

        # CLIP (visual + text)
        self.clip    = CLIPModel.from_pretrained(clip_model_name)
        clip_vis_dim = self.clip.config.vision_config.hidden_size
        clip_txt_dim = self.clip.config.text_config.hidden_size

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            for layer in self.clip.vision_model.encoder.layers[-6:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            for layer in self.clip.text_model.encoder.layers[-6:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.text_model.final_layer_norm.parameters():
                p.requires_grad = True

        # DINOv2
        self.dino    = Dinov2Model.from_pretrained(dino_model_name)
        dino_dim     = self.dino.config.hidden_size

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False
            for layer in self.dino.encoder.layer[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.dino.layernorm.parameters():
                p.requires_grad = True

        # Normalisation buffers
        self.register_buffer('_clip_mean', torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_clip_std',  torch.tensor(_CLIP_STD ).view(1, 3, 1, 1))
        self.register_buffer('_dino_mean', torch.tensor(_DINO_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_dino_std',  torch.tensor(_DINO_STD ).view(1, 3, 1, 1))

        # Patch projections: backbone dim -> d_model
        self.clip_vis_proj = nn.Linear(clip_vis_dim, d_model)
        self.dino_vis_proj = nn.Linear(dino_dim,     d_model)
        self.lang_proj     = nn.Linear(clip_txt_dim, d_model)

        # Object projection: clip(mean+max) + dino(mean+max) = 4*d_model -> d_model
        self.obj_proj = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Type and positional embeddings
        self.obj_type_embed  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.obj_pos_embed   = nn.Parameter(torch.randn(1, max_objects, d_model) * 0.02)

        # Joint transformer
        self.joint_transformer = JointTransformerEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, dropout=dropout,
        )

        # Instruction pooling (attention over language tokens)
        self.instruction_pool_attn = nn.Linear(d_model, 1)

        # FiLM: condition object tokens on the instruction BEFORE the joint transformer.
        # Small non-zero init (not zero) so gradients flow from step 1.
        self.film = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.normal_(self.film.weight, std=0.01)
        nn.init.zeros_(self.film.bias)

        # Auxiliary language head: predicts the per-frame mean cost from instr_vec alone.
        # Gradient path: aux_loss -> lang_cost_head -> instr_vec -> text_encoder.
        # No visual features on this path — the text encoder must adapt or aux_loss won't drop.
        self.lang_cost_head = nn.Linear(d_model, 1)

        # Cost head: [obj_feat (d) | instr_vec (d)] -> cost in [0, 1].
        self.cost_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def _renorm_for_dino(self, images: torch.Tensor) -> torch.Tensor:
        rgb01 = images * self._clip_std + self._clip_mean
        return (rgb01 - self._dino_mean) / self._dino_std

    def _encode_visual_clip(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, N_patches, d_model]"""
        grad_on = any(p.requires_grad for p in self.clip.vision_model.parameters())
        with torch.set_grad_enabled(grad_on):
            patches = self.clip.vision_model(pixel_values=images).last_hidden_state[:, 1:]
        return self.clip_vis_proj(patches)

    def _encode_visual_dino(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, N_patches, d_model]"""
        grad_on = any(p.requires_grad for p in self.dino.parameters())
        with torch.set_grad_enabled(grad_on):
            patches = self.dino(
                pixel_values=self._renorm_for_dino(images),
                interpolate_pos_encoding=True,
            ).last_hidden_state[:, 1:]
        return self.dino_vis_proj(patches)

    def _encode_language(self, input_ids, attention_mask):
        """-> [B, L, d_model], bool mask"""
        grad_on = any(p.requires_grad for p in self.clip.text_model.parameters())
        with torch.set_grad_enabled(grad_on):
            out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.lang_proj(out.last_hidden_state), attention_mask.bool()

    def _pool_instruction(self, lang_feats: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        scores = self.instruction_pool_attn(lang_feats).squeeze(-1)
        scores = scores.masked_fill(~lang_mask, -1e9)
        return torch.bmm(F.softmax(scores, dim=-1).unsqueeze(1), lang_feats).squeeze(1)

    def forward(self, images, masks_list, input_ids, attention_mask):
        """
        Args:
            images         : [B, 3, H, W]  CLIP-normalised
            masks_list     : list[B] of [K_b, H, W] bool tensors
            input_ids      : [B, L]
            attention_mask : [B, L]

        Returns:
            predictions : list[B] of [K_b] float tensors in [0, 1]
        """
        B      = images.shape[0]
        device = images.device

        clip_feats = self._encode_visual_clip(images)    # [B, N, d]
        dino_feats = self._encode_visual_dino(images)    # [B, N, d]
        lang_feats, lang_mask = self._encode_language(input_ids, attention_mask)

        clip_pool = _masked_mean_max_pool(clip_feats, masks_list)  # list[B] of [K_b, 2d]
        dino_pool = _masked_mean_max_pool(dino_feats, masks_list)  # list[B] of [K_b, 2d]

        K_counts = [masks_list[b].shape[0] for b in range(B)]
        K_max    = max(K_counts) if K_counts else 1

        obj_feats_list = []
        for b in range(B):
            if K_counts[b] == 0:
                obj_feats_list.append(torch.zeros(0, self.d_model, device=device))
            else:
                combined = torch.cat([clip_pool[b], dino_pool[b]], dim=-1)  # [K_b, 4d]
                obj_feats_list.append(self.obj_proj(combined))              # [K_b, d]

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

        obj_tokens  = obj_padded + self.obj_type_embed + pos_emb
        lang_tokens = lang_feats + self.lang_type_embed

        # FiLM: condition object tokens on the instruction BEFORE the joint transformer.
        # Gradient path: loss -> obj_out -> obj_tokens (pre-joint) -> film -> instr_pre -> text_encoder.
        instr_pre   = self._pool_instruction(lang_feats, lang_mask)  # [B, d]
        film_params = self.film(instr_pre)                            # [B, 2d]
        gamma, beta = film_params.chunk(2, dim=-1)                    # each [B, d]
        obj_tokens  = obj_tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        joint_tokens = torch.cat([obj_tokens,  lang_tokens], dim=1)  # [B, K+L, d]
        joint_valid  = torch.cat([obj_valid,   lang_mask   ], dim=1)  # [B, K+L]
        joint_out    = self.joint_transformer(joint_tokens, joint_valid)

        obj_out  = joint_out[:, :K_max]   # [B, K_max, d]
        lang_out = joint_out[:, K_max:]   # [B, L, d]

        instr_vec = self._pool_instruction(lang_out, lang_mask)  # [B, d]

        # Cost head receives obj_out and the full instruction vector per object position.
        # The d-dimensional instr_vec gives a much richer gradient path than a scalar alignment.
        head_in   = torch.cat(
            [obj_out, instr_vec.unsqueeze(1).expand(-1, K_max, -1)], dim=-1
        )                                                            # [B, K_max, 2d]
        cost_pred = self.cost_head(head_in).squeeze(-1)             # [B, K_max]

        # Auxiliary prediction from language only — used to compute aux loss in train.py.
        lang_aux_pred = torch.sigmoid(self.lang_cost_head(instr_vec)).squeeze(-1)  # [B]

        return (
            [cost_pred[b, :min(K_counts[b], K_max)] for b in range(B)],
            lang_aux_pred,
        )

    @torch.enable_grad()
    def modality_contributions(self, images, masks_list, input_ids, attention_mask):
        """Measure the percentage contribution of visual vs language features to cost predictions.

        Runs the full encoder pipeline once, then creates leaf proxy tensors for the
        visual half (obj_out) and language half (instr_vec broadcast) of the cost-head
        input.  Sums all cost predictions, backprops, and computes the Frobenius norm
        of each proxy's gradient.  The ratio gives the attribution percentage.

        Returns:
            vis_pct  : float  visual contribution  [0, 100]
            lang_pct : float  language contribution [0, 100]
        """
        was_training = self.training
        self.eval()
        try:
            B      = images.shape[0]
            device = images.device

            with torch.no_grad():
                clip_feats = self._encode_visual_clip(images)
                dino_feats = self._encode_visual_dino(images)
                lang_feats, lang_mask = self._encode_language(input_ids, attention_mask)

                clip_pool = _masked_mean_max_pool(clip_feats, masks_list)
                dino_pool = _masked_mean_max_pool(dino_feats, masks_list)

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

                obj_tokens  = obj_padded + self.obj_type_embed + pos_emb
                lang_tokens = lang_feats + self.lang_type_embed

                instr_pre   = self._pool_instruction(lang_feats, lang_mask)
                film_params = self.film(instr_pre)
                gamma, beta = film_params.chunk(2, dim=-1)
                obj_tokens  = obj_tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

                joint_tokens = torch.cat([obj_tokens, lang_tokens], dim=1)
                joint_valid  = torch.cat([obj_valid,  lang_mask   ], dim=1)
                joint_out    = self.joint_transformer(joint_tokens, joint_valid)

                obj_out_np   = joint_out[:, :K_max].detach()
                lang_out_np  = joint_out[:, K_max:].detach()
                instr_vec_np = self._pool_instruction(lang_out_np, lang_mask).detach()

            # Leaf proxies — both shaped [B, K_max, d] so gradient norms are
            # directly comparable (no expand-accumulation bias).
            # If lang_proxy were [B, d] and expanded, backprop would SUM gradients
            # from all K_max slots into it, inflating lang_norm by ~K_max vs vis_norm.
            vis_proxy  = obj_out_np.requires_grad_(True)                              # [B, K_max, d]
            lang_proxy = (
                instr_vec_np.unsqueeze(1).expand(-1, K_max, -1).contiguous()
                .detach().requires_grad_(True)
            )                                                                         # [B, K_max, d]

            head_in   = torch.cat([vis_proxy, lang_proxy], dim=-1)                   # [B, K_max, 2d]
            cost_pred = self.cost_head(head_in).squeeze(-1)                           # [B, K_max]

            # Mask padding and sum — this is the scalar we differentiate.
            mask = obj_valid.float()                                                  # [B, K_max]
            cost_pred.mul(mask).sum().backward()

            vis_norm  = vis_proxy.grad.norm().item()   if vis_proxy.grad  is not None else 0.0
            lang_norm = lang_proxy.grad.norm().item()  if lang_proxy.grad is not None else 0.0
            total     = vis_norm + lang_norm + 1e-9

            return 100.0 * vis_norm / total, 100.0 * lang_norm / total
        finally:
            if was_training:
                self.train()


def build_langgeonet(
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    clip_model: str = "openai/clip-vit-base-patch16",
    dino_model: str = "facebook/dinov2-small",
) -> LangGeoNet:
    return LangGeoNet(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        clip_model_name=clip_model,
        dino_model_name=dino_model,
        freeze_clip=True,
        freeze_dino=True,
    )
