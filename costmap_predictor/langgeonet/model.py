"""
LangGeoNet: Language-Guided Geodesic Distance Prediction Network

Predicts normalized geodesic distance for each object in the current frame,
conditioned on the language instruction and instance segmentation masks.

Architecture:
    1. Visual Encoder (CLIP ViT) -> per-object features via masked pooling
    2. Language Encoder (CLIP Text) -> instruction features
    3. Cross-Modal Transformer -> fused object-language features
    4. Geodesic Distance Head -> per-object normalized geodesic distance [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


# -------------------------------------------------------
# Module 1: Masked Object Pooling
# -------------------------------------------------------

class MaskedObjectPooling(nn.Module):
    """
    Extracts per-object features from a visual feature map using instance masks.
    For each object mask, performs masked average pooling over the spatial feature map,
    then concatenates geometric features and class embeddings.
    """

    def __init__(self, visual_dim, geom_dim=5, class_embed_dim=64, num_classes=150, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(visual_dim + class_embed_dim + geom_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def compute_geometric_features(self, masks):
        """
        Compute geometric features for each object mask.

        Args:
            masks: [K, H, W] binary instance masks

        Returns:
            geom_feats: [K, 5] - (centroid_x, centroid_y, area, bbox_w, bbox_h) all normalized
        """
        K, H, W = masks.shape
        device = masks.device
        geom_feats = torch.zeros(K, 5, device=device)

        for k in range(K):
            mask = masks[k]  # [H, W]
            area = mask.float().sum()
            if area < 1:
                continue

            ys, xs = torch.where(mask > 0)
            cx = xs.float().mean() / W
            cy = ys.float().mean() / H

            x_min, x_max = xs.min().float(), xs.max().float()
            y_min, y_max = ys.min().float(), ys.max().float()
            bbox_w = (x_max - x_min + 1) / W
            bbox_h = (y_max - y_min + 1) / H
            area_norm = area / (H * W)

            geom_feats[k] = torch.tensor([cx, cy, area_norm, bbox_w, bbox_h], device=device)

        return geom_feats

    def forward(self, feature_map, masks, class_ids):
        """
        Args:
            feature_map: [B, N_patches, D] CLIP ViT patch features
            masks:       list of [K_b, H, W] binary masks per batch item
            class_ids:   list of [K_b] class IDs per batch item

        Returns:
            object_features: list of [K_b, out_dim] per batch item
        """
        

        batch_size = len(masks)
        all_obj_feats = []

        for b in range(batch_size):
            mask_b = masks[b]    # [K, H, W]
            class_b = class_ids[b]  # [K]
            K = mask_b.shape[0]

            if K == 0:
                all_obj_feats.append(
                    torch.zeros(0, self.out_dim, device=feature_map.device)
                )
                continue

            # n_emb = self.class_embedding.num_embeddings
            # invalid = (class_b < 0) | (class_b >= n_emb)
            # if invalid.any():
            #     print(f"[WARN] batch {b}: invalid class_ids {class_b[invalid].tolist()} "
            #           f"(num_embeddings={n_emb}), clamping.")
            # class_b = class_b.clamp(0, n_emb - 1)

            feat = feature_map[b]  # [N_patches, D]
            N, D = feat.shape
            H_f = W_f = int(N ** 0.5)
            feat_spatial = feat.reshape(H_f, W_f, D)  # [H_f, W_f, D]

            # Resize masks to feature map resolution
            masks_resized = F.interpolate(
                mask_b.unsqueeze(1).float(),  # [K, 1, H, W]
                size=(H_f, W_f),
                mode='nearest'
            ).squeeze(1)  # [K, H_f, W_f]

            # Masked average pooling: [K, D]
            masks_flat = masks_resized.reshape(K, -1)           # [K, H_f*W_f]
            feat_flat = feat_spatial.reshape(-1, D)              # [H_f*W_f, D]
            mask_sum = masks_flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pooled = torch.mm(masks_flat, feat_flat) / mask_sum  # [K, D]

            # Geometric features
            geom = self.compute_geometric_features(mask_b)  # [K, 5]

            # Class embedding
            cls_emb = self.class_embedding(class_b)  # [K, class_embed_dim]

            # Concatenate and project
            combined = torch.cat([pooled, cls_emb, geom], dim=-1)
            obj_feat = self.projection(combined)  # [K, out_dim]

            all_obj_feats.append(obj_feat)

        return all_obj_feats


# -------------------------------------------------------
# Module 2: Cross-Modal Transformer Layer
# -------------------------------------------------------

class CrossModalTransformerLayer(nn.Module):
    """
    Transformer layer where object tokens self-attend, then cross-attend to language.
    
    Flow:
        objects -> self-attention -> cross-attention(objects, language) -> FFN
    """

    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()

        # Object self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        # Object-Language cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query, context, query_mask=None, context_mask=None):
        """
        Args:
            query:        [B, K, D] object tokens
            context:      [B, L, D] language tokens
            query_mask:   [B, K] True = valid token
            context_mask: [B, L] True = valid token

        Returns:
            out:          [B, K, D] updated object tokens
            attn_weights: [B, K, L] cross-attention weights
        """
        q_pad = ~query_mask if query_mask is not None else None
        c_pad = ~context_mask if context_mask is not None else None

        # Self-attention among objects
        residual = query
        x, _ = self.self_attn(query, query, query, key_padding_mask=q_pad)
        x = self.norm1(residual + self.drop1(x))

        # Cross-attention: objects attend to language
        residual = x
        x_cross, attn_weights = self.cross_attn(
            x, context, context, key_padding_mask=c_pad
        )
        x = self.norm2(residual + self.drop2(x_cross))

        # FFN
        residual = x
        x = self.norm3(residual + self.ffn(x))

        return x, attn_weights


# -------------------------------------------------------
# Module 3: LangGeoNet (Full Model)
# -------------------------------------------------------

class LangGeoNet(nn.Module):
    """
    Language-Guided Geodesic Distance Prediction Network.

    Input:
        - RGB frame [B, 3, 224, 224]
        - Instance masks: list of [K_b, H, W]
        - Class IDs: list of [K_b]
        - Instruction token IDs [B, L]  (CLIP tokenizer output)
        - Instruction attention mask [B, L]

    Output:
        - Predicted normalized geodesic distance [0, 1] per object
          (list of [K_b] tensors)
    """

    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        num_classes=150,
        clip_model_name="openai/clip-vit-base-patch16",
        freeze_clip=True,
        max_objects=50,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_objects = max_objects

        # ========== Visual + Language Encoder (CLIP) ==========
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.clip_visual_dim = self.clip.config.vision_config.hidden_size
        self.clip_text_dim = self.clip.config.text_config.hidden_size

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.visual_proj = nn.Linear(self.clip_visual_dim, d_model)
        self.lang_proj = nn.Linear(self.clip_text_dim, d_model)

        # ========== Object Encoder ==========
        self.object_encoder = MaskedObjectPooling(
            visual_dim=d_model,
            geom_dim=5,
            class_embed_dim=64,
            num_classes=num_classes,
            out_dim=d_model,
        )

        # ========== Type + Position Embeddings ==========
        self.obj_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.obj_pos_embed = nn.Parameter(torch.randn(1, max_objects, d_model) * 0.02)

        # ========== Cross-Modal Transformer ==========
        self.transformer_layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ========== Instruction Pooling ==========
        self.instruction_pool_attn = nn.Linear(d_model, 1)

        # ========== Geodesic Distance Prediction Head ==========
        # Input: object feature (d_model) + global instruction (d_model)
        self.geo_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # normalized geodesic in [0, 1]
        )

    def encode_visual(self, images):
        """Extract CLIP ViT patch features -> [B, N_patches, d_model]"""
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.clip.parameters())
        ):
            vision_out = self.clip.vision_model(pixel_values=images)
            patch_feats = vision_out.last_hidden_state[:, 1:, :]  # drop [CLS]

        return self.visual_proj(patch_feats)

    def encode_language(self, input_ids, attention_mask):
        """Encode instruction with CLIP text encoder -> [B, L, d_model] + mask"""
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.clip.text_model.parameters())
        ):
            text_out = self.clip.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        lang_feats = self.lang_proj(text_out.last_hidden_state)
        return lang_feats, attention_mask.bool()

    def pool_instruction(self, lang_features, lang_mask):
        """Attention-weighted pooling -> [B, d_model] global instruction vector."""
        scores = self.instruction_pool_attn(lang_features).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~lang_mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), lang_features).squeeze(1)

    def forward(self, images, masks_list, class_ids_list, input_ids, attention_mask):
        """
        Full forward pass.

        Returns:
            predictions: list of [K_b] predicted geodesic distances
            attn_weights_all: list of cross-attention weight tensors
        """
        B = images.shape[0]
        device = images.device

        # --- Encode ---
        patch_feats = self.encode_visual(images)               # [B, N_p, d]
        lang_feats, lang_mask = self.encode_language(input_ids, attention_mask)  # [B, L, d]
        instr_vec = self.pool_instruction(lang_feats, lang_mask)  # [B, d]

        # --- Object features via masked pooling ---
        obj_feats_list = self.object_encoder(patch_feats, masks_list, class_ids_list)

        # --- Pad objects to [B, K_max, d] ---
        K_counts = [f.shape[0] for f in obj_feats_list]
        K_max = min(max(K_counts), self.max_objects) if K_counts else 1

        obj_padded = torch.zeros(B, K_max, self.d_model, device=device)
        obj_mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)

        for b, (feats, K_b) in enumerate(zip(obj_feats_list, K_counts)):
            K_b = min(K_b, K_max)
            if K_b > 0:
                obj_padded[b, :K_b] = feats[:K_b]
                obj_mask[b, :K_b] = True

        # --- Add type + position embeddings ---
        obj_tokens = obj_padded + self.obj_type_embed + self.obj_pos_embed[:, :K_max]
        lang_tokens = lang_feats + self.lang_type_embed

        # --- Cross-modal transformer ---
        x = obj_tokens
        attn_weights_all = []

        for layer in self.transformer_layers:
            x, attn_w = layer(x, lang_tokens, obj_mask, lang_mask)
            attn_weights_all.append(attn_w)

        # --- Predict geodesic per object ---
        instr_expanded = instr_vec.unsqueeze(1).expand(-1, K_max, -1)
        head_input = torch.cat([x, instr_expanded], dim=-1)  # [B, K_max, 2d]
        geo_pred = self.geo_head(head_input).squeeze(-1)      # [B, K_max]

        # --- Unpad ---
        predictions = []
        for b, K_b in enumerate(K_counts):
            K_b = min(K_b, K_max)
            predictions.append(geo_pred[b, :K_b])

        return predictions, attn_weights_all


# -------------------------------------------------------
# Factory
# -------------------------------------------------------

def build_langgeonet(
    d_model=256,
    n_heads=8,
    n_layers=6,
    num_classes=151,
    clip_model="openai/clip-vit-base-patch16",
):
    """Build LangGeoNet with default config."""
    return LangGeoNet(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        num_classes=num_classes,
        clip_model_name=clip_model,
    )
