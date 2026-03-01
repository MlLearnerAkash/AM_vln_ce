import torch
import torch.nn as nn

class GeodesicCostDecoder(nn.Module):
    def __init__(self, clip_dim=512, num_layers=4):
        super().__init__()
        # 1. Project CLIP text embedding to match visual patches
        self.text_projection = nn.Linear(clip_dim, clip_dim)
        
        # 2. Goal Seed Encoder: Encodes (x, y) coordinates of the goal
        self.goal_seed_encoder = nn.Sequential(
            nn.Linear(2, clip_dim),
            nn.LayerNorm(clip_dim)
        )
        
        # 3. Transformer Blocks for Cost Propagation
        # Self-attention allows patches to calculate "distance" from the Seed
        encoder_layer = nn.TransformerEncoderLayer(d_model=clip_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Regressor: Maps latent features to a single cost value per patch
        self.cost_head = nn.Sequential(
            nn.Linear(clip_dim, 1),
            nn.Softplus() # Cost must be positive and continuous
        )

    def forward(self, patch_feats, text_feats, goal_xy):
        # patch_feats: [B, 256, 512] (from CLIP ViT)
        # text_feats: [B, 512] (from CLIP Text)
        # goal_xy: [B, 2] (Normalized coordinates of the goal)
        
        # A. Fuse Instruction: Add projected text to every visual patch
        instr_context = self.text_projection(text_feats).unsqueeze(1)
        x = patch_feats + instr_context 
        
        # B. Inject Goal Seed: Treat goal as a special "Source" token
        seed_token = self.goal_seed_encoder(goal_xy).unsqueeze(1) # [B, 1, 512]
        x = torch.cat([seed_token, x], dim=1) # [B, 257, 512]
        
        # C. Propagate: Transformer calculates relative costs
        latent_map = self.transformer(x)
        
        # D. Predict: Remove seed token and get per-patch cost
        predicted_costs = self.cost_head(latent_map[:, 1:, :]) # [B, 256, 1]
        return predicted_costs.squeeze(-1)
