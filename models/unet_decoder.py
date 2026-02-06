# import torch
# import torch.nn as nn
# import math

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.deconv = nn.ConvTranspose2d(
#             in_channels, out_channels, kernel_size=4, stride=2, padding=1
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.deconv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class QwenVLHeatmapDecoder(nn.Module):
#     """
#     Takes Qwen2.5VL hidden state (batch_size, variable_tokens, 3584) and outputs a heatmap of image size (batch_size, 1, 640, 480).
#     GAN-style decoder: uses ConvTranspose2d blocks for upsampling.
#     """
#     def __init__(self, hidden_dim=3584, out_size=(640, 480)):
#         super().__init__()
#         self.out_size = out_size
#         self.proj = nn.Linear(hidden_dim, 256)
#         self.token_agg = nn.AdaptiveAvgPool1d(1)

#         self.fc = nn.Sequential(
#             nn.Linear(256, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 16 * 8 * 6),
#             nn.ReLU(inplace=True)
#         )
#         self.up1 = DecoderBlock(16, 32)   # 8x6 -> 16x12
#         self.up2 = DecoderBlock(32, 64)   # 16x12 -> 32x24
#         self.up3 = DecoderBlock(64, 128)  # 32x24 -> 64x48
#         self.up4 = DecoderBlock(128, 256) # 64x48 -> 128x96
#         self.up5 = DecoderBlock(256, 128) # 128x96 -> 256x192
#         self.up6 = DecoderBlock(128, 64)  # 256x192 -> 512x384
#         self.out_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
#         self.final_upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=True)

#     def forward(self, x):
#         b, n, c = x.shape
#         x = self.proj(x)  # (B, n, 256)
#         x = x.transpose(1, 2)  # (B, 256, n)
#         x = self.token_agg(x)  # (B, 256, 1)
#         x = x.squeeze(-1)      # (B, 256)

#         x = self.fc(x)         # (B, 16*8*6)
#         x = x.view(b, 16, 8, 6)  # (B, 16, 8, 6)

#         x = self.up1(x)   # (B, 32, 16, 12)
#         x = self.up2(x)   # (B, 64, 32, 24)
#         x = self.up3(x)   # (B, 128, 64, 48)
#         x = self.up4(x)   # (B, 256, 128, 96)
#         x = self.up5(x)   # (B, 128, 256, 192)
#         x = self.up6(x)   # (B, 64, 512, 384)
#         x = self.out_conv(x)  # (B, 1, 512, 384)
#         x = self.final_upsample(x)  # (B, 1, 640, 480)
#         return x

# if __name__== "__main__":
#     batch_size = 2
#     hidden_dim = 3584
#     num_tokens = 157
#     out_size = (640, 480)

#     model = QwenVLHeatmapDecoder(hidden_dim=hidden_dim, out_size=out_size)
#     input_tensor = torch.randn(batch_size, num_tokens, hidden_dim)
#     output = model(input_tensor)

#     print(f"Input shape: {input_tensor.shape}")
#     print(f"Output shape: {output.shape}")  # Should be (batch_size, 1, 640, 480)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) * 2
    The standard building block of a U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetUpBlock(nn.Module):
    """
    Upscaling then DoubleConv.
    Using 'bilinear' upsampling avoids the checkerboard artifacts 
    common with ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class QwenVLHeatmapDecoder(nn.Module):
    def __init__(self, hidden_dim=3584, out_size=(640, 480)):
        super().__init__()
        self.out_size = out_size
        
        # 1. Projection Head (Token Aggregation)
        # Reduce Qwen's massive 3584 dim to 512 for the bottleneck
        self.proj = nn.Linear(hidden_dim, 512)
        self.token_agg = nn.AdaptiveAvgPool1d(1)

        # 2. Bottleneck Setup
        # We map the aggregated token to a 8x6 grid with 512 channels.
        # This preserves semantic density before we start upsampling.
        self.bottleneck_h, self.bottleneck_w = 8, 6
        self.bottleneck_channels = 512
        
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.bottleneck_channels * self.bottleneck_h * self.bottleneck_w),
            nn.ReLU(inplace=True)
        )

        # 3. U-Net Expansion Path (Decoder)
        # Channels decrease as resolution increases: 512 -> 256 -> 128 ...
        self.up1 = UNetUpBlock(512, 256)  # 8x6   -> 16x12
        self.up2 = UNetUpBlock(256, 128)  # 16x12 -> 32x24
        self.up3 = UNetUpBlock(128, 64)   # 32x24 -> 64x48
        self.up4 = UNetUpBlock(64, 32)    # 64x48 -> 128x96
        self.up5 = UNetUpBlock(32, 16)    # 128x96 -> 256x192
        self.up6 = UNetUpBlock(16, 16)    # 256x192 -> 512x384

        # 4. Final Output Head
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        nn.init.constant_(self.final_conv.bias, -2)
        self.final_upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # x shape: (Batch, Tokens, HiddenDim) 
        b, n, c = x.shape

        # --- Step 1: Compress Tokens ---
        x = self.proj(x)       # (B, n, 512)
        x = x.transpose(1, 2)  # (B, 512, n)
        x = self.token_agg(x)  # (B, 512, 1)
        x = x.squeeze(-1)      # (B, 512)

        # --- Step 2: Reshape to Spatial Grid ---
        x = self.fc(x)         # (B, 512*8*6)
        x = x.view(b, self.bottleneck_channels, self.bottleneck_h, self.bottleneck_w)

        # --- Step 3: U-Net Expansion ---
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)

        # --- Step 4: Final Prediction ---
        x = self.final_conv(x)     # (B, 1, 512, 384)
        x = self.final_upsample(x) # (B, 1, 640, 480)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    model = QwenVLHeatmapDecoder()
    # Simulate Qwen2.5-VL output
    dummy_input = torch.randn(2, 157, 3584)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}") # (2, 1, 640, 480)