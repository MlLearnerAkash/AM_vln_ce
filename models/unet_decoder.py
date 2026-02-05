import torch
import torch.nn as nn
import math

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class QwenVLHeatmapDecoder(nn.Module):
    """
    Takes Qwen2.5VL hidden state (batch_size, variable_tokens, 3584) and outputs a heatmap of image size (batch_size, 1, 640, 480).
    No grid assumption; uses 1D tokens, aggregates and decodes to heatmap.
    """
    def __init__(self, hidden_dim=3584, out_size=(640, 480)):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Linear(hidden_dim, 256)
        self.token_agg = nn.AdaptiveAvgPool1d(1)  # Aggregate variable tokens to a single feature vector

        # Decoder path: 256 -> 128 -> 64 -> 32 -> 16
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 16 * 14 * 14),
            nn.ReLU(inplace=True)
        )
        self.up1 = DecoderBlock(16, 32)
        self.up2 = DecoderBlock(32, 64)
        self.up3 = DecoderBlock(64, 128)
        self.up4 = DecoderBlock(128, 256)
        self.out_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.final_upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # x: (batch_size, variable_tokens, 3584)
        b, n, c = x.shape
        x = self.proj(x)  # (B, n, 256)
        x = x.transpose(1, 2)  # (B, 256, n)
        x = self.token_agg(x)  # (B, 256, 1)
        x = x.squeeze(-1)      # (B, 256)

        # Fully connected to initial spatial feature map
        x = self.fc(x)         # (B, 16*14*14)
        x = x.view(b, 16, 14, 14)  # (B, 16, 14, 14)

        x = self.up1(x)   # (B, 32, 28, 28)
        x = self.up2(x)   # (B, 64, 56, 56)
        x = self.up3(x)   # (B, 128, 112, 112)
        x = self.up4(x)   # (B, 256, 224, 224)
        x = self.out_conv(x)  # (B, 1, 224, 224)
        x = self.final_upsample(x)  # (B, 1, 640, 480)
        return x

if __name__== "__main__":
    # Test the QwenVLHeatmapDecoder with a random tensor
    batch_size = 2
    hidden_dim = 3584
    num_tokens = 157
    out_size = (640, 480)

    model = QwenVLHeatmapDecoder(hidden_dim=hidden_dim, out_size=out_size)
    input_tensor = torch.randn(batch_size, num_tokens, hidden_dim)
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Should be (batch_size, 1, 640, 480)