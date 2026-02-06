# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import clip


# class DoubleConv3D(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv3d = nn.Sequential(
#             nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv3d(x)


# class Down3D(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(2),
#             DoubleConv3D(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up3D(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#             self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv3D(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffZ = x2.size()[2] - x1.size()[2]
#         diffY = x2.size()[3] - x1.size()[3]
#         diffX = x2.size()[4] - x1.size()[4]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2,
#                         diffZ // 2, diffZ - diffZ // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class UNet3D(nn.Module):
#     def __init__(self, inChannel, outChannel, fChannel=32, bilinear=True):
#         super(UNet3D, self).__init__()
#         self.inChannel = inChannel
#         self.outChannel = outChannel
#         self.fChannel=fChannel
#         self.bilinear = bilinear

#         self.inc = DoubleConv3D(inChannel, fChannel)
#         self.down1 = Down3D(fChannel, fChannel*2)
#         self.down2 = Down3D(fChannel*2, fChannel*4)
#         self.down3 = Down3D(fChannel*4, fChannel*8)
#         factor = 2 if bilinear else 1
#         self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

#         self.up1 = Up3D(fChannel*16, fChannel*8 // factor, bilinear)
#         self.up2 = Up3D(fChannel*8, fChannel*4 // factor, bilinear)
#         self.up3 = Up3D(fChannel*4, fChannel*2 // factor, bilinear)
#         self.up4 = Up3D(fChannel*2, fChannel, bilinear)
#         self.outc = OutConv3D(fChannel, outChannel)

#     def forward(self, x,args=None):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
#     def forward_x1(self,x):
#         x1 = self.inc(x)
#         return x1

# class UNet3D_PSA(nn.Module):
#     def __init__(self, inChannel, outChannel, fChannel=32, bilinear=True):
#         super(UNet3D_PSA, self).__init__()
#         self.inChannel = inChannel
#         self.outChannel = outChannel
#         self.fChannel=fChannel
#         self.bilinear = bilinear

#         self.inc = DoubleConv3D(inChannel, fChannel)
#         self.down1 = Down3D(fChannel, fChannel*2)
#         self.down2 = Down3D(fChannel*2, fChannel*4)
#         self.down3 = Down3D(fChannel*4, fChannel*8)
#         factor = 2 if bilinear else 1
#         self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

#         #decoder
#         self.up1 = nn.ModuleList([Up3D(fChannel*16, fChannel*8 // factor, bilinear) for i in range(5)])
#         self.up2 = nn.ModuleList([Up3D(fChannel*8, fChannel*4 // factor, bilinear)for i in range(5)])
#         self.up3 = nn.ModuleList([Up3D(fChannel*4, fChannel*2 // factor, bilinear)for i in range(5)])
#         self.up4 = nn.ModuleList([Up3D(fChannel*2, fChannel, bilinear)for i in range(5)])
#         self.outc = nn.ModuleList([OutConv3D(fChannel, outChannel)for i in range(5)])
#         self.train_decoder0=True

#     def forward(self, x, sentence):
#         if self.train_decoder0:
#             decoder=0
#             self.train_decoder0=False
#         else:
#             if sentence[-2]=='A':#Desai, Neil
#                 decoder=1
#             elif sentence[-2]=='D':#Hannan, Raquibul
#                 decoder=2
#             elif sentence[-2]=='I':#Yang, Daniel
#                 decoder=3
#             elif sentence[-2]=='J':#Garant, Aurelie
#                 decoder=4
#             else:
#                 decoder=0 #Others
#             self.train_decoder0=True

#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         x = self.up1[decoder](x5, x4)
#         x = self.up2[decoder](x, x3)
#         x = self.up3[decoder](x, x2)
#         x = self.up4[decoder](x, x1)
#         logits = self.outc[decoder](x)
#         return logits

# class CLIPUNet3D(nn.Module):
#     def __init__(self, inChannel, outChannel, fChannel=32, bilinear=True):
#         super(CLIPUNet3D, self).__init__()
#         self.inChannel = inChannel
#         self.outChannel = outChannel
#         self.fChannel=fChannel
#         self.bilinear = bilinear

#         self.inc = DoubleConv3D(inChannel, fChannel)
#         self.down1 = Down3D(fChannel, fChannel*2)
#         self.down2 = Down3D(fChannel*2, fChannel*4)
#         self.down3 = Down3D(fChannel*4, fChannel*8)
#         factor = 2 if bilinear else 1
#         self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

#         self.up1 = Up3D(fChannel*16, fChannel*8 // factor, bilinear)
#         self.up2 = Up3D(fChannel*8, fChannel*4 // factor, bilinear)
#         self.up3 = Up3D(fChannel*4, fChannel*2 // factor, bilinear)
#         self.up4 = Up3D(fChannel*2, fChannel, bilinear)
#         self.outc = OutConv3D(fChannel, outChannel)

#         self.clip_model, _ = clip.load("ViT-B/32", device='cpu')
#         self.downtext=nn.AvgPool1d(kernel_size=2,stride=2)
#         #text = clip.tokenize([r'There is no spacer hydrogel in the patient.',r'There is a spacer hydrogel in the patient.'])
#         #text = clip.tokenize([r'There is a type 2 spacer hydrogel in the patient.',r'There is no type 2 spacer hydrogel in the patient.'])
#         return

#     def forward(self, x, text):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         text=clip.tokenize(text).to(x.device)
#         text_feature = self.clip_model.encode_text(text)
#         text_feature.unsqueeze_(dim=1).detach_()
#         text_feature=self.downtext(text_feature)
#         text_feature=(text_feature-0.015)/0.27
#         #text_feature=self.downtext(text_feature)
#         x5=x5*text_feature.view(1,text_feature.shape[2],1,1,1)
#         x4=x4*text_feature.view(1,text_feature.shape[2],1,1,1)
#         text_feature=self.downtext(text_feature)
#         x3=x3*text_feature.view(1,text_feature.shape[2],1,1,1)
#         text_feature=self.downtext(text_feature)
#         x2=x2*text_feature.view(1,text_feature.shape[2],1,1,1)
#         text_feature=self.downtext(text_feature)
#         x1=x1*text_feature.view(1,text_feature.shape[2],1,1,1)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

# if __name__ == '__main__':
#     model1=UNet3D_PSA(1,1,fChannel=32)
#     input=torch.randn(size=[1,1,64,64,64])
#     output1=model1(input,0)
#     pass


import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),  # Changed to InstanceNorm for batch_size=1
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CLIPUNet2D(nn.Module):
    """
    CLIP-conditioned U-Net for single-channel heatmap generation.
    Takes an RGB image and text instruction, outputs a continuous heatmap [0, 1].
    """
    def __init__(self, in_channels=3, out_channels=1, fChannel=64, bilinear=True, clip_model_name="ViT-B/32"):
        super(CLIPUNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fChannel = fChannel
        self.bilinear = bilinear

        # U-Net encoder
        self.inc = DoubleConv(in_channels, fChannel)
        self.down1 = Down(fChannel, fChannel * 2)
        self.down2 = Down(fChannel * 2, fChannel * 4)
        self.down3 = Down(fChannel * 4, fChannel * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(fChannel * 8, fChannel * 16 // factor)

        # U-Net decoder
        self.up1 = Up(fChannel * 16, fChannel * 8 // factor, bilinear)
        self.up2 = Up(fChannel * 8, fChannel * 4 // factor, bilinear)
        self.up3 = Up(fChannel * 4, fChannel * 2 // factor, bilinear)
        self.up4 = Up(fChannel * 2, fChannel, bilinear)
        self.outc = OutConv(fChannel, out_channels)

        # CLIP text encoder
        self.clip_model, _ = clip.load(clip_model_name, device='cuda' if torch.cuda.is_available() else "cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False  # Freeze CLIP weights

        # Text feature projection layers to match U-Net channel dimensions
        clip_dim = 512 if "ViT-B" in clip_model_name else 768
        self.text_proj1 = nn.Linear(clip_dim, fChannel * 16 // factor)  # bottleneck
        self.text_proj2 = nn.Linear(clip_dim, fChannel * 8)
        self.text_proj3 = nn.Linear(clip_dim, fChannel * 4)
        self.text_proj4 = nn.Linear(clip_dim, fChannel * 2)
        self.text_proj5 = nn.Linear(clip_dim, fChannel)

    def forward(self, image, text):
        """
        Args:
            image: [B, 3, H, W] - RGB image
            text: List[str] - text instructions (batch of strings)
        
        Returns:
            heatmap: [B, 1, H, W] - continuous heatmap in range [0, 1]
        """
        # Encode image through U-Net
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Encode text with CLIP
        text_tokens = clip.tokenize(text, truncate=True).to(image.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens).float()  # [B, 512]

        # Project text features to different resolutions
        text_feat1 = self.text_proj1(text_features)  # [B, fChannel*16//factor]
        text_feat2 = self.text_proj2(text_features)  # [B, fChannel*8]
        text_feat3 = self.text_proj3(text_features)  # [B, fChannel*4]
        text_feat4 = self.text_proj4(text_features)  # [B, fChannel*2]
        text_feat5 = self.text_proj5(text_features)  # [B, fChannel]

        # Modulate encoder features with text (element-wise multiplication)
        x5 = x5 * text_feat1.view(-1, text_feat1.shape[1], 1, 1)
        x4 = x4 * text_feat2.view(-1, text_feat2.shape[1], 1, 1)
        x3 = x3 * text_feat3.view(-1, text_feat3.shape[1], 1, 1)
        x2 = x2 * text_feat4.view(-1, text_feat4.shape[1], 1, 1)
        x1 = x1 * text_feat5.view(-1, text_feat5.shape[1], 1, 1)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # Apply sigmoid for continuous output [0, 1]
        heatmap = torch.sigmoid(logits)
        return heatmap


if __name__ == '__main__':
    # Test the model
    model = CLIPUNet2D(in_channels=3, out_channels=1, fChannel=64)
    model.eval()
    
    # Dummy input
    image = torch.randn(2, 3, 480, 640)
    text = ["Go to the kitchen", "Find the cup on the table"]
    
    with torch.no_grad():
        output = model(image, text)
    
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")