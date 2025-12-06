import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class KmapGenerator(nn.Module):
    def __init__(self, k_min=0.1, k_max=0.5):
        super(KmapGenerator, self).__init__()
        
        self.k_min = k_min
        self.k_max = k_max
        
        self.inc = ConvBlock(3, 16)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(16, 32)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(32, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = ConvBlock(64 + 32, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = ConvBlock(32 + 16, 16)
        
        self.outc = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        
        # Decoder
        x = self.up1(x3)
        # Skip connection
        x = torch.cat([x, F.interpolate(x2, x.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x = self.conv3(x)
        
        x = self.up2(x)
        # Skip connection
        x = torch.cat([x, F.interpolate(x1, x.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x = self.conv4(x)
        
        # Output
        k_map = self.outc(x)
        k_map = self.sigmoid(k_map)
        
        # Scale k_map to [k_min, k_max]
        k_map = self.k_min + k_map * (self.k_max - self.k_min)
        
        return k_map
