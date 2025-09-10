import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class RSKT_Upsample(nn.Module):
    """"Upscaling using feat from dino and clip"""
    def __init__(self, 
                in_channels, 
                out_channels, 
                clip_guidance_channels, 
                dino_guidance_channels,
                use_remote_clip,
                use_remote_dino):
        super().__init__()
        if use_remote_clip == True and use_remote_dino == True:
            self.up = nn.ConvTranspose2d(in_channels, in_channels-clip_guidance_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels+clip_guidance_channels+dino_guidance_channels , out_channels)
        elif use_remote_clip == True and use_remote_dino == False:
            self.up = nn.ConvTranspose2d(in_channels, in_channels-clip_guidance_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels+clip_guidance_channels , out_channels)
        elif use_remote_clip == False and use_remote_dino == True:
            self.up = nn.ConvTranspose2d(in_channels, in_channels-dino_guidance_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels+dino_guidance_channels , out_channels)
        elif use_remote_clip == False and use_remote_dino == False:
            self.up = nn.ConvTranspose2d(in_channels, in_channels-dino_guidance_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels , out_channels)

    def forward(self, x, clip_guidance, clip_guidance_remote, dino_guidance):
        x = self.up(x)
        if clip_guidance is not None:
            T = x.size(0) // clip_guidance.size(0)
            clip_guidance = repeat(clip_guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, clip_guidance], dim=1)
        if clip_guidance_remote is not None:
            T = x.size(0) // clip_guidance_remote.size(0)
            clip_guidance_remote = repeat(clip_guidance_remote, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, clip_guidance_remote], dim=1)
        if dino_guidance is not None: 
            T = x.size(0) // dino_guidance.size(0)
            dino_guidance = repeat(dino_guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x,dino_guidance], dim=1)
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
