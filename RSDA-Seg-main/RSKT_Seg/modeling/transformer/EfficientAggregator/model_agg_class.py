import numpy as np
import torch
import math

from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

TORCH_VERSION = torch.__version__

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU6(inplace=True)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.act(x1) * x2
        x = self.fc3(x)
        x = self.act(x)
        return x


class Attention(nn.Module):
    """Refer form SegFormer"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.k = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        if TORCH_VERSION < '2.2.0':
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = attn_drop
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, guidance):
        B, N, C = x.shape
        x_ = torch.cat([x, guidance], dim=-1)
        
        q = self.q(x_).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if TORCH_VERSION < '2.2.0':
            attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        else:
            # if torch>=2.2, we recommend flash attention
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MiTBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.,
                 attn_drop=0., drop=0., qkv_bias=True, norm_layer=nn.LayerNorm):
        super(MiTBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.dim = dim
        self.attention = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=3*dim, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, guidance):
        x = x + self.drop_path(self.attention(self.norm1(x), guidance))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassFusionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, drop_path=0., pooling_size=(2, 2)):
        super(ClassFusionBlock, self).__init__()
        inter_channels = in_channels
        self.sa = MiTBlock(inter_channels, num_heads, drop_path=drop_path)
        self.pool = nn.AvgPool2d(pooling_size)
        self.dim = in_channels

    def forward(self, x, guidance):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        input = x

        x = self.pool(x) # (B T) C H/2 W/2
        _, _, H_pool, W_pool = x.shape
        x = rearrange(x, '(B T) C H W -> (B H W) T C', T=T)
        guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)
        x = self.sa(x, guidance)

        x = rearrange(x, '(B H W) T C -> (B T) C H W', T=T, H=H_pool, W=W_pool)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = input + x
        return x
