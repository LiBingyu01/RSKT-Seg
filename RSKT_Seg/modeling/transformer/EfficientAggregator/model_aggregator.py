import torch
import math
import random
import torch.nn as nn

from einops import rearrange, repeat
from .model_agg_spatial import SptialFusionBlock
from .model_agg_class import ClassFusionBlock

  
class EffAggregatorLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, drop_path=0., sr_ratio=2):
        super().__init__()
        self.spatial_fusion = SptialFusionBlock(hidden_dim, num_heads, drop_path=drop_path, sr_ratio=sr_ratio)
        self.class_fusion = ClassFusionBlock(hidden_dim, num_heads, drop_path=drop_path)

    def forward(self, x, img_guidance, text_guidance):
        B, _, T, H, W = x.shape
        
        # spatial fusion
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        img_guidance = repeat(img_guidance, "B C H W -> (B T) C H W", T=T)
        x = self.spatial_fusion(x, img_guidance)
        
        # class fusion
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        x = self.class_fusion(x, text_guidance)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)

        return x
