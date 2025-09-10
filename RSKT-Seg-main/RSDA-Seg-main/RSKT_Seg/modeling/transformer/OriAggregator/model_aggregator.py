
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_agg_class import ClassTransformerLayer
from .model_agg_spatial import SwinTransformerBlockWrapper

class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256) -> None:
        super().__init__()
        self.swin_block = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.attention = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)

    def forward(self, x, appearance_guidance, text_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """

        x = self.swin_block(x, appearance_guidance)
        x = self.attention(x, text_guidance)
        return x
        