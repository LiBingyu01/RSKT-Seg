import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from .EfficientAggregator import EffAggregatorLayer
from .OriAggregator import OriAggregatorLayer
from .RSKT_Upsample import RSKT_Upsample
from .visualize_corr import visualize_corr
class RSKT_Decoder(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        decoder_clip_guidance_dims=(256, 128),
        decoder_clip_guidance_proj_dims=(32, 16),
        decoder_dino_guidance_dims=(256, 128),
        decoder_dino_guidance_proj_dims=(32, 16),
        feat_dim = 512,
        use_clip_corr = True,
        use_dino_corr = True,
        fusion_type = "simple_separate",
        use_remote_clip = True,
        use_remote_dino = True,
        use_rotate = True,
        num_layers=4,
        use_efficient=True,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:

        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        self.use_remote_clip = use_remote_clip
        self.use_remote_dino = use_remote_dino
        self.use_rotate = use_rotate
        if use_efficient:
            self.layers = nn.ModuleList([
                EffAggregatorLayer(
                    hidden_dim=hidden_dim
                ) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                OriAggregatorLayer(
                    hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                    nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
                ) for _ in range(num_layers)
            ])

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_clip_guidance_proj_dims)
        ]) if decoder_clip_guidance_dims[0] > 0 else None
        
        if self.use_rotate:
            self.conv1 = nn.Conv2d(prompt_channel*4, hidden_dim, kernel_size=7, stride=1, padding=3) 
        else:
            self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3) 

        if self.use_remote_clip:
            self.CLIP_decoder_guidance_projection_remote = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ) for d, dp in zip(decoder_guidance_dims, decoder_clip_guidance_proj_dims)
            ]) if decoder_clip_guidance_dims[0] > 0 else None
        else:
            self.CLIP_decoder_guidance_projection_remote=None

        if self.use_remote_dino: 
            # dino corr
            if fusion_type=='simple_concatenation':
                self.conv1 = nn.Conv2d(prompt_channel*5, hidden_dim, kernel_size=7, stride=1, padding=3)
            elif fusion_type=='simple_mean':
                self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
            elif fusion_type=='simple_separate':    
                self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
                self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3) 
            # decoder
            self.DINO_decoder_guidance_projection = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ) for d, dp in zip(decoder_guidance_dims, decoder_dino_guidance_proj_dims)
            ]) if decoder_dino_guidance_dims[0] > 0 else None
        else:
            self.conv2=None
            self.fusion_corr=None
            self.fusion_feats=None
            self.DINO_decoder_guidance_projection=None
        
        # RSKT_Upsample
        self.Fusiondecoder1=RSKT_Upsample(hidden_dim, 
        decoder_dims[0],
        decoder_clip_guidance_proj_dims[0],
        decoder_dino_guidance_proj_dims[0],
        use_remote_clip = use_remote_clip,
        use_remote_dino = use_remote_dino)
        self.Fusiondecoder2=RSKT_Upsample(decoder_dims[0], 
        decoder_dims[1], 
        decoder_clip_guidance_proj_dims[1], 
        decoder_dino_guidance_proj_dims[1],
        use_remote_clip = use_remote_clip,
        use_remote_dino = use_remote_dino)

        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.pad_len = pad_len
        
    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def correlation_rotate(self, img_feats, text_feats):
        img_feats0 = F.normalize(img_feats[0], dim=1) # B C H W
        img_feats1 = F.normalize(img_feats[1], dim=1) # B C H W
        img_feats2 = F.normalize(img_feats[2], dim=1) # B C H W
        img_feats3 = F.normalize(img_feats[3], dim=1) # B C H W
        img_feats1= torch.rot90(img_feats1, k=3, dims=(2, 3))
        img_feats2 = torch.rot90(img_feats2, k=2, dims=(2, 3))
        img_feats3 = torch.rot90(img_feats3, k=1, dims=(2, 3))
        img_feats = torch.cat((img_feats0.unsqueeze(dim=1),img_feats1.unsqueeze(dim=1),img_feats2.unsqueeze(dim=1),img_feats3.unsqueeze(dim=1)),dim=1)

        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bnchw, btpc -> bnpthw', img_feats, text_feats)
        corr = rearrange(corr, 'B N P T H W -> B (N P) T H W')
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def simple_separate_corr(self,clip_corr,dino_corr,files_name):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')

        # visualize_corr(clip_corr.permute(1,0,2,3)[0].unsqueeze(0), files_name[0], save_prefix='./vis_cost_clip_DLRSD/')
        # visualize_corr(dino_corr.permute(1,0,2,3), files_name[0], save_prefix='./vis_cost_dino_DLRSD/')

        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)

        # clip_corr_mean = clip_embed_corr.mean(dim=1, keepdim=True)
        # dino_corr_mean = dino_embed_corr.mean(dim=1, keepdim=True)

        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr

    def simple_concatenation_corr(self, clip_corr, dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = self.conv1(torch.cat([clip_corr,dino_corr],dim = 1))
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr 

    def simple_mean_corr(self, clip_corr, dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = self.conv1(torch.mean(
            torch.cat([clip_corr,dino_corr],dim=1), 
            dim = 1, 
            keepdim=True))
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr 

    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def Fusion_conv_decoer(self, x, clip_guidance, clip_guidance_remote, dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],clip_guidance_remote[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],clip_guidance_remote[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

    def forward(self, files_name, img_feats, dino_feat, text_feats, appearance_guidance, appearance_guidance_remote, dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T is class number
            apperance_guidance: tuple of (B, C, H, W)
        """
        classes = None

        if dino_feat is not None and img_feats is not None:
            if self.fusion_type == 'simple_separate':
                if isinstance(img_feats, list):
                    corr = self.correlation_rotate(img_feats, text_feats)
                else:
                    corr = self.correlation(img_feats, text_feats)
                dino_corr = self.correlation(dino_feat,text_feats)
                fused_corr_embed, clip_embed_corr, dino_embed_corr  = self.simple_separate_corr(clip_corr=corr, dino_corr=dino_corr,files_name=files_name)
                fused_corr_embed = fused_corr_embed + clip_embed_corr
            elif self.fusion_type == 'simple_concatenation':
                if isinstance(img_feats, list):
                    corr = self.correlation_rotate(img_feats, text_feats)
                else:
                    corr = self.correlation(img_feats, text_feats)
                dino_corr = self.correlation(dino_feat,text_feats)
                fused_corr_embed = self.simple_concatenation_corr(corr,dino_corr)
            elif self.fusion_type == 'simple_mean':
                if isinstance(img_feats, list):
                    corr = self.correlation_rotate(img_feats, text_feats)
                else:
                    corr = self.correlation(img_feats, text_feats)
                dino_corr = self.correlation(dino_feat,text_feats)
                fused_corr_embed = self.simple_mean_corr(corr,dino_corr)
                
        elif dino_feat is not None and img_feats is None:
            corr = self.correlation(dino_feat,text_feats)
            embed_corr = self.corr_embed(corr)
            fused_corr_embed = embed_corr
            print(f"2222222222222")
        elif dino_feat is None and img_feats is not None:
            if isinstance(img_feats, list):
                corr = self.correlation_rotate(img_feats, text_feats)
            else:
                corr = self.correlation(img_feats,text_feats)
            embed_corr = self.corr_embed(corr)
            fused_corr_embed = embed_corr
            print(f"3333333333333333")

        projected_guidance, projected_text_guidance = None, None
        CLIP_projected_decoder_guidance = [None, None]
        CLIP_projected_decoder_guidance_remote = [None, None]
        DINO_projected_decoder_guidance = [None,None]

        if self.guidance_projection is not None and appearance_guidance is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.guidance_projection is not None and appearance_guidance is None:
            projected_guidance = self.guidance_projection(dino_feat)

        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]

        if self.CLIP_decoder_guidance_projection_remote is not None:
            CLIP_projected_decoder_guidance_remote = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection_remote, appearance_guidance_remote)]

        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, 
                                        CLIP_projected_decoder_guidance, 
                                        CLIP_projected_decoder_guidance_remote, 
                                        DINO_projected_decoder_guidance)

        return logit