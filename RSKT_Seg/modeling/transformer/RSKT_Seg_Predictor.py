# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .RSKT_Decoder import RSKT_Decoder
from RSKT_Seg.third_party import clip
from RSKT_Seg.third_party import imagenet_templates

import numpy as np
import open_clip

class RSKT_Seg_Predictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        clip_pretrained_remote: str,
        clip_pretrained_weights_remote: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_clip_guidance_dims: list,
        decoder_clip_guidance_proj_dims:list, 
        decoder_dino_guidance_dims:list,
        decoder_dino_guidance_proj_dims:list, 
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        use_clip_corr: bool,
        use_dino_corr: bool,
        fusion_type: str,
        use_remote_clip: bool, 
        use_remote_dino: bool, 
        use_rotate: bool, 
        num_heads: int,
        num_layers: tuple,
        use_efficient: bool,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
    ):
        """
        Args:
            
        """
        super().__init__()
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None

        if use_remote_clip:
            clip_model_remote, _ = clip.load(clip_pretrained_remote, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
            ckpt = torch.load(clip_pretrained_weights_remote)
            mapped_state_dict = {}
            for key, value in ckpt.items():
                if 'in_proj_weight' in key:
                    # Split the in_proj_weight into q_proj_weight, k_proj_weight, and v_proj_weight
                    qkv = torch.chunk(value, 3, dim=0)
                    base_key = key.replace('in_proj_weight', '')
                    mapped_state_dict[base_key + 'q_proj_weight'] = qkv[0]
                    mapped_state_dict[base_key + 'k_proj_weight'] = qkv[1]
                    mapped_state_dict[base_key + 'v_proj_weight'] = qkv[2]
                else:
                    # Directly copy other weights
                    mapped_state_dict[key] = value
            message = clip_model_remote.load_state_dict(mapped_state_dict,strict=True)
            print(f"message_{message}")

        # CLIP
        clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        
        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates

        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        if use_remote_clip:
            self.clip_model_remote = clip_model_remote.float()
        
        transformer = RSKT_Decoder(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            decoder_clip_guidance_dims=decoder_clip_guidance_dims,
            decoder_clip_guidance_proj_dims=decoder_clip_guidance_proj_dims,
            decoder_dino_guidance_dims=decoder_dino_guidance_dims,
            decoder_dino_guidance_proj_dims=decoder_dino_guidance_proj_dims,
            use_clip_corr = use_clip_corr,
            use_dino_corr = use_dino_corr,
            fusion_type = fusion_type,
            use_remote_clip = use_remote_clip,
            use_remote_dino = use_remote_dino,
            use_rotate = use_rotate,
            num_layers=num_layers,
            use_efficient=use_efficient,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            )
        self.transformer = transformer
        
        self.tokens = None
        self.cache = None

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["clip_pretrained_remote"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_REMOTE
        ret["clip_pretrained_weights_remote"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_WEIGHTS_REMOTE
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS
        ret["decoder_clip_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_DIMS
        ret["decoder_clip_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_PROJ_DIMS
        ret["decoder_dino_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DINO_GUIDANCE_DIMS
        ret["decoder_dino_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DINO_GUIDANCE_PROJ_DIMS       
        ret['use_clip_corr'] = cfg.MODEL.SEM_SEG_HEAD.USE_CLIP_CORR
        ret['use_dino_corr'] = cfg.MODEL.SEM_SEG_HEAD.USE_DINO_CORR

        ret["use_remote_clip"] = cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP
        ret["use_remote_dino"] = cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_DINO
        ret["use_rotate"] = cfg.MODEL.SEM_SEG_HEAD.USE_ROTATE

        ret['fusion_type'] = cfg.MODEL.SEM_SEG_HEAD.FUSION_TYPE
        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["use_efficient"] = cfg.MODEL.SEM_SEG_HEAD.USE_EFFICIENT
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        return ret

    def forward(self, files_name, x, dino_feat, vis_guidance, vis_guidance_remote, dino_guidance, prompt=None, gt_cls=None):
        if vis_guidance is not None:
            vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        else:
            vis = None
        
        if vis_guidance_remote is not None:
            vis_remote = [vis_guidance_remote[k] for k in vis_guidance_remote.keys()][::-1]
        else:
            vis_remote = None

        text = self.class_texts if self.training else self.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text

        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        if x is not None:
            if isinstance(x, list):
                text = text.repeat(x[0].shape[0], 1, 1, 1)
            else:
                text = text.repeat(x.shape[0], 1, 1, 1)
        else:
            text = text.repeat(dino_feat.shape[0], 1, 1, 1)
        # text: [B,N,1,512]
        out = self.transformer(files_name, x, dino_feat, text, vis, vis_remote, dino_guidance)
        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)

            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens
        # tokens: [N,77]
        class_embeddings = clip_model.encode_text(tokens, prompt)

        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings