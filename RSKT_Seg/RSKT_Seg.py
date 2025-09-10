# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange
from .vision_transformer import vit_base
import os
from .visulize_kmeans import kmeans_cluster_and_fuse
from .visualize_corr import visualize_corr

def BuildRSIB(Weights):
    model = vit_base(patch_size=8, num_classes=0)
    if os.path.isfile(Weights):
        state_dict = torch.load(Weights, map_location='cpu')
        checkpoint_key = "teacher"
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(Weights, msg))
        model = model.float()
        return model
    else:
        raise FileNotFoundError(f"Pretrained weights not found at {Weights}. Please check the file path.")
    
@META_ARCH_REGISTRY.register()
class RSKT_Seg(nn.Module):
    @configurable
    
    
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        use_clip: bool,
        clip_decod_guid_dim: list,
        text_guidance_dim:int,
        # use_rotate
        use_rotate : bool,
        # clip_remote
        use_remote_clip:bool,
        clip_pretrained_remote:str,
        # dino
        use_remote_dino:bool,
        dino_decod_guid_dim: list,
        dino_weights:str,
        dino_ft: str,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone

        self.use_clip = use_clip
        self.use_rotate = use_rotate
        self.use_remote_clip = use_remote_clip
        self.use_remote_dino = use_remote_dino

        self.clip_decod_dim = clip_decod_guid_dim
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        # set finetune mode
        self.clip_finetune = clip_finetune
        self.set_clip_finetune(clip_finetune)
        # CLIP
        self.sliding_window = sliding_window
        self.dino_resolution = (384,384)
        if clip_pretrained == "ViT-B/16": 
            self.clip_resolution = (384, 384)
            self.proj_dim = 768
        else: 
            self.clip_resolution = (336, 336)
            self.proj_dim = 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2) if self.use_clip and self.clip_decod_dim[0]!=0 else None
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4) if self.use_clip and self.clip_decod_dim[1]!=0 else None
        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
        
        # remote clip
        if self.use_remote_clip:
            self.set_clip_remote_finetune_frozen()
            if clip_pretrained_remote == "ViT-B/32": 
                self.proj_dim_remote = 768
            else:
                self.proj_dim_remote = 1024
            self.clip_resolution_remote = (768,768)
            self.upsample1_remote = nn.ConvTranspose2d(self.proj_dim_remote, 256, kernel_size=2, stride=2) if self.use_clip and self.clip_decod_dim[0]!=0 else None
            self.upsample2_remote = nn.ConvTranspose2d(self.proj_dim_remote, 128, kernel_size=4, stride=4) if self.use_clip and self.clip_decod_dim[1]!=0 else None
            self.layer_indexes_remote = [3, 7] if clip_pretrained_remote == "ViT-B/32" else [7, 15] 
            self.layers_remote = []
            for l in self.layer_indexes_remote:
                self.sem_seg_head.predictor.clip_model_remote.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers_remote.append(o))
        # dino
        if self.use_remote_dino:
            self.dino_model = self.build_dino_set_finetune(dino_weights, dino_ft)
            self.dino_decod_dim = dino_decod_guid_dim
            self.dino_decod_proj1 = nn.Conv2d(in_channels = 768, out_channels=256, kernel_size=1, stride=1, padding=0) if self.dino_model and self.dino_decod_dim[0]!=0 else None
            self.dino_decod_proj2 = nn.ConvTranspose2d(in_channels= 768, out_channels=128, kernel_size=2, stride=2) if self.dino_model and self.dino_decod_dim[0]!=0 else None
            self.dino_down_sample = nn.Conv2d(in_channels=768, out_channels=text_guidance_dim, kernel_size=2, stride=2, padding=0) if self.dino_model else None

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "use_clip":cfg.MODEL.SEM_SEG_HEAD.USE_CLIP_CORR, 
            "clip_decod_guid_dim":cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_DIMS,
            "text_guidance_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM,
            # ROTATE
            "use_rotate":cfg.MODEL.SEM_SEG_HEAD.USE_ROTATE,
            "clip_pretrained_remote":cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_REMOTE,
            # REMOTE CLIP
            "use_remote_clip" : cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP,
            # DINO
            "dino_decod_guid_dim":cfg.MODEL.SEM_SEG_HEAD.DECODER_DINO_GUIDANCE_DIMS,
            "dino_ft":cfg.MODEL.SEM_SEG_HEAD.DINO_FINETUNE,
            "use_remote_dino": cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_DINO,
            "dino_weights": cfg.MODEL.SEM_SEG_HEAD.DINO_WEIGHTS
        }

    def build_dino_set_finetune(self, dino_weights, dino_ft):
        dino = BuildRSIB(dino_weights)
        for name, params in dino.named_parameters():
            if dino_ft == "attention":
                if "attn.qkv.weight" in name:
                    params.requires_grad = True
                elif "pos_embed" in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            elif dino_ft == "full":
                params.requires_grad = True
            else:
                params.requires_grad = False
        return dino

    def set_clip_finetune(self, clip_finetune):
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if clip_finetune == "freezeIMG":
                if "attn" in name:
                    # QV fine-tuning for attention blocks
                    params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                elif "position" in name:
                    params.requires_grad = True
                else: params.requires_grad = False
                if "visual" in name:
                    params.requires_grad = False   
            elif "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            
            else:
                params.requires_grad = False

    def set_clip_remote_finetune_frozen(self):
        for name, params in self.sem_seg_head.predictor.clip_model_remote.named_parameters():
            params.requires_grad = False
    @property
    def device(self):
        return self.pixel_mean.device
    # @profile(precision=4,stream=open('./log.txt','w+',encoding="utf-8"))
    def forward(self, batched_inputs):

        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """

        if self.training:
            # images_shape: 384*384
            images = [x["image"].to(self.device) for x in batched_inputs]
            # print(batched_inputs)
            # print(ss)
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
            self.layers = []
            clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )

            if self.use_rotate:
                clip_images_resized_90 = torch.rot90(clip_images_resized, k=1, dims=(2, 3))
                clip_images_resized_180  = torch.rot90(clip_images_resized, k=2, dims=(2, 3))
                clip_images_resized_270 = torch.rot90(clip_images_resized, k=3, dims=(2, 3))
            if self.use_remote_clip:
                self.layers_remote = []
                clip_images_resized_remote = F.interpolate(clip_images.tensor, size=self.clip_resolution_remote, mode='bilinear', align_corners=False, )
            if self.use_remote_dino:
                dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, )
        else:
            if self.sliding_window:
                with torch.no_grad():
                    kernel=384
                    overlap=0.333
                    out_res=[640, 640]
                    images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
                    stride = int(kernel * (1 - overlap))
                    unfold = nn.Unfold(kernel_size=kernel, stride=stride)
                    fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

                    image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
                    image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
                    global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
                    image = torch.cat((image, global_image), dim=0)

                    images = (image - self.pixel_mean) / self.pixel_std
                    clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
                    # clip
                    self.layers = []
                    clip_images_resized = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )

                    if self.use_rotate:
                        clip_images_resized_90 = torch.rot90(clip_images_resized, k=1, dims=(2, 3))
                        clip_images_resized_180  = torch.rot90(clip_images_resized, k=2, dims=(2, 3))
                        clip_images_resized_270 = torch.rot90(clip_images_resized, k=3, dims=(2, 3))
                    if self.use_remote_clip:
                        self.layers_remote = []  
                        clip_images_resized_remote = F.interpolate(clip_images, size=self.clip_resolution_remote, mode='bilinear', align_corners=False, )
                    if self.use_remote_dino:
                        dino_images_resized = F.interpolate(clip_images, size=self.dino_resolution, mode='bilinear', align_corners=False, )    
            else:
                with torch.no_grad():
                    images = [x["image"].to(self.device) for x in batched_inputs]
                    clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
                    clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
                    # CLIP
                    self.layers = []
                    clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
                    if self.use_rotate:
                        clip_images_resized_90 = torch.rot90(clip_images_resized, k=1, dims=(2, 3))
                        clip_images_resized_180  = torch.rot90(clip_images_resized, k=2, dims=(2, 3))
                        clip_images_resized_270 = torch.rot90(clip_images_resized, k=3, dims=(2, 3))
                    if self.use_remote_clip:
                        self.layers_remote = []
                        clip_images_resized_remote = F.interpolate(clip_images.tensor, size=self.clip_resolution_remote, mode='bilinear', align_corners=False, )
                    if self.use_remote_dino:
                        dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, )

        if self.use_remote_dino:
            dino_feat = self.dino_model.get_intermediate_layers(dino_images_resized, n=12) # actually only 12 layers, but use a large num to avoid ambiguity
            dino_patch_feat_last_unfold = rearrange(dino_feat[-1][:,1:,:],"B (H W) C -> B C H W", H=48)

            dino_feat_input = self.dino_down_sample(dino_patch_feat_last_unfold) # B,512,24,24
            dino_feat_L4 = rearrange(dino_feat[3][:,1:,:],"B (H W) C -> B C H W", H=48)
            dino_feat_L8 = rearrange(dino_feat[7][:,1:,:],"B (H W) C -> B C H W", H=48)
            
            dino_feat_L4_proj = self.dino_decod_proj1(dino_feat_L4) if self.dino_decod_proj1 is not None else None
            dino_feat_L8_proj = self.dino_decod_proj2(dino_feat_L8) if self.dino_decod_proj2 is not None else None
            dino_feat_guidance = [dino_feat_L4_proj,dino_feat_L8_proj]
        else:
            dino_feat_input, dino_feat_guidance = None, None
        
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        clip_image_features = clip_features[:, 1:, :]
        res3 = rearrange(clip_image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4) if self.upsample1 is not None else None
        res5 = self.upsample2(res5) if self.upsample2 is not None else None
        clip_features_guidance = {'res5': res5, 'res4': res4, 'res3': res3,}

        if self.use_rotate:
            clip_features1 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_90, dense=True)
            clip_features2 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_180, dense=True)
            clip_features3 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_270, dense=True)
            clip_features_input = [clip_features,clip_features1,clip_features2,clip_features3]
        else:
            clip_features_input = clip_features

        if self.use_remote_clip:
            _ = self.sem_seg_head.predictor.clip_model_remote.encode_image(clip_images_resized_remote, dense=True)
            res4 = rearrange(self.layers_remote[0][1:, :, :], "(H W) B C -> B C H W", H=24)
            res5 = rearrange(self.layers_remote[1][1:, :, :], "(H W) B C -> B C H W", H=24)
            res4 = self.upsample1_remote(res4) if self.upsample1_remote is not None else None
            res5 = self.upsample2_remote(res5) if self.upsample2_remote is not None else None
            clip_features_guidance_remote = {'res5': res5, 'res4': res4,}
        else:
            clip_features_guidance_remote = None

        files_name = [x["file_name"] for x in batched_inputs]
        print(files_name)

        outputs = self.sem_seg_head(files_name,
        clip_features_input, 
        dino_feat_input, 
        clip_features_guidance, 
        clip_features_guidance_remote,
        dino_feat_guidance)
        
        visualize_corr(outputs, files_name[0], save_prefix='./vis_cost_out_DLRSD/')
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses
            
        else:
            if self.sliding_window:
                with torch.no_grad():
                    outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
                    outputs = outputs.sigmoid()
                    
                    global_output = outputs[-1:]
                    global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
                    outputs = outputs[:-1]
                    outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
                    outputs = (outputs + global_output) / 2.

                    height = batched_inputs[0].get("height", out_res[0])
                    width = batched_inputs[0].get("width", out_res[1])
                    output = sem_seg_postprocess(outputs[0], out_res, height, width)
                    return [{'sem_seg': output}]
            else:
                with torch.no_grad():
                    outputs = outputs.sigmoid()
                    image_size = clip_images.image_sizes[0]
                    height = batched_inputs[0].get("height", image_size[0])
                    width = batched_inputs[0].get("width", image_size[1])

                    output = sem_seg_postprocess(outputs[0], image_size, height, width)
                    processed_results = [{'sem_seg': output}]
                    return processed_results        
        