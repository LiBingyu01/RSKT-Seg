# =================================
# .sh file to reproduce our RSKT-Seg results
# =================================


# =================================
# eval all the open-sourced .pth file of RSKT-Seg
# sh eval.sh $config $gpus $output $opts
# $config : configuration file, refer to the /configs/xxx
# $gpus: gpu number, we use 4 4090 Nvidia GPUs
# $output: path to the output file 
# $opts: the parameter you want to use in infer 
# (For $opts, we have already specified it in the following code, you only run to reproduce our results)
# =================================

# =================================
# DLRSD CLIP-B
# =================================
sh eval.sh \
    configs/vitb_384_DLRSD.yaml \
    4 \
    output_vitb_384_DLRSD/eval \
    MODEL.WEIGHTS fx_download/0SAVEoutput_vitb_384_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth \
    MODEL.SEM_SEG_HEAD.USE_ROTATE True \
    MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP True \
    MODEL.SEM_SEG_HEAD.USE_REMOTE_DINO True \
    MODEL.SEM_SEG_HEAD.NUM_LAYERS 5 \

# =================================
# iSAID CLIP-B
# =================================
sh eval.sh \
    configs/vitb_384_iSAID.yaml \
    4 \
    output_vitb_384_iSAID/eval \
    MODEL.WEIGHTS fx_download/0SAVEoutput_vitb_384_iSAID_rotate_dino_remoteclip_3W_layer3/model_final.pth \
    MODEL.SEM_SEG_HEAD.USE_ROTATE True \
    MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP True \
    MODEL.SEM_SEG_HEAD.USE_REMOTE_DINO True \
    MODEL.SEM_SEG_HEAD.NUM_LAYERS 3 \


# =================================
# DLRSD CLIP-L
# =================================
sh eval.sh \
    configs/vitl_336_DLRSD.yaml \
    4 \
    output_vitl_336_DLRSD/eval \
    MODEL.WEIGHTS fx_download/0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth \
    MODEL.SEM_SEG_HEAD.USE_ROTATE True \
    MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP True \
    MODEL.SEM_SEG_HEAD.USE_REMOTE_DINO True \
    MODEL.SEM_SEG_HEAD.NUM_LAYERS 5 \


# =================================
# iSAID CLIP-L
# =================================
sh eval.sh \
    configs/vitl_336_iSAID.yaml \
    4 \
    output_vitl_336_iSAID/eval \
    MODEL.WEIGHTS fx_download/0SAVEoutput_vitl_336_iSAID_rotate_dino_remoteclip_3W_layer2/model_final.pth \
    MODEL.SEM_SEG_HEAD.USE_ROTATE True \
    MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP True \
    MODEL.SEM_SEG_HEAD.USE_REMOTE_DINO True \
    MODEL.SEM_SEG_HEAD.NUM_LAYERS 2 \
