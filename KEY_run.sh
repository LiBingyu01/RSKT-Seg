# =================================
# .sh file to train your own RSKT-Seg
# =================================

# =================================
# python train_net.py --config $config \
#  --num-gpus $gpus \
#  --dist-url "auto" \
#  --resume \
#  OUTPUT_DIR $output \
#  $opts
# sh eval.sh $config $gpus $output $opts
# =================================

# =================================
# DLRSD 
# =================================
sh run.sh configs/vitl_336_DLRSD.yaml 4 output_vitl_336_DLRSD/
sh run.sh configs/vitb_384_DLRSD.yaml 4 output_vitb_384_DLRSD/


# =================================
# iSAID
# =================================
sh run.sh configs/vitl_336_iSAID.yaml 4 output_vitl_336_iSAID/
sh run.sh configs/vitb_384_iSAID.yaml 4 output_vitb_384_iSAID/