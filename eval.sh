# !/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}


# iSAID_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-iSAID_noslide \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/iSAID.json" \
 DATASETS.TEST \(\"iSAID_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#DLRSD_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-DLRSD_noslide \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/DLRSD.json" \
 DATASETS.TEST \(\"DLRSD_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts


#Postdam_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-Potsdam_noslide \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/Potsdam.json" \
 DATASETS.TEST \(\"Potsdam_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Vaihingen_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-Vaihingen_noslide \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/Vaihingen.json" \
 DATASETS.TEST \(\"Vaihingen_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# UDD5_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-UDD5_noslide \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/UDD5.json" \
 DATASETS.TEST \(\"UDD5_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# LoveDA_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-LoveDA_noslide \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/LoveDA.json" \
 DATASETS.TEST \(\"LoveDA_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# uavid_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-uavid_noslide \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/uavid.json" \
 DATASETS.TEST \(\"uavid_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts
 
# VDD_all
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval-VDD_noslide \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/VDD.json" \
 DATASETS.TEST \(\"VDD_all_sem_seg\"\,\) \
 TEST.SLIDING_WINDOW "False" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

cat $output/eval-iSAID_noslide/log.txt | grep copypaste
cat $output/eval-DLRSD_noslide/log.txt | grep copypaste
cat $output/eval-Potsdam_noslide/log.txt | grep copypaste
cat $output/eval-Vaihingen_noslide/log.txt | grep copypaste
cat $output/eval-UDD5_noslide/log.txt | grep copypaste
cat $output/eval-LoveDA_noslide/log.txt | grep copypaste
cat $output/eval-uavid_noslide/log.txt | grep copypaste
cat $output/eval-VDD_noslide/log.txt | grep copypaste
