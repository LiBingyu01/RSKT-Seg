

# iSAID
python demo/demo.py --config-file configs/vitb_384_iSAID.yaml \
 --input  PATH/datasets/iSAID/imgs/P0002_1030_1286_1442_1698.png \
  PATH/datasets/iSAID/imgs/P0002_1030_1286_1648_1904.png \
  PATH/datasets/iSAID/imgs/P0002_1030_1286_1854_2110.png \
  PATH/datasets/iSAID/imgs/P0002_1030_1286_2266_2522.png \
  PATH/datasets/iSAID/imgs/P0002_1030_1286_2301_2557.png \
  PATH/datasets/iSAID/imgs/P0002_1442_1698_2266_2522.png \
  PATH/datasets/iSAID/imgs/P0002_1442_1698_2301_2557.png \
  PATH/datasets/iSAID/imgs/P0002_1648_1904_2266_2522.png \
  PATH/datasets/iSAID/imgs/P0002_1648_1904_2301_2557.png \
  PATH/datasets/iSAID/imgs/P0003_206_462_206_462.png \
  PATH/datasets/iSAID/imgs/P0003_206_462_412_668.png \
  PATH/datasets/iSAID/imgs/P0003_412_668_206_462.png \
  PATH/datasets/iSAID/imgs/P0003_412_668_412_668.png \
  PATH/datasets/iSAID/imgs/P0003_767_1023_618_874.png \
 --output  PATH/vis_convnextB_768_trainiSAID_iSAID\
 --opts MODEL.WEIGHTS  PATH2CKPT/model_final.pth

python demo/demo_visual_gt.py --config-file configs/vitb_384_iSAID.yaml \
 --input  PATH/datasets/iSAID/imgs/P0000_1648_1904_3090_3346.png \
 --input  PATH/datasets/iSAID/imgs/P0002_1030_1286_1442_1698.png \
  PATH/datasets/iSAID/imgs/P0002_1030_1286_1648_1904.png \
  PATH/datasets/iSAID/imgs/P0002_1030_1286_1854_2110.png \
  PATH/datasets/iSAID/imgs/P0002_1030_1286_2266_2522.png \
  PATH/datasets/iSAID/imgs/P0002_1030_1286_2301_2557.png \
  PATH/datasets/iSAID/imgs/P0002_1442_1698_2266_2522.png \
  PATH/datasets/iSAID/imgs/P0002_1442_1698_2301_2557.png \
  PATH/datasets/iSAID/imgs/P0002_1648_1904_2266_2522.png \
  PATH/datasets/iSAID/imgs/P0002_1648_1904_2301_2557.png \
  PATH/datasets/iSAID/imgs/P0003_206_462_206_462.png \
  PATH/datasets/iSAID/imgs/P0003_206_462_412_668.png \
  PATH/datasets/iSAID/imgs/P0003_412_668_206_462.png \
  PATH/datasets/iSAID/imgs/P0003_412_668_412_668.png \
  PATH/datasets/iSAID/imgs/P0003_767_1023_618_874.png \
 --gt   PATH/datasets/iSAID/D2masks \
 --output  output