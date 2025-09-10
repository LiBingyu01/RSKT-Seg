

# DLRSD
python demo/demo.py --config-file configs/vitb_384_DLRSD.yaml \
 --input  PATH2IMG/datasets/DLRSD/imgs/Airplane00.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane01.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane02.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane03.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane04.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane05.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane06.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane07.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane08.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane09.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane10.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane11.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane12.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane13.jpg \
 --output  vis_output\
 --opts MODEL.WEIGHTS  PATH2CKPT/model_final.pth

python demo/demo_visual_gt.py --config-file configs/vitb_384_DLRSD.yaml \
 --input  PATH2IMG/datasets/DLRSD/imgs/Airplane00.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane01.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane02.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane03.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane04.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane05.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane06.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane07.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane08.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane09.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane10.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane11.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane12.jpg \
  PATH2IMG/datasets/DLRSD/imgs/Airplane13.jpg \
 --gt   PATH2IMG/datasets/DLRSD/D2masks/ \
 --output  PATH2CKPT \
