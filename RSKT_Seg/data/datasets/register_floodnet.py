import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy
# change unlabled to background

FLOODNET_CLASSES = [
    'Background',
    'building-flooded',
    'building-non-flooded',
    'road-flooded',
    'road-non-flooded',
    'water',
    'tree',
    'vehicle',
    'pool',
    'grass'
]

def _get_landdiscover50k_meta():
    classes = FLOODNET_CLASSES
    ret = {
        "stuff_classes" : classes,
    }
    return ret

def register_floodnet(root):
    # root = os.path.join(root, "FAST")
    # ValidList = os.path.join(root, "valid.txt")
    # root = '/media/zpp2/Datamy/ycy/RSSG/SAMRS/FAST/'
    meta = _get_landdiscover50k_meta()
    # for name, image_dirname, sem_seg_dirname in [
    #     ("test", "images/validation", "annotations_detectron2/validation"),
    # ]:
    root = os.path.join(root,"FloodNet","val+test")
    for name,image_dirname, sem_seg_dirname in [
         ("FloodNet", "img", "lbl"),
     ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = name
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=0, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_floodnet(_root)
