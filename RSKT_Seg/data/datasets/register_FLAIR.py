import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy
# change unlabled to background

FLAIR_CLASSES = ["building","pervious surface","impervious surface","bare soil",
                 "water","coniferous","deciduous","brushwood","vineyard",
                 "herbaceous vegetation","agricultural land","plowed land","other"]

def _get_landdiscover50k_meta():
    classes = FLAIR_CLASSES
    ret = {
        "stuff_classes" : classes,
    }
    return ret

def register_flair(root):
    # root = os.path.join(root, "FAST")
    # ValidList = os.path.join(root, "valid.txt")
    # root = '/media/zpp2/Datamy/ycy/RSSG/SAMRS/FAST/'
    meta = _get_landdiscover50k_meta()
    # for name, image_dirname, sem_seg_dirname in [
    #     ("test", "images/validation", "annotations_detectron2/validation"),
    # ]:
    root = os.path.join(root,"FLAIR")
    for name,image_dirname, sem_seg_dirname in [
         ("FLAIR_test", "test/image", "test/mask"),
     ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = name
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='png'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=12, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_flair(_root)
