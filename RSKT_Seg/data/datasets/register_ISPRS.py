import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy
# change unlabled to background

CLASSES_ISPRS = ['impervious surface', 'building', 'low vegetation', 'tree','car','clutter']
def _get_landdiscover50k_meta():
    classes = CLASSES_ISPRS
    ret = {
        "stuff_classes" : classes,
    }
    return ret

def register_potsdam(root):
    root = os.path.join(root, "PotsdamSplit")
    meta = _get_landdiscover50k_meta()
    # for name, image_dirname, sem_seg_dirname in [
    #     ("test", "images/validation", "annotations_detectron2/validation"),
    # ]:
    for name,image_dirname, sem_seg_dirname in [
         ("potsdam_all", "img_dir/val", "ann_dir/val"),
     ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = name
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='png'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=5, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_potsdam(_root)
