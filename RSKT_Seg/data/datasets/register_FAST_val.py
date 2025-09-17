import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy
# change unlabled to background

FAST = ['A220','A321','A330','A350','ARJ21','Baseball-Field','Basketball-Court',
'Boeing737','Boeing747','Boeing777','Boeing787','Bridge','Bus','C919','Cargo-Truck',
'Dry-Cargo-Ship','Dump-Truck','Engineering-Ship','Excavator','Fishing-Boat',
'Football-Field','Intersection','Liquid-Cargo-Ship','Motorboat','other-airplane',
'other-ship','other-vehicle','Passenger-Ship','Roundabout','Small-Car','Tennis-Court',
'Tractor','Trailer','Truck-Tractor','Tugboat','Van','Warship']
def _get_landdiscover50k_meta():
    classes = FAST
    ret = {
        "stuff_classes" : classes,
    }
    return ret

def register_FAST(root):
    root = os.path.join(root, "SAMRS")
    root = os.path.join(root, "FAST")
    # ValidList = os.path.join(root, "valid.txt")
    # root = '/media/zpp2/Datamy/ycy/RSSG/SAMRS/FAST/'
    meta = _get_landdiscover50k_meta()
    # for name, image_dirname, sem_seg_dirname in [
    #     ("test", "images/validation", "annotations_detectron2/validation"),
    # ]:
    for name,image_dirname, sem_seg_dirname in [
         ("FAST_val", "val/images", "val/semlabels/gray"),
     ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = name
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='png'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_FAST(_root)
