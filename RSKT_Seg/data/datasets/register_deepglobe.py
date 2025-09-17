import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy
# change unlabled to background
CLASSES_LandDiscover50K = [ 'background','bare land','grass','pavement','road','tree','water',
                            'agriculture land','buildings','forest land','barren land','urban land',
                            'large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                            'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                            'ground-track-field', 'small-vehicle', 'baseball-diamond',
                            'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                            'container-crane', 'airport', 'helipad', 'chimney',
                            'expressway service area','expresswalltoll station','dam',
                            'golf field','overpass','stadium','train station',
                            'vehicle','windmill' ]

def _get_landdiscover50k_meta():
    classes = CLASSES_LandDiscover50K
    ret = {
        "stuff_classes" : classes,
    }
    return ret

def register_deepglobe(root):
    root_img = os.path.join(root, "deepglobe")
    root_mask = os.path.join(root, "LandDiscover50K")
    meta = _get_landdiscover50k_meta()
    # for name, image_dirname, sem_seg_dirname in [
    #     ("test", "images/validation", "annotations_detectron2/validation"),
    # ]:
    for name,image_dirname, sem_seg_dirname in [
         ("deepglobe", "TR_Image", "GT_ID"),
     ]:
        image_dir = os.path.join(root_img)
        gt_dir = os.path.join(root_mask, sem_seg_dirname)
        name = name
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='png'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=0, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_deepglobe(_root)
