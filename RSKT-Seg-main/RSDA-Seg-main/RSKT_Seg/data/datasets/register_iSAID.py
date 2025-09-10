import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

iSAID_CATEGORIES = [
    {"color": [0, 0, 63], "id": 1, "name": "ship"},
    {"color": [0, 63, 63], "id": 2, "name": "storage tank"},
    {"color": [0, 63, 0], "id": 3, "name": "baseball diamond"},
    {"color": [0, 63, 127], "id": 4, "name": "tennis court"},
    {"color": [0, 63, 191], "id": 5, "name": "basketball court"},
    {"color": [0, 63, 255], "id": 6, "name": "ground track field"},
    {"color": [0, 127, 63], "id": 7, "name": "bridge"},
    {"color": [0, 127, 127], "id": 8, "name": "large vehicle"},
    {"color": [0, 0, 127], "id": 9, "name": "small vehicle"},
    {"color": [0, 0, 191], "id": 10, "name": "helicopter"},
    {"color": [0, 0, 255], "id": 11, "name": "swimming pool"},
    {"color": [0, 191, 127], "id": 12, "name": "roundabout"},
    {"color": [0, 127, 191], "id": 13, "name": "soccer ball field"},
    {"color": [0, 127, 255], "id": 14, "name": "plane"},
    {"color": [0, 100, 155], "id": 15, "name": "harbor"},
    # {"color": [0, 0, 128], "id": 16, "name": "building"},
]

def _get_iSAID_meta():
    stuff_ids = [k["id"] for k in iSAID_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in iSAID_CATEGORIES]
    stuff_colors = [k["color"] for k in iSAID_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_iSAID(root):
    meta = _get_iSAID_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "iSAID_split/train/imgs", "iSAID_split/train/D2masks"),
        ("val", "iSAID_split/val/imgs", "iSAID_split/val/D2masks"),
        ("all", "iSAID/imgs", "iSAID/D2masks"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"iSAID_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_iSAID(root)
