import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

DLRSD_CATEGORIES = [
    {"color": [166, 202, 240], "id": 1, "name": "airplane"},
    {"color": [128, 128, 0], "id": 2, "name": "bare soil"},
    {"color": [0, 0, 128], "id": 3, "name": "buildings"},
    {"color": [255, 0, 0], "id": 4, "name": "cars"},
    {"color": [0, 128, 0], "id": 5, "name": "chaparral"},
    {"color": [128, 0, 0], "id": 6, "name": "court"},
    {"color": [255, 233, 233], "id": 7, "name": "dock"},
    {"color": [160, 160, 164], "id": 8, "name": "field"},
    {"color": [0, 128, 128], "id": 9, "name": "grass"},
    {"color": [90, 87, 255], "id": 10, "name": "mobile home"},
    {"color": [255, 255, 0], "id": 11, "name": "pavement"},
    {"color": [255, 192, 0], "id": 12, "name": "sand"},
    {"color": [0, 0, 255], "id": 13, "name": "sea"},
    {"color": [255, 0, 192], "id": 14, "name": "ship"},
    {"color": [128, 0, 128], "id": 15, "name": "tanks"},
    {"color": [0, 255, 0], "id": 16, "name": "trees"},
    {"color": [0, 255, 255], "id": 17, "name": "water"},
]

def _get_DLRSD_meta():
    stuff_ids = [k["id"] for k in DLRSD_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in DLRSD_CATEGORIES]
    stuff_colors = [k["color"] for k in DLRSD_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_DLRSD(root):
    meta = _get_DLRSD_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "DLRSD_split/train/imgs", "DLRSD_split/train/D2masks"),
        ("val", "DLRSD_split/val/imgs", "DLRSD_split/val/D2masks"),
        ("all", "DLRSD/imgs", "DLRSD/D2masks"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"DLRSD_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_DLRSD(root)
