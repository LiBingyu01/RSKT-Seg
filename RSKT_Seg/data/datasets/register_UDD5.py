import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

UDD5_CATEGORIES = [
    {"color": [107,142,35], "id": 1, "name": "Vegetation"},
    {"color": [102,102,156], "id": 2, "name": "Building"},
    {"color": [128,64,128], "id": 3, "name": "Road"},
    {"color": [0,0,142], "id": 4, "name": "Vehicle"},
    {"color": [0,0,0], "id": 5, "name": "Other"},
]

def _get_UDD5_meta():
    stuff_ids = [k["id"] for k in UDD5_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in UDD5_CATEGORIES]
    stuff_colors = [k["color"] for k in UDD5_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_UDD5(root):
    meta = _get_UDD5_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "UDD5/all/src", "UDD5/all/gt"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"UDD5_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="JPG")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "datasets"
register_UDD5(root)
