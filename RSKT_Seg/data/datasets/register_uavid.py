import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

uavid_CATEGORIES = [
    {"color": [0, 0, 0], "id": 1, "name": "Background clutter"},
    {"color": [128, 0, 0], "id": 2, "name": "Building"},
    {"color": [128,64,128], "id": 3, "name": "Road"},
    {"color": [0,128,0], "id": 4, "name": "Tree"},
    {"color": [128,128,0], "id": 5, "name": "Low vegetation"},
    {"color": [64,0,128], "id": 6, "name": "Moving car"},
    {"color": [192,0,192], "id": 7, "name": "Static car"},
    {"color": [64,64,0], "id": 8, "name": "Human"}
]

def _get_uavid_meta():
    stuff_ids = [k["id"] for k in uavid_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in uavid_CATEGORIES]
    stuff_colors = [k["color"] for k in uavid_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_uavid(root):
    meta = _get_uavid_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "uavid/Images", "uavid/Labels"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"uavid_{name}_sem_seg"
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

root = "datasets"
register_uavid(root)
