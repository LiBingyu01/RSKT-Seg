import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

LoveDA_CATEGORIES = [
    {"color": [255, 255, 255], "id": 1, "name": "no-data"},
    {"color": [255, 248, 220], "id": 2, "name": "Background"},
    {"color": [100, 149, 237], "id": 3, "name": "Building"},
    {"color": [102, 205, 170], "id": 4, "name": "Road"},
    {"color": [205, 133, 63], "id": 5, "name": "Water"},
    {"color": [160, 32, 240], "id": 6, "name": "Barren"},
    {"color": [255, 64, 64], "id": 7, "name": "Forest"},
    {"color": [139, 69, 19], "id": 8, "name": "Agriculture"}
]

def _get_LoveDA_meta():
    stuff_ids = [k["id"] for k in LoveDA_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in LoveDA_CATEGORIES]
    stuff_colors = [k["color"] for k in LoveDA_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_LoveDA(root):
    meta = _get_LoveDA_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "LoveDA/images_png", "LoveDA/masks_png"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"LoveDA_{name}_sem_seg"
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
register_LoveDA(root)
