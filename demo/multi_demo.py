import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from utils.image2coco import Image2Coco

# Define the paths
data_path = "datasets/multi_dataset/"
coco_path = "output/multi_coco/"

category_ids = {
    "left": 0,
    "right": 1,
}

category_colours = {
    "(0, 0, 255)": 0,
    "(0, 255, 0)": 1
}

paths = ["train"]
sub_paths = ["images", "mask"]

# Create the Image2Coco object
converter = Image2Coco()

# Convert the dataset to COCO format
for path in paths:
    image_path = os.path.join(data_path, path, sub_paths[0])
    mask_path = os.path.join(data_path, path, sub_paths[1])
    out_path = os.path.join(coco_path, path)
    converter.multi_convert(image_path, mask_path, out_path, category_ids, category_colours)
    converter.save(out_path)