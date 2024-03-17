import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from utils.image2coco import Image2Coco

# Define the paths relative to the current terminal directory
data_path = "datasets/binary_dataset/"
coco_path = "output/binary_coco/"

paths = ["train", "valid", "test"]
sub_paths = ["images", "mask/hand"]

# Create the Image2Coco object
converter = Image2Coco()

# Convert the dataset to COCO format
for path in paths:
    image_path = os.path.join(data_path, path, sub_paths[0])
    mask_path = os.path.join(data_path, path, sub_paths[1])
    out_path = os.path.join(coco_path, path)
    converter.binary_convert(image_path, mask_path, out_path, "hand")
    converter.save(out_path)