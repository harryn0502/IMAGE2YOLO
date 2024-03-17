import os
import json
import yaml
from ultralytics.data.converter import convert_coco
from shutil import copy2, copytree, ignore_patterns

class Coco2Yolo:
    def convert(self, coco_path, yolo_path, copy_images=False):
        # Convert COCO to YOLO
        image_path = os.path.join(coco_path)
        convert_coco(image_path, yolo_path, use_segments=True, cls91to80=False)

        if copy_images:
            copytree(coco_path, os.path.join(yolo_path, 'images'), copy_function=copy2, ignore=ignore_patterns("*.json"),dirs_exist_ok=True) if copy_images else None

    def create_yaml(self, coco_path, yolo_path):
        with open(os.path.join(coco_path, '_annotations.coco.json')) as f:
            data = json.load(f)
        
        # Extract the category names
        names = [category['name'] for category in data['categories']]
        
        # Number of classes
        nc = len(names)

        # Create a dictionary with the required content
        yaml_data = {
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': nc,
            'names': names,
        }

        # Write the dictionary to a YAML file
        with open(os.path.join(yolo_path, 'data.yaml'), 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)

if __name__ == '__main__':
    # Define the paths
    coco_path = 'coco/'
    yolo_path = 'yolo/'

    # Create the Coco2Yolo object
    converter = Coco2Yolo()

    paths = ["train", "valid", "test"]

    # Convert the COCO format to YOLO format
    for path in paths:
        converter.convert(os.path.join(coco_path, path), os.path.join(yolo_path, path), True)

    # Create the data.yaml file
    converter.create_yaml(os.path.join(coco_path, "train"), yolo_path)