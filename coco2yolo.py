import os
import json
import yaml
from ultralytics.data.converter import convert_coco

class Coco2Yolo:
    def convert(self, coco_path, yolo_path):
        # Convert COCO to YOLO
        image_path = os.path.join(coco_path)
        convert_coco(image_path, yolo_path, use_segments=True, cls91to80=False)

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
    train_path = 'coco/train/'
    valid_path = 'coco/valid/'
    yolo_path = 'yolo/'
    yolo_train_path = yolo_path + 'train/'
    yolo_valid_path = yolo_path + 'valid/'

    # Create the Coco2Yolo object
    conveter = Coco2Yolo()

    # Convert the COCO format to YOLO format
    conveter.convert(train_path, yolo_train_path)
    conveter.convert(valid_path, yolo_valid_path)

    # Create the data.yaml file
    conveter.create_yaml(train_path, yolo_path)