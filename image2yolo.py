from image2coco import ImageToCoco
from ultralytics.data.converter import convert_coco

import json, yaml

from shutil import copytree, copy2, ignore_patterns

def create_yaml(input_json_path, output_yaml_path, test_path=None):
    with open(input_json_path) as f:
        data = json.load(f)
    
    # Extract the category names
    names = [category['name'] for category in data['categories']]
    
    # Number of classes
    nc = len(names)

    # Create a dictionary with the required content
    yaml_data = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': test_path if test_path else '',
        'nc': nc,
        'names': names,
    }

    # Write the dictionary to a YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)

# Define the category ids
category_ids = {
    "hand": 1,
}

# Define the paths
train_mask_path = 'dataset/train/mask/'
train_image_path = 'dataset/train/images/'
train_json_path = train_image_path + '_annotations.coco.json'
valid_mask_path = 'dataset/valid/mask/'
valid_image_path = 'dataset/valid/images/'
valid_json_path = valid_image_path + '_annotations.coco.json'

# Create the ImageToCoco object
conveter = ImageToCoco(category_ids)

# Convert the mask to COCO format
conveter.convert(train_mask_path, train_json_path)
conveter.convert(valid_mask_path, valid_json_path)

# Convert Image mask to COCO to YOLO
convert_coco(train_image_path, 'coco_converted/train/', use_segments=True, cls91to80=False)
convert_coco(valid_image_path, 'coco_converted/valid/', use_segments=True, cls91to80=False)

# Create data.yaml file
create_yaml(train_json_path, 'coco_converted/data.yaml')

# Copy images to the new YOLO format
copytree(train_image_path, 'coco_converted/train/images', dirs_exist_ok=True, copy_function=copy2, ignore=ignore_patterns('*.json'))
copytree(valid_image_path, 'coco_converted/valid/images', dirs_exist_ok=True, copy_function=copy2, ignore=ignore_patterns('*.json'))