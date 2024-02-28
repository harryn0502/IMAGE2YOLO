import os
from image2coco import Image2Coco
from coco2yolo import Coco2Yolo
from shutil import rmtree

class Image2Yolo:
    def __init__(self):
        self.image2coco = Image2Coco()
        self.coco2yolo = Coco2Yolo()

    def convert(self, data_path, yolo_path, copy_images=False):
        # Define the paths
        path_name = os.path.basename(data_path)
        yolo_dir = os.path.dirname(yolo_path)
        temp_dir = os.path.join(yolo_dir, "temp")
        temp_path = os.path.join(temp_dir, path_name)

        # Convert the mask to COCO format into a temporary folder
        self.image2coco.convert(data_path, temp_path, copy_images)

        # Convert the COCO format to YOLO format from the temporary folder
        self.coco2yolo.convert(temp_path, yolo_path, copy_images)

        # Create the data.yaml file
        self.coco2yolo.create_yaml(temp_path, yolo_dir)

        # Remove the temporary files
        if os.path.exists(temp_dir):
            rmtree(temp_dir)
            
if __name__ == "__main__":
    # Define the paths
    data_path = 'dataset/'
    yolo_path = 'yolo/'

    # Create the Image2Yolo object
    converter = Image2Yolo()

    paths = ["train", "valid", "test"]

    # Convert the mask to Yolo format
    for path in paths:
        converter.convert(os.path.join(data_path, path), os.path.join(yolo_path, path), True)