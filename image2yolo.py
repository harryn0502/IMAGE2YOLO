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
        data_train_path = os.path.join(data_path, 'train')
        data_valid_path = os.path.join(data_path, 'valid')
        yolo_train_path = os.path.join(yolo_path, 'train')
        yolo_valid_path = os.path.join(yolo_path, 'valid')
        temp_path = os.path.join(yolo_path, 'temp')
        temp_train_path = os.path.join(temp_path, 'train')
        temp_valid_path = os.path.join(temp_path, 'valid')

        # Create the Image2Coco object
        conveter = Image2Coco()

        # Convert the mask to COCO format into a temporary folder
        conveter.convert(data_train_path, temp_train_path, copy_images)
        conveter.convert(data_valid_path, temp_valid_path, copy_images)

        # Create the Coco2Yolo object
        conveter = Coco2Yolo()

        # Convert the COCO format to YOLO format from the temporary folder
        conveter.convert(temp_train_path, yolo_train_path, copy_images)
        conveter.convert(temp_valid_path, yolo_valid_path, copy_images)

        # Create the data.yaml file
        conveter.create_yaml(temp_train_path, yolo_path)

        # Remove the temporary files
        if os.path.exists(temp_path):
            rmtree(temp_path)
            
if __name__ == "__main__":
    # Define the paths
    data_path = 'dataset/'
    yolo_path = 'yolo/'

    # Create the Image2Yolo object
    conveter = Image2Yolo()

    # Convert the mask to Yolo format
    conveter.convert(data_path, yolo_path, True)