import os
from image2coco import Image2Coco
from coco2yolo import Coco2Yolo
from shutil import rmtree

if __name__ == '__main__':
    # Define the paths
    data_train_path = 'dataset/train/'
    data_valid_path = 'dataset/valid/'
    yolo_path = 'yolo/'
    yolo_train_path = yolo_path + 'train/'
    yolo_valid_path = yolo_path + 'valid/'

    temp_path = yolo_path + 'temp/'
    temp_train_path = temp_path + 'train/'
    temp_valid_path = temp_path + 'valid/'

    # Create the Image2Coco object
    conveter = Image2Coco()

    # Convert the mask to COCO format into a temporary folder
    conveter.convert(data_train_path, temp_train_path, True)
    conveter.convert(data_valid_path, temp_valid_path, True)

    # Create the Coco2Yolo object
    conveter = Coco2Yolo()

    # Convert the COCO format to YOLO format from the temporary folder
    conveter.convert(temp_train_path, yolo_train_path, True)
    conveter.convert(temp_valid_path, yolo_valid_path, True)

    # Create the data.yaml file
    conveter.create_yaml(temp_train_path, yolo_path)

    # Remove the temporary files
    if os.path.exists(temp_path):
        rmtree(temp_path)