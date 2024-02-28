from image2coco import Image2Coco
from coco2yolo import Coco2Yolo

if __name__ == '__main__':
    # Define the paths
    train_mask_path = 'dataset/train/mask/'
    train_image_path = 'dataset/train/images/'
    valid_mask_path = 'dataset/valid/mask/'
    valid_image_path = 'dataset/valid/images/'

    # Create the Image2Coco object
    conveter = Image2Coco()

    # Convert the mask to COCO format
    conveter.convert(train_mask_path, train_image_path)
    conveter.convert(valid_mask_path, valid_image_path)