import json
import glob
import os
import cv2

MASK_EXT = 'png'
IMG_EXT = 'jpg'

class ImageToCoco:
    def __init__(self, category_ids):
        self.category_ids = category_ids

    def convert(self, mask_path, json_path):
        # Get COCO format
        coco_format = self._get_coco_format()

        # Set categories to the COCO format
        coco_format["categories"] = self._create_category_annotation()

        # Get "images" and "annotations" info
        coco_format["images"], coco_format["annotations"], count = self._images_annotations_info(mask_path)

        # Save the COCO JSON to a file
        with open(json_path, "w") as file:
            json.dump(coco_format, file, sort_keys=False, indent=4)

        # Print the number of annotations created
        print(f'Created {count} annotations for images in folder: {mask_path}')

    # Standard COCO format
    def _get_coco_format(self):
        return {
            "info": {},
            "licenses": [],
            "images": [{}],
            "categories": [{}],
            "annotations": [{}]
        }


    def _create_category_annotation(self):
        category_list = []

        # Create the category list
        for key, value in self.category_ids.items():
            category = {
                "supercategory": key,
                "id": value,
                "name": key
            }
            category_list.append(category)

        return category_list
    
    def _images_annotations_info(self, mask_path):
        image_id = 0
        annotation_id = 0
        annotations = []
        images = []

        # Get the images and annotations info
        for category in self.category_ids.keys():
            for mask_image in glob.glob(os.path.join(mask_path, category, f'*.{MASK_EXT}')):
                original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{IMG_EXT}'
                mask_image_open = cv2.imread(mask_image)

                # Get the height and width of the mask image
                height, width, _ = mask_image_open.shape

                # Create the images info
                if original_file_name not in map(lambda img: img['file_name'], images):
                    image = self._create_image_annotation(image_id, width, height, original_file_name)
                    images.append(image)
                    image_id += 1
                else:
                    image = [element for element in images if element['file_name'] == original_file_name][0]

                # Find the contours in the mask image
                contours = self._find_contours(mask_image_open)

                for contour in contours:
                    annotation = self._create_annotation_format(annotation_id, image["id"], self.category_ids[category], contour)
                    if annotation["area"] > 0:
                        annotations.append(annotation)
                        annotation_id += 1

        return images, annotations, annotation_id

    def _create_image_annotation(self, image_id, width, height, file_name):
        return {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
        }
    
    def _find_contours(self, mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    def _create_annotation_format(self, annotation_id, image_id, category_id, contour):
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [contour.flatten().tolist()],
            "area": cv2.contourArea(contour),
            "bbox": cv2.boundingRect(contour),
            "iscrowd": 0
        }

if __name__ == '__main__':
    # Define the paths
    train_mask_path = 'dataset/train/mask/'
    train_json_path = 'dataset/train.json'
    valid_mask_path = 'dataset/valid/mask/'
    valid_json_path = 'dataset/valid.json'

    # Define the category ids
    category_ids = {
        "hand": 0,
    }

    # Create the ImageToCoco object
    conveter = ImageToCoco(category_ids)

    # Convert the mask to COCO format
    conveter.convert(train_mask_path, train_json_path)
    conveter.convert(valid_mask_path, valid_json_path)
