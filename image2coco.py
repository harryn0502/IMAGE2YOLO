import json
import glob
import os
import cv2
from shutil import copy

class Image2Coco:
    def __init__(self, img_ext='jpg', mask_ext='png', area_threshold=100, only_annotations=False, depth_map=False, data_limit=-1):
        self._img_ext = img_ext
        self._mask_ext = mask_ext
        self._area_threshold = area_threshold
        self.only_annotations = only_annotations
        self.depth_map = depth_map
        self.data_limit = data_limit
        self.coco_format = self._get_coco_format()

    def set_area_threshold(self, area_threshold):
        self._area_threshold = area_threshold

    def set_img_ext(self, img_ext):
        self._img_ext = img_ext

    def set_mask_ext(self, mask_ext):
        self._mask_ext = mask_ext

    def set_only_annotations(self, only_annotations):
        self.only_annotations = only_annotations

    def set_depth_map(self, depth_map):
        self.depth_map = depth_map

    def set_data_limit(self, data_limit):
        self.data_limit = data_limit

    def load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as file:
            self.coco_format = json.load(file)

    def _get_max_image_id(self):
        if self.coco_format['images'] == []:
            return -1
        return max([img['id'] for img in self.coco_format['images']])
    
    def _get_max_annotation_id(self):
        if self.coco_format['annotations'] == []:
            return -1
        return max([ann['id'] for ann in self.coco_format['annotations']])
    
    def _get_max_category_id(self):
        if self.coco_format['categories'] == []:
            return -1
        return max([cat['id'] for cat in self.coco_format['categories']])

    def convert(self, image_path, mask_path, out_path, category=""):
        # Create output path folder
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Set categories to the COCO format
        self._add_category_annotation(category)

        # # Get "images" and "annotations" info
        self._add_images_annotations_info(image_path, out_path, mask_path)

    def save(self, out_path):
        # Save the COCO JSON to a file
        with open(os.path.join(out_path, "_annotations.coco.json"), "w") as file:
            json.dump(self.coco_format, file, sort_keys=False, indent=4)

        # Print the number of images copied
        print(f'Copied {self._get_max_image_id() + 1} images in folder: {out_path}')
        # Print the number of annotations created
        print(f'Created {self._get_max_annotation_id() + 1} annotations')

    # Standard COCO format
    def _get_coco_format(self):
        return {
            "info": {},
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": []
        }

    def _add_category_annotation(self, key):
        # Create the category list
        category = {
            "id": self._get_max_category_id() + 1,
            "name": key,
            "supercategory": key,
        }

        # Add the category to the COCO format
        self.coco_format["categories"].append(category)
    
    def _add_images_annotations_info(self, image_path, out_path, mask_path):
        # Get the images and annotations info
        for mask_image in glob.glob(os.path.join(mask_path, f'*.{self._mask_ext}')):
            if self.data_limit != -1 and self._get_max_image_id() + 1 == self.data_limit:
                break
            # Get the original file name and the new file name
            original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{self._img_ext}'
            new_file_name = f'{self._get_max_image_id() + 1}.{self._img_ext}'

            # Copy the images to the COCO images folder
            if not self.only_annotations:
                copy(os.path.join(image_path, original_file_name), os.path.join(out_path, new_file_name))

            # Open the mask image
            mask_image_open = cv2.imread(mask_image)

            # Get the height and width of the mask image
            height, width, _ = mask_image_open.shape

            # Create the images info
            if new_file_name not in map(lambda img: img['file_name'], self.coco_format['images']):
                image = self._add_image_annotation(width, height, new_file_name)
                self.coco_format["images"].append(image)
            else:
                image = [element for element in self.coco_format["images"] if element['file_name'] == new_file_name][0]

            # Find the contours in the mask image
            contours = self._find_contours(mask_image_open)

            for contour in contours:
                annotation = self._add_annotation(image["id"], self._get_max_category_id(), contour)
                if annotation["area"] > self._area_threshold:
                    self.coco_format["annotations"].append(annotation)

    def _add_image_annotation(self, width, height, file_name):
        return {
            "id": self._get_max_image_id() + 1,
            "width": width,
            "height": height,
            "file_name": file_name,
        }
    
    def _find_contours(self, mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.depth_map:
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    def _add_annotation(self, image_id, category_id, contour):
        return {
            "id": self._get_max_annotation_id() + 1,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": cv2.boundingRect(contour),
            "area": cv2.contourArea(contour),
            "segmentation": [contour.flatten().tolist()],
            "iscrowd": 0
        }

if __name__ == '__main__':
    # Define the paths
    data_path = "dataset/"
    coco_path = "coco/"

    paths = ["train", "valid", "test"]
    sub_paths = ["images", "mask/hand"]

    # Create the Image2Coco object
    converter = Image2Coco()

    # Convert the dataset to COCO format
    for path in paths:
        image_path = os.path.join(data_path, path, sub_paths[0])
        mask_path = os.path.join(data_path, path, sub_paths[1])
        out_path = os.path.join(coco_path, path)
        converter.convert(image_path, mask_path, out_path, "hand")
        converter.save(out_path)