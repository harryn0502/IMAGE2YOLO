import json
import glob
import os
import cv2
from shutil import copy

class Image2Coco:
    def __init__(self, img_ext='jpg', mask_ext='png', area_threshold=100):
        self._img_ext = img_ext
        self._mask_ext = mask_ext
        self._area_threshold = area_threshold
        self._category_ids = {}
        self._image_id = 0

    def set_area_threshold(self, area_threshold):
        self._area_threshold = area_threshold

    def set_img_ext(self, img_ext):
        self._img_ext = img_ext

    def set_mask_ext(self, mask_ext):
        self._mask_ext = mask_ext


    def convert(self, image_path, mask_path, coco_path, copy_images=False, depth_map=False):
        coco_images_path = coco_path
        if image_path == coco_path:
            coco_images_path = os.path.join(coco_path, 'images')

        # Create coco image folder
        if not os.path.exists(coco_images_path):
            os.makedirs(coco_images_path)

        # Get COCO format
        coco_format = self._get_coco_format()

        # Create the category_ids
        self._category_ids = self._create_category_ids(mask_path)

        # Set categories to the COCO format
        coco_format["categories"] = self._create_category_annotation()

        # Get "images" and "annotations" info
        coco_format["images"], coco_format["annotations"], count = self._images_annotations_info(image_path, coco_images_path, copy_images, mask_path, depth_map)

        # Save the COCO JSON to a file
        with open(os.path.join(coco_images_path, "_annotations.coco.json"), "w") as file:
            json.dump(coco_format, file, sort_keys=False, indent=4)

        # Print the number of images copied
        print(f'Copied {count} images in folder: {coco_images_path}')
        # Print the number of annotations created
        print(f'Created {count} annotations for images in folder: {mask_path}')

    # Standard COCO format
    def _get_coco_format(self):
        return {
            "info": {},
            "licenses": [],
            "categories": [{}],
            "images": [{}],
            "annotations": [{}]
        }
    
    def _create_category_ids(self, mask_path):
        category_ids = {}
        for i, category in enumerate(glob.glob(os.path.join(mask_path, '*'))):
            category_ids[os.path.basename(category)] = i + 1
        return category_ids

    def _create_category_annotation(self):
        category_list = []

        # Create the category list
        for key, value in self._category_ids.items():
            category = {
                "id": value,
                "name": key,
                "supercategory": key,
            }
            category_list.append(category)

        return category_list
    
    def _images_annotations_info(self, image_path, coco_images_path, copy_images, mask_path, depth_map=False):
        annotation_id = 0
        annotations = []
        images = []

        # Get the images and annotations info
        for category in self._category_ids.keys():
            for mask_image in glob.glob(os.path.join(mask_path, category, f'*.{self._mask_ext}')):
                # Get the original file name and the new file name
                original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{self._img_ext}'
                new_file_name = f'{self._image_id}.{self._img_ext}'

                # Copy the images to the COCO images folder
                if image_path != coco_path and copy_images:
                    copy(os.path.join(image_path, original_file_name), os.path.join(coco_images_path, new_file_name)) if copy_images else None

                # Open the mask image
                mask_image_open = cv2.imread(mask_image)

                # Get the height and width of the mask image
                height, width, _ = mask_image_open.shape

                # Create the images info
                if self._image_id not in map(lambda img: img['file_name'], images):
                    image = self._create_image_annotation(self._image_id, width, height, new_file_name)
                    images.append(image)
                    self._image_id += 1
                else:
                    image = [element for element in images if element['file_name'] == self._image_id][0]

                # Find the contours in the mask image
                contours = self._find_contours(mask_image_open, depth_map)

                for contour in contours:
                    annotation = self._create_annotation_format(annotation_id, image["id"], self._category_ids[category], contour)
                    if annotation["area"] > self._area_threshold:
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
    
    def _find_contours(self, mask, depth_map=False):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if depth_map:
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    def _create_annotation_format(self, annotation_id, image_id, category_id, contour):
        return {
            "id": annotation_id,
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
    sub_paths = ["images", "mask"]

    # Create the Image2Coco object
    converter = Image2Coco()

    # Convert the dataset to COCO format
    for path in paths:
        image_path = os.path.join(data_path, path, sub_paths[0])
        mask_path = os.path.join(data_path, path, sub_paths[1])
        out_path = os.path.join(coco_path, path)
        converter.convert(image_path, mask_path, out_path, True)