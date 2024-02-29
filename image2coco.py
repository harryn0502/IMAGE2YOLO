import json
import glob
import os
import cv2

from shutil import copytree, copy2
class Image2Coco:
    def __init__(self, img_ext='jpg', mask_ext='png'):
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.category_ids = {}

    def convert(self, data_path, coco_path, copy_images=False, depth_map=False):
        mask_path = os.path.join(data_path, 'mask')
        coco_images_path = coco_path
        if data_path == coco_path:
            coco_images_path = os.path.join(coco_path, 'images')
        # Get COCO format
        coco_format = self._get_coco_format()

        # Create the category_ids
        self.category_ids = self._create_category_ids(mask_path)

        # Set categories to the COCO format
        coco_format["categories"] = self._create_category_annotation()

        # Get "images" and "annotations" info
        coco_format["images"], coco_format["annotations"], count = self._images_annotations_info(mask_path, depth_map)

        # Create coco image folder
        if not os.path.exists(coco_images_path):
            os.makedirs(coco_images_path)
        
        if data_path != coco_path and copy_images:
            copytree(os.path.join(data_path, 'images'), coco_images_path, copy_function=copy2, dirs_exist_ok=True) if copy_images else None

        # Save the COCO JSON to a file
        with open(os.path.join(coco_images_path, "_annotations.coco.json"), "w") as file:
            json.dump(coco_format, file, sort_keys=False, indent=4)

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
        for key, value in self.category_ids.items():
            category = {
                "id": value,
                "name": key,
                "supercategory": key,
            }
            category_list.append(category)

        return category_list
    
    def _images_annotations_info(self, mask_path, depth_map=False):
        image_id = 0
        annotation_id = 0
        annotations = []
        images = []

        # Get the images and annotations info
        for category in self.category_ids.keys():
            for mask_image in glob.glob(os.path.join(mask_path, category, f'*.{self.mask_ext}')):
                original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{self.img_ext}'
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
                contours = self._find_contours(mask_image_open, depth_map)

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

    # Create the Image2Coco object
    converter = Image2Coco()

    for path in paths:
        converter.convert(os.path.join(data_path, path), os.path.join(coco_path, path), True)