# IMAGE2YOLO Converter

This is a simple tool written in Python to convert mask images to YOLO format.

## Pre-requisites
- ultralytics

## Dataset Directory Structures

### Mask Image Dataset Directory Structure
```
./
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   │   ├── 0000.jpg
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   ├── ...
│   │   ├── masks/
│   │   │   ├── hand/
│   │   │   │   ├── 0000.png
│   │   │   │   ├── 0001.png
│   │   │   │   ├── 0002.png
│   │   │   │   ├── ...
│   ├── valid/
│   │   ├── images/
│   │   │   ├── 0003.jpg
│   │   │   ├── 0004.jpg
│   │   │   ├── 0005.jpg
│   │   │   ├── ...
│   │   ├── masks/
│   │   │   ├── hand/
│   │   │   │   ├── 0003.png
│   │   │   │   ├── 0004.png
│   │   │   │   ├── 0005.png
│   │   │   │   ├── ...
│   ├── test/
│   │   ├── images/
│   │   │   ├── 0006.jpg
│   │   │   ├── 0007.jpg
│   │   │   ├── 0008.jpg
│   │   │   ├── ...
│   │   ├── masks/
│   │   │   ├── hand/
│   │   │   │   ├── 0006.png
│   │   │   │   ├── 0007.png
│   │   │   │   ├── 0008.png
│   │   │   │   ├── ...
```

### COCO Dataset Directory Structure
```
./
├── coco/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   ├── ...
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   ├── 0003.jpg
│   │   ├── 0004.jpg
│   │   ├── 0005.jpg
│   │   ├── ...
│   ├── test/
│   │   ├── _annotations.coco.json
│   │   ├── 0006.jpg
│   │   ├── 0007.jpg
│   │   ├── 0008.jpg
│   │   ├── ...
```

### Yolo Dataset Directory Structure
```
./
├── yolo/
│   ├── train/
│   │   ├── images/
│   │   │   ├── 0000.jpg
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   ├── ...
│   │   ├── labels/
│   │   │   ├── _annotations.coco/
│   │   │   │   ├── 0000.txt
│   │   │   │   ├── 0001.txt
│   │   │   │   ├── 0002.txt
│   │   │   │   ├── ...
│   ├── valid/
│   │   ├── images/
│   │   │   ├── 0003.jpg
│   │   │   ├── 0004.jpg
│   │   │   ├── 0005.jpg
│   │   │   ├── ...
│   │   ├── labels/
│   │   │   ├── _annotations.coco/
│   │   │   │   ├── 0003.txt
│   │   │   │   ├── 0004.txt
│   │   │   │   ├── 0005.txt
│   │   │   │   ├── ...
│   ├── test/
│   │   ├── images/
│   │   │   ├── 0006.jpg
│   │   │   ├── 0007.jpg
│   │   │   ├── 0008.jpg
│   │   │   ├── ...
│   │   ├── labels/
│   │   │   ├── _annotations.coco/
│   │   │   │   ├── 0006.txt
│   │   │   │   ├── 0007.txt
│   │   │   │   ├── 0008.txt
│   │   │   │   ├── ...
```

## COCO to YOLO Format
```bash
python coco2yolo.py
```

## Mask Image to YOLO Format
```bash
python image2yolo.py
```