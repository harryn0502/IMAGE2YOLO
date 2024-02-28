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
│   │   │   ├── ...
│   │   ├── masks/
│   │   │   ├── hand/
│   │   │   │   ├── 0000.png
│   │   │   │   ├── 0001.png
│   │   │   │   ├── ...
│   ├── valid/
│   │   ├── images/
│   │   │   ├── 0002.jpg
│   │   │   ├── 0003.jpg
│   │   │   ├── ...
│   │   ├── masks/
│   │   │   ├── hand/
│   │   │   │   ├── 0002.png
│   │   │   │   ├── 0003.png
│   │   │   │   ├── ...
│   ├── test/
│   │   ├── images/
│   │   │   ├── 0004.jpg
│   │   │   ├── 0005.jpg
│   │   │   ├── ...
│   │   ├── masks/
│   │   │   ├── hand/
│   │   │   │   ├── 0004.png
│   │   │   │   ├── 0005.png
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
│   │   ├── ...
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   ├── 0002.jpg
│   │   ├── 0003.jpg
│   │   ├── ...
│   ├── test/
│   │   ├── _annotations.coco.json
│   │   ├── 0004.jpg
│   │   ├── 0005.jpg
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
│   │   │   ├── ...
│   │   ├── labels/
│   │   │   ├── _annotations.coco/
│   │   │   │   ├── 0000.txt
│   │   │   │   ├── 0001.txt
│   │   │   │   ├── ...
│   ├── valid/
│   │   ├── images/
│   │   │   ├── 0002.jpg
│   │   │   ├── 0003.jpg
│   │   │   ├── ...
│   │   ├── labels/
│   │   │   ├── _annotations.coco/
│   │   │   │   ├── 0002.txt
│   │   │   │   ├── 0003.txt
│   │   │   │   ├── ...
│   ├── test/
│   │   ├── images/
│   │   │   ├── 0004.jpg
│   │   │   ├── 0005.jpg
│   │   │   ├── ...
│   │   ├── labels/
│   │   │   ├── _annotations.coco/
│   │   │   │   ├── 0004.txt
│   │   │   │   ├── 0005.txt
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