# Object Detection with YOLOv8

Simple object detection system using YOLOv8 for analyzing images.

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate object_detection_cv
```

2. The YOLOv8 model files will download in the directory, when you attempt to run the code for the first time eg:
   - `yolov8n.pt` (nano - fastest)
   - `yolov8s.pt` (small)
   - `yolov8m.pt` (medium - default)

## Adding Image Files

To detect objects in images, add your image files to this directory:

1. **Supported formats**: `.jpg`, `.png`, and other common image formats


**Note**: Image files are ignored by git (see `.gitignore`), so they won't be committed to the repository.

## Running Detection

```bash
python simple_object_detection.py
```

The script will:
1. Load an image (`./lady.jpg`)
2. Run YOLOv8 object detection
3. Display the annotated image with detected objects
4. Show the result at 30% scale for easier viewing

## Project Files

- `simple_object_detection.py` - Main detection script
- `environment.yml` - Conda environment configuration
- `yolov8*.pt` - Pre-trained YOLOv8 models
- `.gitignore` - Excludes model files and image files from git tracking
