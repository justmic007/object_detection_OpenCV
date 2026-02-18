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

## Adding Video Files

To detect objects in videos, add your video files to the `videos/` directory:

1. Create the videos folder: `mkdir -p videos`
2. Download free videos from [Pexels](https://www.pexels.com/videos/) or other sources
3. **Supported formats**: `.mp4`, `.avi`, `.mov`, and other common video formats
4. Update the video path in `multi_object_from_video.py` if needed

**Note**: Video files are ignored by git (see `.gitignore`), so they won't be committed to the repository.

## Running Detection

### Static Image Detection
```bash
python simple_object_detection.py
```

The script will:
1. Load an image (`./img1.jpg`)
2. Run YOLOv8 object detection
3. Display the annotated image with detected objects
4. Show the result at 30% scale for easier viewing

### Live Camera Feed Detection
```bash
python live_camera_feed.py
```

The script will:
1. Open your default camera (webcam)
2. Run real-time YOLOv8 object detection on the video feed
3. Display the annotated video with detected objects
4. Mirror the camera feed for a selfie-like view
5. Press 'q' to quit

### Video File Detection
```bash
python multi_object_from_video.py
```

The script will:
1. Load a video file (`./videos/street1.mp4`)
2. Run YOLOv8 object detection on each frame
3. Display the annotated video with detected objects
4. Show the result at 50% scale for easier viewing
5. Press 'q' to quit

### Object Tracking & Counting
```bash
python object_counting.py
```

The script will:
1. Load a video file (`./videos/bottles.mp4`)
2. Track objects across frames using unique IDs
3. Count total unique objects detected throughout the video
4. Display the count on screen
5. Press 'q' to quit

## Project Files

- `simple_object_detection.py` - Static image detection script
- `live_camera_feed.py` - Real-time camera detection script
- `multi_object_from_video.py` - Video file detection script
- `object_counting.py` - Object tracking and counting script
- `environment.yml` - Conda environment configuration
- `yolov8*.pt` - Pre-trained YOLOv8 models
- `.gitignore` - Excludes model files and image files from git tracking

## COCO Classes

YOLOv8 is trained on 80 COCO classes:

**People:** 0=person

**Vehicles:** 1=bicycle, 2=car, 3=motorcycle, 4=airplane, 5=bus, 6=train, 7=truck, 8=boat

**Animals:** 14=bird, 15=cat, 16=dog, 17=horse, 18=sheep, 19=cow, 20=elephant, 21=bear, 22=zebra, 23=giraffe

**Outdoor:** 9=traffic light, 10=fire hydrant, 11=stop sign, 12=parking meter, 13=bench

**Sports:** 32=sports ball, 33=baseball bat, 34=baseball glove, 35=skateboard, 36=surfboard, 37=tennis racket

**Kitchen:** 39=bottle, 40=wine glass, 41=cup, 42=fork, 43=knife, 44=spoon, 45=bowl

**Food:** 46=banana, 47=apple, 48=sandwich, 49=orange, 50=broccoli, 51=carrot, 52=hot dog, 53=pizza, 54=donut, 55=cake

**Furniture:** 56=chair, 57=couch, 58=potted plant, 59=bed, 60=dining table, 61=toilet

**Electronics:** 62=tv, 63=laptop, 64=mouse, 65=remote, 66=keyboard, 67=cell phone

**Appliances:** 68=microwave, 69=oven, 70=toaster, 71=sink, 72=refrigerator

**Indoor:** 73=book, 74=clock, 75=vase, 76=scissors, 77=teddy bear, 78=hair drier, 79=toothbrush

To filter specific classes, modify the `classes` parameter in the scripts. Remove it to detect all objects.
