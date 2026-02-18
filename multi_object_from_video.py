import cv2
from ultralytics import YOLO

model = YOLO(
    "yolov8n.pt"
)  # Load the YOLOv8n model. You can choose other variants like 'yolov8s.pt', 'yolov8m.pt', etc. based on your needs.

cap = cv2.VideoCapture("./videos/street1.mp4")  # Open the default camera (0)

while True:
    ret, frame = cap.read()  # Read a frame from the video

    results = model.track(
        frame,
        classes=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ],  # Filter for all vehicles: 1=bicycle, 2=car, 3=motorcycle, 4=airplane, 5=bus, 6=train, 7=truck, 8=boat
        persist=True,  # Persist tracks between frames
    )  # Perform object detection and tracking on the frame

    annotated_frame = results[0].plot()  # Get the annotated frame with detections

    # Resize frame for display
    scale_percent = 50  # Resize to 15% of original size
    width = int(annotated_frame.shape[1] * scale_percent / 100)
    height = int(annotated_frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(annotated_frame, (width, height))

    cv2.imshow("Multi Object Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
