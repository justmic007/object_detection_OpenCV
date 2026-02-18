# Simple Object Detection for individual image
import cv2
from ultralytics import YOLO

model = YOLO(
    "yolov8n.pt"
)  # Load the YOLOv8n model. You can choose other variants like 'yolov8s.pt', 'yolov8m.pt', etc. based on your needs.

image = cv2.imread("./img2.jpg")

results = model(image)  # Perform object detection on the image

annotated_image = results[0].plot()

# Resize image for display
scale_percent = 15  # Resize to 50% of original size
width = int(annotated_image.shape[1] * scale_percent / 100)
height = int(annotated_image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(annotated_image, (width, height))

cv2.imshow("Annotated Image", resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
