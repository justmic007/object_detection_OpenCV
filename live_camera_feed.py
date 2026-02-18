import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)  # Open the default camera (0)
model = YOLO("yolov8n.pt")  # Load the YOLOv8n model

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    frame = cv2.flip(frame, 1)  # Flip horizontally (mirror effect)
    results = model(frame)  # Perform object detection on the frame
    annotated_frame = results[0].plot()
    cv2.imshow("Live Camera Feed", annotated_frame)  # Display the annotated frame
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
