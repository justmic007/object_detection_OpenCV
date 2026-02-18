import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("./videos/bottles.mp4")

unique_ids = set()  # Set to store unique object IDs

while True:
    ret, frame = cap.read()
    results = model.track(
        frame, classes=[39], persist=True, verbose=False
    )  # Track only 'bottle' class (class ID 39)

    annotated_frame = results[0].plot()

    # Extract unique IDs from the results
    if results[0].boxes and results[0].boxes.id is not None:
        unique_ids.update(results[0].boxes.id.cpu().numpy())

        # if results[0].boxes and results[0].boxes.id is not None:
        #     ids = results[0].boxes.id.numpy()
        #     for oid in ids:
        #         unique_ids.add(oid)

        cv2.putText(
            annotated_frame,
            f"Count: {len(unique_ids)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Object Tracking & Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
