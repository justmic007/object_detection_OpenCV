import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque


model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("./videos/people_walking.mp4")

# Video writer setup
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("annotated_people_tracking.mp4", cv2.VideoWriter_fourcc(*"avc1"), fps, (frame_width, frame_height))

id_map = {}  # Map from YOLO's object ID to our custom ID
next_id = 1  # Next custom ID to assign

trail = defaultdict(
    lambda: deque(maxlen=30)
)  # Dictionary to store trails for each object ID
appearedIn = defaultdict(int)  # Dictionary to count appearances for each object ID


while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame")
        break

    res = model.track(
        frame, classes=[0], persist=True, verbose=False
    )  # Track only 'person' class (class ID 0)
    annotated_frame = frame.copy()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.cpu().numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(
                int, box
            )  # Extract coordinates and Convert box coordinates to integers
            cx, cy = (x1 + x2) // 2, (
                y1 + y2
            ) // 2  # Calculate the center point of the bounding box

            appearedIn[oid] += 1  # Increment the count for this object ID

            if (
                appearedIn[oid] >= 5 and oid not in id_map
            ):  # If the object has appeared in at least 5 frames
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))  # Append the center point to the trail
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"ID: {sid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

                # Draw the trail
                for i in range(1, len(trail[oid])):
                    cv2.line(
                        annotated_frame,
                        trail[oid][i - 1],
                        trail[oid][i],
                        (0, 255, 0),
                        2,
                    )
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

    # Write annotated frame to video
    out.write(annotated_frame)
    
    # Resize frame for display
    scale_percent = 50  # Resize to 50% of original size
    width = int(annotated_frame.shape[1] * scale_percent / 100)
    height = int(annotated_frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(annotated_frame, (width, height))

    cv2.imshow("Tracking with Trail", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Tracking complete. Total unique people tracked: {len(id_map)}")
print("Annotated video saved as 'annotated_people_tracking.mp4'")
