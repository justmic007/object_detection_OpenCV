import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture("./videos/people_walking2.mp4")

while True:
    ret, frame = cap.read()
    results = model.track(
        source=frame, classes=[0], persist=True, verbose=False
    )  # Track only 'person' class (class ID 0)

    for result in results:
        annotated_frame = result.plot()
        # Blend the annotated frame with the original frame to add transparency
        alpha = 0.4  # Transparency level (0.4 = 60% original, 40% mask)
        annotated_frame = cv2.addWeighted(annotated_frame, alpha, frame, 1 - alpha, 0)
        if (
            result.masks is not None
            and result.boxes is not None
            and result.boxes.id is not None
        ):
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()

            for i, mask in enumerate(masks):
                person_id = ids[i]
                x1, y1, x2, y2 = boxes[i]
                mask_resized = cv2.resize(
                    mask.astype(np.uint8) * 255, (frame.shape[1], frame.shape[0])
                )
                contours, _ = cv2.findContours(
                    mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated_frame, contours, -1, (150, 150, 255), 2)

                # Add a dark background box for better text visibility
                text = f"ID: {person_id}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x, text_y = int(x1), int(y1) - 10
                cv2.rectangle(
                    annotated_frame,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1,
                )  # Black background

                cv2.putText(
                    annotated_frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),  # Bright cyan text
                    2,
                )

    # Resize frame for display
    scale_percent = 50  # Resize to 50% of original size
    width = int(annotated_frame.shape[1] * scale_percent / 100)
    height = int(annotated_frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(annotated_frame, (width, height))

    cv2.imshow("Object Tracking with Segmentation", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Video processing completed.")
