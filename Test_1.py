from ultralytics import YOLO
import cv2
import supervision as sv

# Load YOLOv8 pretrained on COCO
model = YOLO("yolov8n.pt")  # or yolov8m.pt for better accuracy

# Tracker
tracker = sv.ByteTrack()

# Class names from COCO
CLASS_NAMES = model.model.names

# Define categories of interest
TARGET_CLASSES = {
    "person": (0, (0, 255, 0)),       # green
    "train": (7, (255, 0, 0)),        # blue
    "backpack": (24, (0, 165, 255)),  # orange
    "handbag": (26, (255, 255, 0)),   # cyan
    "suitcase": (28, (255, 0, 255)),  # magenta
}

# Open video stream
cap = cv2.VideoCapture("warehouse_video.mp4")  # or 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.4)

    # Convert to supervision detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Update tracker
    tracked = tracker.update_with_detections(detections)

    # Draw tracked objects
    for xyxy, class_id, track_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
        if track_id is None:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        class_name = CLASS_NAMES.get(int(class_id), "other")

        # Pick color: if in target list → custom color, else gray
        if class_name in TARGET_CLASSES:
            _, color = TARGET_CLASSES[class_name]
        else:
            color = (128, 128, 128)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show video
    cv2.imshow("Bag/Train/Person Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
