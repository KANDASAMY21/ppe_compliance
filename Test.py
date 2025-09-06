from ultralytics import YOLO
import cv2
import supervision as sv

# Load YOLO model
model = YOLO("yolov8n.pt")  # replace with fine-tuned bag model

# Tracker
tracker = sv.ByteTrack()

# Video stream
cap = cv2.VideoCapture("warehouse_video.mp4")  # or 0 for webcam

bag_count = 0
moving_bags = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.4)

    # Convert to supervision detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Filter for "bag" class (assuming 0 is bag in your custom model)
    bag_detections = detections[detections.class_id == 0]

    # Update tracker
    tracked = tracker.update_with_detections(bag_detections)

    # Iterate through tracked detections
    for xyxy, track_id in zip(tracked.xyxy, tracked.tracker_id):
        if track_id is None:
            continue  # skip untracked

        # Extract box coords
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Count unique IDs
        if track_id not in moving_bags:
            moving_bags.add(track_id)
            bag_count += 1

        # Draw bounding box + ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Bag {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show counter
    cv2.putText(frame, f"Moving Bags: {bag_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Bag Movement Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
