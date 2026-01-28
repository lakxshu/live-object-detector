import cv2
from ultralytics import YOLO
from datetime import datetime
from playsound import playsound
import threading
import time

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open webcam with fixed resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Objects to detect with colors (BGR)
TARGET_OBJECTS = {
    "bottle": (255, 0, 0),
    "cup": (0, 255, 0),
    "fork": (0, 255, 255),
    "knife": (255, 255, 0),  # ALERT OBJECT
    "spoon": (255, 0, 255),
    "bowl": (0, 128, 255),
    "mouse": (128, 0, 255),
    "remote": (128, 255, 0),
    "cell phone": (0, 0, 255),
    "book": (200, 200, 200),
    "scissors": (100, 100, 255)
}

# Alert cooldown to prevent repeated alerts
alert_cooldown = 3  # seconds
last_alert_time = 0

# Play alert sound (non-blocking)
def play_alert():
    playsound(r"e:\ads_classifier\mixkit-classic-alarm-995.wav")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Count objects
    counts = {obj: 0 for obj in TARGET_OBJECTS}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.cpu().numpy())
            label = model.names[cls_id]

            if label in TARGET_OBJECTS:
                counts[label] += 1
                color = TARGET_OBJECTS[label]

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

                # Thicker box for knife
                thickness = 5 if label == "knife" else 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Knife alert
                if label == "knife":
                    cv2.putText(frame, "ALERT : KNIFE DETECTED!",
                                (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 3)

                    # Trigger alert only if cooldown passed
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        last_alert_time = current_time
                        threading.Thread(target=play_alert, daemon=True).start()

    # Show counts for detected objects only
    y = 30
    for obj, count in counts.items():
        if count > 0:
            cv2.putText(frame, f"{obj}: {count}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        TARGET_OBJECTS[obj],
                        2)
            y += 25

    # Show current time
    time_now = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {time_now}",
                (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 




