"""
YOLO-based Hospital Fall Detection System
Developer: Madhu Venkata Sai
"""

import os
import time
import pandas as pd
from ultralytics import YOLO
import cv2
import cvzone

# ---------------- CONFIG ----------------
VIDEO_PATH = 'fallBed.mp4'
fall_threshold = 5        # frames to confirm fall
skip_frames = 2           # process every 2nd frame
alarm_cooldown = 5        # seconds between alerts
output_dir = 'fall_outputs'
os.makedirs(output_dir, exist_ok=True)

# ---------------- DOWNLOAD SAMPLE VIDEO ----------------
if not os.path.exists(VIDEO_PATH):
    print("Downloading sample test video...")
    os.system("wget -O fallBed.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/person-detection.mp4")

# ---------------- LOAD YOLO MODEL ----------------
print("Downloading and loading YOLOv8 model...")
model = YOLO('yolov8n.pt')   # auto-downloads
class_list = list(model.names.values())

# ---------------- OPEN VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError("Video not found.")

print("Video opened successfully.")

fall_frame_count = {}
last_alert_time = {}
frame_index = 0
saved_count = 0

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_index += 1
    if frame_index % skip_frames != 0:
        continue

    frame = cv2.resize(frame, (1024, 600))
    results = model(frame)
    detections = results[0].boxes.data

    if len(detections) == 0:
        continue

    px = pd.DataFrame(detections).astype(float)

    for _, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        class_id = int(row[5])
        class_name = class_list[class_id]

        if class_name != 'person':
            continue

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = height / width

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pid = f"{cx}-{cy}"

        if aspect_ratio < 0.8:
            fall_frame_count[pid] = fall_frame_count.get(pid, 0) + 1
            print(f"[Frame {frame_index}] Possible fall detected")

            if fall_frame_count[pid] >= fall_threshold:
                now = time.time()
                if now - last_alert_time.get(pid, 0) >= alarm_cooldown:

                    cvzone.putTextRect(frame, "FALL DETECTED!", (x1, y1 - 10), 1, 1, colorR=(0, 0, 255))
                    cv2.circle(frame, (cx, cy), int(max(width, height) * 0.6), (0, 0, 255), 5)

                    out_path = f"{output_dir}/fall_detected_{saved_count}.jpg"
                    cv2.imwrite(out_path, frame)

                    print("FALL CONFIRMED — Image saved:", out_path)

                    last_alert_time[pid] = now
                    saved_count += 1
        else:
            fall_frame_count[pid] = 0

cap.release()
print("Processing complete. Check saved images in fall_outputs folder.")
