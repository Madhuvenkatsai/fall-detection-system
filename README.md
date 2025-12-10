# Fall Detection System (YOLO + OpenCV)

This project implements an intelligent hospital fall detection system using
YOLO for human detection and OpenCV for video frame processing. It is designed
to detect when a patient has fallen (lying posture) and save annotated evidence
frames automatically.

---

## Features

- YOLO-based person detection (Ultralytics YOLOv8)
- Aspect-ratio based fall classification
- Automatic saving of frames when a fall is confirmed
- Works with any input video (default sample video auto-downloads)
- Can run in Google Colab or on a local machine

---

## Tech Stack

- Python 3.x
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- OpenCV
- Pandas
- cvzone

---

## Installation (Local Machine)

```bash
git clone https://github.com/Madhuvenkatsai/fall-detection-system.git
cd fall-detection-system

pip install -r requirements.txt
