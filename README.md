# AI-Powered Drowning Incident Detection System

Real-time computer vision system for automated drowning incident detection, continuous distress alarms, and multi-channel emergency alert dispatch (WhatsApp via WAHA / Twilio & Cloud Evidence Hosting).

---

## 🏊 Key Features & Enhancements (v2.0)

- **Multi-Person Detection Ensemble**: Integrated dedicated YOLOv11 person detection model (`weights/yolo11n.pt`) with spatial IoU & containment safeguards to reliably detect multiple swimmers/persons concurrently without box merging or confusion.
- **Continuous Siren / Alarm**: Continuous looping audio alert (`distress.mp3`) that persists throughout the duration of a drowning incident until verified safe recovery.
- **Visual Alert Banner**: Real-time high-visibility "ALARM TRIGGERED" banner overlaid directly on the video feed and in the monitoring metrics dashboard.
- **Multi-Frame Safe Confirmation**:
  - `SAFE_DROWNING_FRAMES = 8`: Multi-frame confirmation prevents single-frame false alarms and water splash noise.
  - `SAFE_RECOVERY_FRAMES = 5`: Recovery stabilization prevents alarm chatter.
- **Optimized Video Benchmarks**:
  - Video 1 (`drowning_1.mp4`): Drowning incident accurately detected and sustained.
  - Video 2 (`test.mp4`): Trimmed to 10s active swimming benchmark clip with 0 false alarms.
  - Video 3 (`12727733-preview.mp4`): Rapid drowning confirmation at frame 23 (<1s).
- **Interactive Test Cases**: Pre-configured dropdown test cases for instant verification:
  1. *Test Case 1*: Drowning Incident (Pool)
  2. *Test Case 2*: Active Swimming (Safe)
  3. *Test Case 3*: Multi-Person (Webcam/Room)
- **Cloud Evidence Upload**: Optional ImgBB API cloud photo upload for dispatching incident evidence photos via WhatsApp.
- **Webcam Latency Fix**: DirectShow backend (`cv2.CAP_DSHOW`) with buffer size 1 eliminates frame freezing and latency backlog.

---

## 🚀 Quick Start Guide

The full project code and assets are located in the [`drowning-incident-detection-main`](./drowning-incident-detection-main) directory.

### 1. Environment Setup
```bash
cd drowning-incident-detection-main
python -m venv venv
venv\Scripts\activate
pip install -r installs.txt
```

### 2. Run Application
```bash
python run.py
```
Or directly using Streamlit:
```bash
streamlit run app.py
```

### 3. WhatsApp Alerts (WAHA / Twilio)
Refer to the detailed setup instructions in [`drowning-incident-detection-main/README.md`](./drowning-incident-detection-main/README.md).
