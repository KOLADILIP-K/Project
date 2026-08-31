import csv
import os
from datetime import datetime


class DetectionLogger:
    def __init__(self, log_file="logs/detections.csv"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        if not os.path.exists(log_file):
            with open(log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Date",
                        "Time",
                        "Camera",
                        "Location",
                        "Confidence",
                        "Image",
                        "Video",
                        "Alert Sent",
                    ]
                )

    def log(self, camera, location, confidence, image_path, video_path, alert=True):
        now = datetime.now()

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    now.strftime("%d-%m-%Y"),
                    now.strftime("%H:%M:%S"),
                    camera,
                    location,
                    round(float(confidence), 2),
                    image_path,
                    video_path,
                    "YES" if alert else "NO",
                ]
            )