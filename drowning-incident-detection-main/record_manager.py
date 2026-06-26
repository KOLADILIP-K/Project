import time
from pathlib import Path

import cv2


class VideoRecorder:
    def __init__(self, output_dir="evidence/videos", fps=20, codec="mp4v"):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.codec = codec
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        self.recording = False
        self.start_time = 0
        self.duration = 0
        self.current_path = ""

    def _safe_name(self, camera_name):
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(camera_name)
        )
        return safe.strip("_") or "camera"

    def start(self, frame, camera_name, duration=15):
        if self.recording:
            return self.current_path

        height, width = frame.shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{self._safe_name(camera_name)}_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(str(filename), fourcc, self.fps, (width, height))

        if not self.writer.isOpened():
            self.writer = None
            self.recording = False
            self.current_path = ""
            return ""

        self.recording = True
        self.start_time = time.time()
        self.duration = duration
        self.current_path = str(filename)

        print(f"[INFO] Recording started -> {self.current_path}")
        return self.current_path

    def write(self, frame):
        if not self.recording or not self.writer:
            return

        self.writer.write(frame)

        if time.time() - self.start_time >= self.duration:
            self.stop()

    def stop(self):
        if self.writer:
            self.writer.release()

        self.writer = None
        self.recording = False
        print("[INFO] Recording stopped")

    def is_recording(self):
        return self.recording