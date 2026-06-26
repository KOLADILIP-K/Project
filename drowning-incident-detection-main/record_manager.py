import os
import cv2
import time
from pathlib import Path


class VideoRecorder:

    def __init__(
        self,
        output_dir="evidence/videos",
        fps=20,
        codec="mp4v"
    ):

        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec

        Path(output_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        self.writer = None
        self.recording = False
        self.start_time = 0

    def start(
        self,
        frame,
        camera_name,
        duration=15
    ):

        if self.recording:
            return

        height, width = frame.shape[:2]

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        filename = os.path.join(
            self.output_dir,
            f"{camera_name}_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        self.writer = cv2.VideoWriter(
            filename,
            fourcc,
            self.fps,
            (width, height)
        )

        self.recording = True
        self.start_time = time.time()
        self.duration = duration

        print(f"[INFO] Recording started -> {filename}")

    def write(self, frame):

        if not self.recording:
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