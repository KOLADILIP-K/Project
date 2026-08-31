import json
import threading
import time
from pathlib import Path

import cv2


class Camera:
    def __init__(self, camera_id, name, location, source):
        self.camera_id = camera_id
        self.name = name
        self.location = location
        self.source = self._normalize_source(source)

        self.cap = None
        self.frame = None
        self.online = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def _normalize_source(self, source):
        try:
            return int(source)
        except (TypeError, ValueError):
            return source

    def connect(self):
        self.release()

        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.online = True
        else:
            self.online = False

    def release(self):
        if self.cap:
            self.cap.release()
        self.cap = None

    def update(self):
        self.running = True
        self.connect()

        while self.running:
            if not self.online:
                time.sleep(1)
                self.connect()
                continue

            success, frame = self.cap.read()

            if success and frame is not None:
                with self.lock:
                    self.frame = frame.copy()
                self.online = True
            else:
                self.online = False
                self.release()
                time.sleep(1)

            time.sleep(0.01)

        self.release()

    def start(self):
        if self.thread and self.thread.is_alive():
            return

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.release()

    def snapshot(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()

        return {
            "id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "source": self.source,
            "online": self.online,
            "frame": frame,
        }


class CameraManager:
    def __init__(self, config_path="camera_config.json"):
        self.config_path = Path(config_path)
        self.cameras = []
        self.load_cameras()

    def load_cameras(self):
        self.stop()
        self.cameras = []

        if not self.config_path.exists():
            print(f"[ERROR] Camera config not found: {self.config_path}")
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        for cam in config.get("cameras", []):
            if not cam.get("enabled", True):
                continue

            camera = Camera(
                cam.get("id"),
                cam.get("name", f"Camera {cam.get('id', '')}"),
                cam.get("location", "Unknown"),
                cam.get("source", 0),
            )

            camera.start()
            self.cameras.append(camera)

    def reload(self):
        self.load_cameras()

    def get_frames(self):
        return [camera.snapshot() for camera in self.cameras]

    def stop(self):
        for camera in self.cameras:
            camera.stop()