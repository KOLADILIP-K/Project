import cv2
import threading
import time


class Camera:

    def __init__(self, camera_id, name, location, source):

        self.camera_id = camera_id
        self.name = name
        self.location = location
        self.source = source

        try:
            self.source = int(source)
        except:
            pass

        self.cap = None
        self.frame = None
        self.online = False
        self.running = False

    def connect(self):

        self.cap = cv2.VideoCapture(self.source)

        if self.cap.isOpened():
            self.online = True
            print(f"[INFO] {self.name} Connected")
        else:
            self.online = False
            print(f"[ERROR] {self.name} Offline")

    def reconnect(self):

        if self.cap:
            self.cap.release()

        time.sleep(2)

        self.connect()

    def update(self):

        self.connect()

        self.running = True

        while self.running:

            if not self.online:

                self.reconnect()

            success, frame = self.cap.read()

            if success:

                self.frame = frame

            else:

                self.online = False

                self.reconnect()

    def start(self):

        thread = threading.Thread(target=self.update)

        thread.daemon = True

        thread.start()

    def stop(self):

        self.running = False

        if self.cap:
            self.cap.release()


class CameraManager:

    def __init__(self):

        self.cameras = []

    def add_camera(self,
                   source,
                   name,
                   location="Unknown"):

        camera = Camera(
            len(self.cameras) + 1,
            name,
            location,
            source
        )

        camera.start()

        self.cameras.append(camera)

    def get_frames(self):

        data = []

        for camera in self.cameras:

            data.append({

                "id": camera.camera_id,

                "name": camera.name,

                "location": camera.location,

                "online": camera.online,

                "frame": camera.frame

            })

        return data

    def stop(self):

        for camera in self.cameras:

            camera.stop()