import time
from pathlib import Path

import cv2
import requests
from twilio.rest import Client


class AlertManager:
    def __init__(
        self,
        account_sid,
        auth_token,
        from_number,
        to_number,
        imgbb_api,
        evidence_dir="evidence/images",
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_number = to_number
        self.imgbb_api = imgbb_api
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, camera_name):
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(camera_name)
        )
        return safe.strip("_") or "camera"

    def save_snapshot(self, frame, camera_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.evidence_dir / f"{self._safe_name(camera_name)}_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        return str(filename)

    def upload_image(self, image_path):
        if not self.imgbb_api:
            return None

        try:
            with open(image_path, "rb") as image:
                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    params={"key": self.imgbb_api},
                    files={"image": image},
                    timeout=20,
                )

            if response.status_code == 200:
                return response.json()["data"]["url"]
        except Exception as exc:
            print(f"[ERROR] ImgBB upload failed: {exc}")

        return None

    def send_whatsapp(self, camera_name, location, confidence, image_url=None):
        required = [
            self.account_sid,
            self.auth_token,
            self.from_number,
            self.to_number,
        ]

        if not all(required):
            print("[WARN] Twilio settings are incomplete. Alert was not sent.")
            return None

        try:
            client = Client(self.account_sid, self.auth_token)

            body = f"""
DROWNING DETECTED

Camera: {camera_name}
Location: {location}
Confidence: {confidence:.2f}
Time: {time.strftime('%d-%m-%Y %H:%M:%S')}
"""

            kwargs = {
                "from_": f"whatsapp:{self.from_number}",
                "to": f"whatsapp:{self.to_number}",
                "body": body,
            }

            if image_url:
                kwargs["media_url"] = [image_url]

            message = client.messages.create(**kwargs)
            print(f"[INFO] Alert sent: {message.sid}")
            return message.sid

        except Exception as exc:
            print(f"[ERROR] Twilio alert failed: {exc}")
            return None

    def trigger(self, frame, camera_name, location, confidence):
        image_path = self.save_snapshot(frame, camera_name)
        image_url = self.upload_image(image_path)
        message_sid = self.send_whatsapp(camera_name, location, confidence, image_url)
        return image_path, message_sid