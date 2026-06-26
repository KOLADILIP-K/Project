import os
import cv2
import time
import requests
from pathlib import Path
from twilio.rest import Client


class AlertManager:

    def __init__(
        self,
        account_sid,
        auth_token,
        from_number,
        to_number,
        imgbb_api,
        evidence_dir="evidence"
    ):

        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_number = to_number
        self.imgbb_api = imgbb_api

        self.evidence_dir = evidence_dir

        Path(self.evidence_dir).mkdir(
            parents=True,
            exist_ok=True
        )

    def save_snapshot(self, frame, camera_name):

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        filename = os.path.join(
            self.evidence_dir,
            f"{camera_name}_{timestamp}.jpg"
        )

        cv2.imwrite(filename, frame)

        return filename

    def upload_image(self, image_path):

        try:

            with open(image_path, "rb") as image:

                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    params={
                        "key": self.imgbb_api
                    },
                    files={
                        "image": image
                    }
                )

            if response.status_code == 200:

                return response.json()["data"]["url"]

        except Exception as e:

            print(e)

        return None

    def send_whatsapp(
        self,
        camera_name,
        location,
        confidence,
        image_url
    ):

        try:

            client = Client(
                self.account_sid,
                self.auth_token
            )

            body = f"""
⚠ DROWNING DETECTED

Camera : {camera_name}

Location : {location}

Confidence : {confidence:.2f}

Time : {time.strftime('%d-%m-%Y %H:%M:%S')}
"""

            message = client.messages.create(

                from_=f"whatsapp:{self.from_number}",

                to=f"whatsapp:{self.to_number}",

                body=body,

                media_url=[image_url]

            )

            print("Alert Sent")

            return message.sid

        except Exception as e:

            print(e)

            return None

    def trigger(
        self,
        frame,
        camera_name,
        location,
        confidence
    ):

        image = self.save_snapshot(
            frame,
            camera_name
        )

        image_url = self.upload_image(image)

        if image_url:

            self.send_whatsapp(

                camera_name,

                location,

                confidence,

                image_url

            )