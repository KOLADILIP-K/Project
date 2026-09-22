import base64
import json
import re
import time
from pathlib import Path

import cv2
import requests

try:
    from twilio.rest import Client
    HAS_TWILIO = True
except ImportError:
    Client = None
    HAS_TWILIO = False


class AlertManager:
    def __init__(
        self,
        account_sid="",
        auth_token="",
        from_number="",
        to_number="",
        imgbb_api="",
        content_sid="",
        evidence_dir="evidence/images",
        provider="WAHA",
        waha_url="http://localhost:3000",
        waha_session="default",
        waha_phone="",
        waha_api_key="",
    ):
        self.provider = (provider or "WAHA").upper()

        # Twilio settings
        self.account_sid = account_sid or ""
        self.auth_token = auth_token or ""
        self.from_number = from_number or ""
        self.to_number = to_number or ""
        self.content_sid = content_sid or ""

        # WAHA settings
        self.waha_url = (waha_url or "http://localhost:3000").rstrip("/")
        self.waha_session = waha_session or "default"
        self.waha_phone = waha_phone or ""
        self.waha_api_key = waha_api_key or ""

        # ImgBB & storage
        self.imgbb_api = imgbb_api or ""
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, camera_name):
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(camera_name)
        )
        return safe.strip("_") or "camera"

    def _format_waha_chat_id(self, phone):
        """Format a phone number into WAHA chatId (e.g. 1234567890@c.us)."""
        phone = str(phone).strip()
        if "@" in phone:
            return phone

        # Strip all non-digit characters (+, -, spaces, parentheses)
        digits = re.sub(r"\D", "", phone)
        # Strip leading double zero if present
        if digits.startswith("00"):
            digits = digits[2:]

        return f"{digits}@c.us"

    def save_snapshot(self, frame, camera_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = (
            self.evidence_dir
            / f"{self._safe_name(camera_name)}_{timestamp}.jpg"
        )

        success = cv2.imwrite(str(filename), frame)
        if not success:
            print("[ERROR] Failed to save snapshot.")
            return None

        print(f"[INFO] Snapshot saved: {filename}")
        return str(filename)

    def upload_image(self, image_path):
        if not self.imgbb_api:
            return None

        if not image_path:
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
                data = response.json()
                image_url = data["data"]["url"]
                print(f"[INFO] Image uploaded to ImgBB: {image_url}")
                return image_url

            print(f"[ERROR] ImgBB upload failed: HTTP {response.status_code}")
        except Exception as exc:
            print(f"[ERROR] ImgBB upload failed: {exc}")

        return None

    @classmethod
    def check_waha_status(cls, waha_url="http://localhost:3000", session="default", api_key=""):
        """Check connection to WAHA server and return session status."""
        base_url = (waha_url or "http://localhost:3000").rstrip("/")
        headers = {}
        if api_key:
            headers["X-Api-Key"] = api_key

        try:
            resp = requests.get(f"{base_url}/api/sessions/{session}", headers=headers, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "UNKNOWN")
                return {
                    "connected": True,
                    "status": status,
                    "message": f"WAHA is running. Session '{session}' status: {status}",
                }
            elif resp.status_code == 404:
                return {
                    "connected": True,
                    "status": "NOT_FOUND",
                    "message": f"WAHA is running, but session '{session}' was not found. Please create/start it in the dashboard.",
                }
            else:
                return {
                    "connected": True,
                    "status": f"HTTP_{resp.status_code}",
                    "message": f"WAHA responded with HTTP {resp.status_code}: {resp.text}",
                }
        except requests.exceptions.ConnectionError:
            return {
                "connected": False,
                "status": "OFFLINE",
                "message": f"Cannot connect to WAHA at {base_url}. Make sure Docker is running and WAHA container is started.",
            }
        except Exception as e:
            return {
                "connected": False,
                "status": "ERROR",
                "message": f"WAHA check failed: {e}",
            }

    def send_waha(
        self,
        camera_name,
        location,
        confidence,
        image_path=None,
        image_url=None,
    ):
        """Send a WhatsApp message and snapshot via WAHA REST API."""
        if not self.waha_phone:
            print("[WARN] WAHA target phone number is not set. Alert not sent.")
            return None

        chat_id = self._format_waha_chat_id(self.waha_phone)
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        alert_text = (
            "🚨 *DROWNING INCIDENT ALERT* 🚨\n\n"
            "⚠️ *Potential drowning detected!*\n"
            f"📹 *Camera:* {camera_name}\n"
            f"📍 *Location:* {location}\n"
            f"📊 *Confidence:* {confidence * 100:.1f}%\n"
            f"🕒 *Timestamp:* {timestamp_str}\n\n"
            "⚡ *Please respond immediately!*"
        )

        headers = {"Content-Type": "application/json"}
        if self.waha_api_key:
            headers["X-Api-Key"] = self.waha_api_key

        # If a local snapshot exists, send it directly via Base64 with caption
        if image_path and Path(image_path).is_file():
            try:
                with open(image_path, "rb") as img_file:
                    b64_data = base64.b64encode(img_file.read()).decode("utf-8")

                payload = {
                    "session": self.waha_session,
                    "chatId": chat_id,
                    "file": {
                        "mimetype": "image/jpeg",
                        "filename": Path(image_path).name,
                        "data": b64_data,
                    },
                    "caption": alert_text,
                }

                url = f"{self.waha_url}/api/sendImage"
                resp = requests.post(url, json=payload, headers=headers, timeout=25)

                if resp.status_code in (200, 201):
                    msg_id = resp.json().get("id", "SENT")
                    print(f"[INFO] WAHA WhatsApp image alert sent: {msg_id}")
                    return str(msg_id)
                else:
                    print(
                        f"[WARN] WAHA sendImage returned HTTP {resp.status_code}: {resp.text}. "
                        "Falling back to text message..."
                    )
            except Exception as e:
                print(f"[WARN] WAHA sendImage exception: {e}. Falling back to text message...")

        # If image sending failed or no snapshot was provided, send text message
        try:
            full_text = alert_text
            if image_url:
                full_text += f"\n📷 *Snapshot Link:* {image_url}"

            payload = {
                "session": self.waha_session,
                "chatId": chat_id,
                "text": full_text,
            }

            url = f"{self.waha_url}/api/sendText"
            resp = requests.post(url, json=payload, headers=headers, timeout=15)

            if resp.status_code in (200, 201):
                msg_id = resp.json().get("id", "SENT")
                print(f"[INFO] WAHA WhatsApp text alert sent: {msg_id}")
                return str(msg_id)
            else:
                print(f"[ERROR] WAHA sendText failed: HTTP {resp.status_code} - {resp.text}")
                return None
        except requests.exceptions.ConnectionError:
            print(
                f"[ERROR] Cannot connect to WAHA at {self.waha_url}. "
                "Is the Docker container running?"
            )
            return None
        except Exception as exc:
            print(f"[ERROR] WAHA WhatsApp alert failed: {exc}")
            return None

    def send_twilio(
        self,
        camera_name,
        location,
        confidence,
        image_url=None,
    ):
        if not HAS_TWILIO:
            print("[ERROR] Twilio package is not installed.")
            return None

        required = [
            self.account_sid,
            self.auth_token,
            self.from_number,
            self.to_number,
            self.content_sid,
        ]

        if not all(required):
            print("[WARN] Twilio settings are incomplete. WhatsApp alert was not sent.")
            return None

        try:
            client = Client(self.account_sid, self.auth_token)

            content_variables = {
                "1": str(camera_name),
                "2": str(location),
                "3": f"{confidence:.2f}",
                "4": str(image_url) if image_url else "No image available",
            }

            message = client.messages.create(
                from_=f"whatsapp:{self.from_number}",
                to=f"whatsapp:{self.to_number}",
                content_sid=self.content_sid,
                content_variables=json.dumps(content_variables),
            )

            print(f"[INFO] Twilio WhatsApp alert sent: {message.sid}")
            return message.sid

        except Exception as exc:
            print(f"[ERROR] Twilio WhatsApp alert failed: {exc}")
            return None

    def trigger(
        self,
        frame,
        camera_name,
        location,
        confidence,
    ):
        # Save drowning evidence snapshot locally
        image_path = self.save_snapshot(frame, camera_name)

        # Upload evidence image to ImgBB if configured
        image_url = self.upload_image(image_path)

        # Send alert via chosen provider
        if self.provider == "WAHA":
            message_id = self.send_waha(
                camera_name=camera_name,
                location=location,
                confidence=confidence,
                image_path=image_path,
                image_url=image_url,
            )
        else:
            message_id = self.send_twilio(
                camera_name=camera_name,
                location=location,
                confidence=confidence,
                image_url=image_url,
            )

        return image_path, message_id