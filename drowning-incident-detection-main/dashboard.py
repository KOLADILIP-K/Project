import math

import cv2
import streamlit as st


class Dashboard:
    def _resize(self, frame):
        if frame is None:
            return None

        h, w = frame.shape[:2]
        max_width = 640

        if w > max_width:
            ratio = max_width / w
            frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))

        return frame

    def statistics(self, camera_data):
        total = len(camera_data)
        online = sum(1 for cam in camera_data if cam.get("online"))
        offline = total - online
        detections = sum(1 for cam in camera_data if cam.get("drowning"))
        alarm_triggered = any(cam.get("drowning") or cam.get("alarm_triggered") for cam in camera_data)

        if alarm_triggered:
            st.error("## 🚨 ALARM TRIGGERED — POTENTIAL DROWNING INCIDENT IN PROGRESS!")
        else:
            st.success("## 🟢 STATUS: NORMAL — All Monitored Areas Secure")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Cameras", total)
        c2.metric("Online", online)
        c3.metric("Offline", offline)
        c4.metric("Detections", detections)
        c5.metric("Alarm Status", "🚨 TRIGGERED" if alarm_triggered else "🟢 NORMAL")

    def draw(self, camera_data):
        count = len(camera_data)

        if count == 0:
            st.warning("No cameras found in camera_config.json")
            return

        columns = min(4, max(1, math.ceil(math.sqrt(count))))
        rows = math.ceil(count / columns)
        index = 0

        for _ in range(rows):
            cols = st.columns(columns)

            for col in cols:
                if index >= count:
                    break

                camera = camera_data[index]

                with col:
                    status = "Online" if camera.get("online") else "Offline"
                    detection = camera.get("drowning", False) or camera.get("alarm_triggered", False)
                    confidence = camera.get("confidence", 0.0)

                    st.markdown(f"### 📷 {camera.get('name', 'Camera')}")
                    st.caption(f"{camera.get('location', 'Unknown')} | {status}")

                    if detection:
                        st.error(f"🚨 **ALARM TRIGGERED** ({confidence * 100:.1f}%)")
                    else:
                        st.caption("🟢 Status: Safe")

                    if camera.get("error"):
                        st.warning(camera["error"])

                    frame = self._resize(camera.get("frame"))

                    if frame is not None:
                        st.image(frame, channels="BGR", use_container_width=True)
                    else:
                        st.info("Waiting for camera...")

                index += 1