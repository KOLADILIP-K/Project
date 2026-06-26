import math
import streamlit as st
import cv2


class Dashboard:

    def __init__(self):
        self.frames = {}

    def _resize(self, frame):
        if frame is None:
            return None

        h, w = frame.shape[:2]

        max_width = 640

        if w > max_width:
            ratio = max_width / w
            frame = cv2.resize(
                frame,
                (int(w * ratio), int(h * ratio))
            )

        return frame

    def draw(self, camera_data):

        count = len(camera_data)

        if count == 0:
            st.warning("No Cameras Connected")
            return

        columns = math.ceil(math.sqrt(count))

        rows = math.ceil(count / columns)

        index = 0

        for r in range(rows):

            cols = st.columns(columns)

            for c in range(columns):

                if index >= count:
                    break

                camera = camera_data[index]

                with cols[c]:

                    st.markdown(
                        f"""
                        ### 📷 {camera['name']}

                        **Location:** {camera['location']}

                        **Status:** {"🟢 Online" if camera['online'] else "🔴 Offline"}
                        """
                    )

                    placeholder = st.empty()

                    frame = camera["frame"]

                    frame = self._resize(frame)

                    if frame is not None:

                        placeholder.image(
                            frame,
                            channels="BGR",
                            use_container_width=True
                        )

                    else:

                        placeholder.warning(
                            "Waiting for Camera..."
                        )

                index += 1

    def statistics(self, camera_data):

        total = len(camera_data)

        online = sum(
            cam["online"]
            for cam in camera_data
        )

        offline = total - online

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Cameras",
            total
        )

        c2.metric(
            "Online",
            online
        )

        c3.metric(
            "Offline",
            offline
        )