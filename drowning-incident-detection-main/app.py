from pathlib import Path

import cv2
import numpy as np
import PIL
import streamlit as st

import helper
import settings
from dashboard import Dashboard

st.set_page_config(
    page_title="Drowning Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Drowning-probability detection")

st.sidebar.header("ML Model Config")
confidence = float(st.sidebar.slider("Select Model Confidence", 15, 60, 25)) / 100
settings.DRY_SCENE_GUARD_ENABLED = st.sidebar.checkbox(
    "Mark dry scenes as out of water",
    value=settings.DRY_SCENE_GUARD_ENABLED,
    help="Prevents a normal webcam image with no visible pool water from being labelled Swimming or Drowning.",
)

st.sidebar.header("Alert Config")
alert_provider = st.sidebar.selectbox(
    "Alert Provider",
    ["WAHA (Free WhatsApp API)", "Twilio (Paid)"],
    index=0,
    help="WAHA connects directly to WhatsApp Web for free without per-message fees.",
)

if alert_provider.startswith("WAHA"):
    settings.ALERT_PROVIDER = "WAHA"
    settings.WAHA_BASE_URL = st.sidebar.text_input(
        "WAHA Server URL",
        value=getattr(settings, "WAHA_BASE_URL", "http://localhost:3000"),
        help="Default is http://localhost:3000 when running WAHA via Docker",
    )
    settings.WAHA_PHONE_NUMBER = st.sidebar.text_input(
        "Alert WhatsApp Number",
        value=getattr(settings, "WAHA_PHONE_NUMBER", ""),
        help="Country code + phone number (e.g. 919876543210 or 12345678901)",
    )
    with st.sidebar.expander("Advanced WAHA Settings"):
        settings.WAHA_SESSION = st.text_input(
            "Session Name",
            value=getattr(settings, "WAHA_SESSION", "default"),
        )
        settings.WAHA_API_KEY = st.text_input(
            "WAHA API Key (Optional)",
            value=getattr(settings, "WAHA_API_KEY", ""),
            type="password",
        )

    if st.sidebar.button("Test WAHA Connection"):
        from alert_manager import AlertManager
        status_info = AlertManager.check_waha_status(
            waha_url=settings.WAHA_BASE_URL,
            session=settings.WAHA_SESSION,
            api_key=settings.WAHA_API_KEY,
        )
        if status_info["connected"]:
            if status_info["status"] == "WORKING":
                st.sidebar.success("WAHA Connected & WhatsApp session is WORKING!")
            elif status_info["status"] == "SCAN_QR_CODE":
                st.sidebar.warning("WAHA is running, but QR code needs to be scanned at http://localhost:3000/dashboard")
            else:
                st.sidebar.info(status_info["message"])
        else:
            st.sidebar.error(status_info["message"])
else:
    settings.ALERT_PROVIDER = "Twilio"
    settings.account_sid = st.sidebar.text_input("Your Twilio account_sid", value=settings.account_sid)
    settings.auth_token = st.sidebar.text_input("Your Twilio auth_token", value=settings.auth_token, type="password")
    settings.to_ = st.sidebar.text_input("Your Twilio to_", value=settings.to_)
    settings.from_ = st.sidebar.text_input("Your Twilio from_", value=settings.from_)
    settings.content_sid = st.sidebar.text_input("Twilio Content SID", value=settings.content_sid)
    settings.imgbb_api = st.sidebar.text_input("Your ImgBB api-key", value=settings.imgbb_api)

model_path = "best.pt"

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()

dashboard = Dashboard()

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    if source_img is None:
        default_image_path = str(settings.DEFAULT_IMAGE)
        st.image(default_image_path, caption="Default Image", use_container_width=True)

        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
        st.image(
            default_detected_image_path,
            caption="Detected Probability",
            use_container_width=True,
        )
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img, caption="Uploaded Image", use_container_width=True)

        if st.sidebar.button("Detect Drowning"):
            res = model.predict(uploaded_image, conf=confidence, verbose=False)
            uploaded_bgr = cv2.cvtColor(
                np.array(uploaded_image), cv2.COLOR_RGB2BGR
            )
            result = helper.refine_predictions(res[0], uploaded_bgr)
            boxes = result.boxes
            res_plotted = result.plot()[:, :, ::-1]
            st.image(res_plotted, caption="Detected Image", use_container_width=True)

            with st.expander("Detected Probability"):
                if len(boxes) == 0:
                    st.write("No detections found.")
                for box in boxes:
                    st.write(box.data)

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

elif source_radio == settings.MULTI_CAMERA:
    helper.play_multi_camera(confidence, model, dashboard)

else:
    st.error("Please select a valid source type!")
