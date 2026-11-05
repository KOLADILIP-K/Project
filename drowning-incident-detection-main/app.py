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
confidence = float(st.sidebar.slider("Select Model Confidence", 5, 80, int(settings.CONFIDENCE * 100), step=5)) / 100
settings.DROWNING_ALERT_CONFIDENCE = float(
    st.sidebar.slider(
        "Drowning Alert Confidence Threshold",
        20,
        80,
        int(getattr(settings, "DROWNING_ALERT_CONFIDENCE", 0.38) * 100),
        step=2,
        help="Minimum confidence required on a frame to count as drowning (lower enables detection on challenging clips).",
    )
) / 100
settings.SAFE_DROWNING_FRAMES = st.sidebar.slider(
    "Safe Drowning Frames (Threshold)",
    3,
    20,
    settings.SAFE_DROWNING_FRAMES,
    help="Number of consecutive drowning frames required before confirming drowning and triggering the alarm.",
)
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

settings.imgbb_api = st.sidebar.text_input(
    "ImgBB Cloud API Key (Optional)",
    value=getattr(settings, "imgbb_api", ""),
    type="password",
    help="Enter ImgBB API key from https://api.imgbb.com/ to upload evidence photos to cloud.",
)

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
    sample_options = {
        "Test Case 1: Drowning Incident (Pool)": settings.ROOT / "images" / "sample_drowning.jpg",
        "Test Case 2: Active Swimming": settings.ROOT / "images" / "sample_swimming.jpg",
        "Test Case 3: Multi-Person (Webcam/Room)": settings.ROOT / "images" / "sample_multiperson.jpg",
        "Upload Custom Image...": None,
    }

    image_source_choice = st.sidebar.selectbox(
        "Choose Image Source",
        options=list(sample_options.keys()),
        index=0,
    )

    image_to_detect = None
    if image_source_choice == "Upload Custom Image...":
        source_img = st.sidebar.file_uploader(
            "Upload Image File", type=("jpg", "jpeg", "png", "bmp", "webp")
        )
        if source_img is not None:
            image_to_detect = PIL.Image.open(source_img)
            st.image(image_to_detect, caption="Uploaded Image", use_container_width=True)
        else:
            st.info("Please upload an image from the sidebar or choose one of the sample test cases.")
    else:
        sample_path = sample_options[image_source_choice]
        if sample_path.exists():
            image_to_detect = PIL.Image.open(str(sample_path))
            st.image(image_to_detect, caption=f"Selected: {image_source_choice}", use_container_width=True)
        else:
            st.warning(f"Sample file not found: {sample_path}")

    if image_to_detect is not None:
        if st.sidebar.button("Detect Drowning"):
            res = model.predict(image_to_detect, conf=confidence, verbose=False)
            uploaded_bgr = cv2.cvtColor(
                np.array(image_to_detect), cv2.COLOR_RGB2BGR
            )
            result = helper.refine_predictions(res[0], uploaded_bgr)
            boxes = result.boxes
            res_plotted = result.plot()[:, :, ::-1]
            st.image(res_plotted, caption="Detected Result", use_container_width=True)

            with st.expander("Detected Probability & Persons Details", expanded=True):
                if len(boxes) == 0:
                    st.info("No detections found.")
                else:
                    st.success(f"Detected {len(boxes)} person(s)")
                    for idx, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        cls_name = model.names.get(cls_id, f"Class {cls_id}")
                        conf_val = float(box.conf[0])
                        coords = [round(float(c), 1) for c in box.xyxy[0]]
                        st.markdown(f"- **Person {idx + 1}:** `{cls_name}` | Confidence: **{conf_val * 100:.1f}%** | Box: `{coords}`")

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
