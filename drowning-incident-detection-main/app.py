from pathlib import Path

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

st.sidebar.header("Alert Config")
settings.account_sid = st.sidebar.text_input("Your Twilio account_sid")
settings.auth_token = st.sidebar.text_input("Your Twilio auth_token")
settings.to_ = st.sidebar.text_input("Your Twilio to_")
settings.from_ = st.sidebar.text_input("Your Twilio from_")
settings.imgbb_api = st.sidebar.text_input("Your ImgBB api-key")

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
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
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