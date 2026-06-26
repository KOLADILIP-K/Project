import base64
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import streamlit as st
import yt_dlp
from ultralytics import YOLO

import settings
from alert_manager import AlertManager
from camera_manager import CameraManager
from logger import DetectionLogger
from record_manager import VideoRecorder

DROWNING_CLASS_ID = 0
DROWNING_ALERT_CONFIDENCE = 0.60


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ("Yes", "No"))

    if display_tracker == "Yes":
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return True, tracker_type

    return False, None


def autoplay_audio(file_path):
    audio_path = Path(file_path)

    if not audio_path.is_absolute():
        audio_path = settings.ROOT / audio_path

    if not audio_path.exists():
        return

    with open(audio_path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def _safe_name(name):
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(name)
    )
    return safe.strip("_") or "camera"


def _get_logger():
    if "detection_logger" not in st.session_state:
        st.session_state.detection_logger = DetectionLogger(str(settings.LOG_FILE))

    return st.session_state.detection_logger


def _get_alert_manager():
    signature = (
        settings.account_sid,
        settings.auth_token,
        settings.from_,
        settings.to_,
        settings.imgbb_api,
    )

    if st.session_state.get("alert_manager_signature") != signature:
        st.session_state.alert_manager = AlertManager(
            settings.account_sid,
            settings.auth_token,
            settings.from_,
            settings.to_,
            settings.imgbb_api,
            evidence_dir=str(settings.IMAGE_DIR),
        )
        st.session_state.alert_manager_signature = signature

    return st.session_state.alert_manager


def _get_recorders():
    if "video_recorders" not in st.session_state:
        st.session_state.video_recorders = {}

    return st.session_state.video_recorders


def _get_recorder(camera_name):
    recorders = _get_recorders()
    key = _safe_name(camera_name)

    if key not in recorders:
        recorders[key] = VideoRecorder(
            output_dir=str(settings.VIDEO_SAVE_DIR),
            fps=settings.FPS,
            codec=settings.VIDEO_CODEC,
        )

    return recorders[key]


def _get_last_alerts():
    if "last_detection_alerts" not in st.session_state:
        st.session_state.last_detection_alerts = {}

    return st.session_state.last_detection_alerts


def _best_confidence(result):
    try:
        classes = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
    except Exception:
        return 0.0

    drowning_scores = [
        float(confidence)
        for class_id, confidence in zip(classes, confidences)
        if int(class_id) == DROWNING_CLASS_ID
    ]

    return max(drowning_scores) if drowning_scores else 0.0


def _has_drowning(result):
    confidence = _best_confidence(result)
    return confidence >= DROWNING_ALERT_CONFIDENCE


def _should_alert(camera_name):
    last_alerts = _get_last_alerts()
    now = time.time()
    last = last_alerts.get(camera_name, 0)

    if now - last < settings.ALERT_COOLDOWN:
        return False

    last_alerts[camera_name] = now
    return True


def _run_detection(model, frame, conf, tracking=False, tracker=None):
    if tracking:
        results = model.track(
            frame,
            conf=conf,
            persist=True,
            tracker=tracker,
            verbose=False,
        )
    else:
        results = model.predict(frame, conf=conf, verbose=False)

    return results[0]


def _write_active_recording(camera_name, frame):
    recorder = _get_recorder(camera_name)

    if recorder.is_recording():
        recorder.write(frame)


def _handle_drowning(frame, result, camera_name, location):
    confidence = _best_confidence(result)

    if confidence < DROWNING_ALERT_CONFIDENCE:
        return

    recorder = _get_recorder(camera_name)

    if not recorder.is_recording():
        recorder.start(frame, camera_name, duration=settings.RECORD_SECONDS)

    recorder.write(frame)
    video_path = recorder.current_path or ""

    if not _should_alert(camera_name):
        return

    st.warning(f"Drowning detected: {camera_name}")
    autoplay_audio(settings.AUDIO_PATH)

    alert_manager = _get_alert_manager()
    logger = _get_logger()

    image_path, alert_sent = alert_manager.trigger(
        frame,
        camera_name,
        location,
        confidence,
    )

    logger.log(
        camera_name,
        location,
        confidence,
        image_path or "",
        video_path,
        bool(alert_sent),
    )


def _process_frame(
    frame,
    conf,
    model,
    camera_name,
    location,
    tracking=False,
    tracker=None,
):
    frame = cv2.resize(frame, (720, int(720 * 9 / 16)))

    result = _run_detection(
        model,
        frame,
        conf,
        tracking,
        tracker,
    )

    plotted = result.plot()

    _write_active_recording(camera_name, frame)

    if _has_drowning(result):
        _handle_drowning(frame, result, camera_name, location)

    return plotted


def _process_capture(
    source,
    conf,
    model,
    camera_name,
    location,
    tracking=False,
    tracker=None,
):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.sidebar.error("Unable to open video source.")
        return

    st_frame = st.empty()

    try:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            plotted = _process_frame(
                frame,
                conf,
                model,
                camera_name,
                location,
                tracking,
                tracker,
            )

            st_frame.image(
                plotted,
                caption="Detected Video",
                channels="BGR",
                use_container_width=True,
            )

    except Exception as exc:
        st.sidebar.error(f"Error processing stream: {exc}")

    finally:
        cap.release()


def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox(
        "Choose a video...",
        settings.VIDEOS_DICT.keys(),
    )

    video_path = settings.VIDEOS_DICT.get(source_vid)

    if video_path and Path(video_path).exists():
        with open(video_path, "rb") as video_file:
            st.video(video_file.read())
    else:
        st.warning("Selected video file was not found.")
        return

    tracking, tracker = display_tracker_options()

    if st.sidebar.button("Detect Drowning"):
        _process_capture(
            str(video_path),
            conf,
            model,
            camera_name=source_vid,
            location="Stored Video",
            tracking=tracking,
            tracker=tracker,
        )


def play_webcam(conf, model):
    tracking, tracker = display_tracker_options()

    if st.sidebar.button("Detect Drowning"):
        _process_capture(
            settings.WEBCAM_PATH,
            conf,
            model,
            camera_name="Webcam",
            location="Local Webcam",
            tracking=tracking,
            tracker=tracker,
        )


def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")

    st.sidebar.caption(
        "Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101"
    )

    tracking, tracker = display_tracker_options()

    if st.sidebar.button("Detect Drowning"):
        if not source_rtsp:
            st.sidebar.error("Please enter an RTSP URL.")
            return

        _process_capture(
            source_rtsp,
            conf,
            model,
            camera_name="RTSP",
            location=source_rtsp,
            tracking=tracking,
            tracker=tracker,
        )


def _resolve_youtube_url(source_youtube):
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source_youtube, download=False)

    return info.get("url") or info.get("formats", [{}])[-1].get("url")


def play_youtube_video(conf, model):
    st.sidebar.write("example link 1: https://youtu.be/3F6GsMESbDc")
    source_youtube = st.sidebar.text_input("YouTube Video url")

    tracking, tracker = display_tracker_options()

    if st.sidebar.button("Detect Drowning"):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL.")
            return

        try:
            video_url = _resolve_youtube_url(source_youtube)

            if not video_url:
                st.sidebar.error("Unable to resolve YouTube video URL.")
                return

            parsed = urlparse(source_youtube)
            location = parsed.netloc or "YouTube"

            _process_capture(
                video_url,
                conf,
                model,
                camera_name="YouTube",
                location=location,
                tracking=tracking,
                tracker=tracker,
            )

        except Exception as exc:
            st.sidebar.error(f"Error loading YouTube video: {exc}")


def _get_camera_manager():
    config_path = str(settings.CAMERA_CONFIG)

    if st.session_state.get("camera_config_path") != config_path:
        old_manager = st.session_state.get("camera_manager")

        if old_manager:
            old_manager.stop()

        st.session_state.camera_manager = CameraManager(config_path)
        st.session_state.camera_config_path = config_path

    return st.session_state.camera_manager


def play_multi_camera(conf, model, dashboard):
    manager = _get_camera_manager()

    st.header("Multi Camera Monitoring")

    c1, c2, c3 = st.columns([1, 1, 6])

    with c1:
        if st.button("Start"):
            st.session_state.multi_camera_running = True

    with c2:
        if st.button("Stop"):
            st.session_state.multi_camera_running = False

    with c3:
        if st.button("Refresh Cameras"):
            manager.reload()

    if "multi_camera_running" not in st.session_state:
        st.session_state.multi_camera_running = True

    cameras = manager.get_frames()
    processed = []

    for camera in cameras:
        frame = camera.get("frame")

        if frame is None:
            camera["drowning"] = False
            camera["confidence"] = 0.0
            processed.append(camera)
            continue

        camera_name = camera.get("name", "Camera")
        location = camera.get("location", "Unknown")

        try:
            result = _run_detection(
                model,
                frame,
                conf,
                tracking=False,
            )

            camera["frame"] = result.plot()
            camera["drowning"] = _has_drowning(result)
            camera["confidence"] = _best_confidence(result)

            _write_active_recording(camera_name, frame)

            if camera["drowning"]:
                _handle_drowning(frame, result, camera_name, location)

        except Exception as exc:
            camera["online"] = False
            camera["error"] = str(exc)

        processed.append(camera)

    dashboard.statistics(processed)
    dashboard.draw(processed)

    if st.session_state.multi_camera_running:
        time.sleep(0.15)
        st.rerun()