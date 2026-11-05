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
from scene_guard import (
    combine_drowning_and_person_detections,
    correct_dry_scene_predictions,
    remove_conflicting_overlaps,
    suppress_weak_scene_conflicts,
)

DROWNING_CLASS_ID = 0
DROWNING_ALERT_CONFIDENCE = getattr(settings, "DROWNING_ALERT_CONFIDENCE", 0.38)


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ("Yes", "No"))

    if display_tracker == "Yes":
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return True, tracker_type

    return False, None


def play_alarm_audio(file_path, placeholder=None):
    audio_path = Path(file_path)

    if not audio_path.is_absolute():
        audio_path = settings.ROOT / audio_path

    if not audio_path.exists():
        return

    with open(audio_path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode()

    audio_html = f"""
        <audio autoplay="true" loop="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    if placeholder is not None:
        placeholder.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.markdown(audio_html, unsafe_allow_html=True)


def stop_alarm_audio(placeholder=None):
    if placeholder is not None:
        placeholder.empty()


def autoplay_audio(file_path):
    play_alarm_audio(file_path)


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
        getattr(settings, "ALERT_PROVIDER", "WAHA"),
        getattr(settings, "WAHA_BASE_URL", "http://localhost:3000"),
        getattr(settings, "WAHA_SESSION", "default"),
        getattr(settings, "WAHA_PHONE_NUMBER", ""),
        getattr(settings, "WAHA_API_KEY", ""),
        settings.account_sid,
        settings.auth_token,
        settings.from_,
        settings.to_,
        settings.imgbb_api,
        settings.content_sid,
    )

    if st.session_state.get("alert_manager_signature") != signature:
        st.session_state.alert_manager = AlertManager(
            account_sid=settings.account_sid,
            auth_token=settings.auth_token,
            from_number=settings.from_,
            to_number=settings.to_,
            imgbb_api=settings.imgbb_api,
            content_sid=settings.content_sid,
            evidence_dir=str(settings.IMAGE_DIR),
            provider=getattr(settings, "ALERT_PROVIDER", "WAHA"),
            waha_url=getattr(settings, "WAHA_BASE_URL", "http://localhost:3000"),
            waha_session=getattr(settings, "WAHA_SESSION", "default"),
            waha_phone=getattr(settings, "WAHA_PHONE_NUMBER", ""),
            waha_api_key=getattr(settings, "WAHA_API_KEY", ""),
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

    result = correct_dry_scene_predictions(
        results[0],
        frame,
        enabled=settings.DRY_SCENE_GUARD_ENABLED,
        min_fraction=settings.MIN_POOL_WATER_FRACTION,
    )
    result = combine_drowning_and_person_detections(result, frame)
    return suppress_weak_scene_conflicts(remove_conflicting_overlaps(result))


def refine_predictions(result, frame):
    """Apply the shared scene and duplicate-label safeguards."""
    result = correct_dry_scene_predictions(
        result,
        frame,
        enabled=settings.DRY_SCENE_GUARD_ENABLED,
        min_fraction=settings.MIN_POOL_WATER_FRACTION,
    )
    result = combine_drowning_and_person_detections(result, frame)
    return suppress_weak_scene_conflicts(remove_conflicting_overlaps(result))


def _write_active_recording(camera_name, frame):
    recorder = _get_recorder(camera_name)

    if recorder.is_recording():
        recorder.write(frame)


def _get_camera_state(camera_name):
    if "camera_states" not in st.session_state:
        st.session_state.camera_states = {}
    key = _safe_name(camera_name)
    if key not in st.session_state.camera_states:
        st.session_state.camera_states[key] = {
            "consecutive_drowning": 0,
            "consecutive_safe": 0,
            "alarm_active": False,
            "audio_playing": False,
        }
    return st.session_state.camera_states[key]


def _update_incident_state(camera_name, has_drowning, best_conf):
    state = _get_camera_state(camera_name)
    safe_frames_threshold = getattr(settings, "SAFE_DROWNING_FRAMES", 8)
    recovery_frames_threshold = getattr(settings, "SAFE_RECOVERY_FRAMES", 5)

    if has_drowning and best_conf >= DROWNING_ALERT_CONFIDENCE:
        state["consecutive_drowning"] += 1
        state["consecutive_safe"] = 0
        if state["consecutive_drowning"] >= safe_frames_threshold:
            state["alarm_active"] = True
    else:
        state["consecutive_safe"] += 1
        state["consecutive_drowning"] = 0
        if state["consecutive_safe"] >= recovery_frames_threshold:
            state["alarm_active"] = False

    return state


def _handle_drowning(frame, result, camera_name, location, state=None, audio_placeholder=None):
    confidence = _best_confidence(result)

    recorder = _get_recorder(camera_name)

    if not recorder.is_recording():
        recorder.start(frame, camera_name, duration=settings.RECORD_SECONDS)

    recorder.write(frame)
    video_path = recorder.current_path or ""

    # Ensure continuous alarm is sounding while drowning persists
    if state is not None and not state.get("audio_playing", False):
        play_alarm_audio(settings.AUDIO_PATH, placeholder=audio_placeholder)
        state["audio_playing"] = True

    if not _should_alert(camera_name):
        return

    alert_manager = _get_alert_manager()
    logger = _get_logger()

    image_path, alert_sent = alert_manager.trigger(
        frame,
        camera_name,
        location,
        confidence,
    )

    if getattr(alert_manager, "last_uploaded_url", None):
        st.sidebar.success(f"☁️ Snapshot uploaded to ImgBB Cloud: {alert_manager.last_uploaded_url}")

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
    audio_placeholder=None,
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

    has_drowning_in_frame = _has_drowning(result)
    best_conf = _best_confidence(result)
    state = _update_incident_state(camera_name, has_drowning_in_frame, best_conf)

    if state["alarm_active"]:
        _handle_drowning(frame, result, camera_name, location, state, audio_placeholder)
    else:
        # If alarm was active and is now stopped, silence audio
        if state.get("audio_playing", False):
            stop_alarm_audio(audio_placeholder)
            state["audio_playing"] = False

    return plotted, result, state


def _process_capture(
    source,
    conf,
    model,
    camera_name,
    location,
    tracking=False,
    tracker=None,
):
    is_webcam = (source == 0 or source == settings.WEBCAM_PATH)
    if is_webcam:
        # On Windows, cv2.CAP_DSHOW prevents webcam from hanging or freezing
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.sidebar.error("Unable to open video source. If webcam is in use, please close other apps.")
        return

    # Stop button controls
    col_stop, _ = st.columns([2, 5])
    with col_stop:
        stop_stream = st.button("⏹ Stop Detection / Camera", key=f"stop_{camera_name}")

    dashboard_placeholder = st.empty()
    st_frame = st.empty()
    audio_placeholder = st.empty()

    # Reset camera state at start of stream/video run
    state = _get_camera_state(camera_name)
    state["consecutive_drowning"] = 0
    state["consecutive_safe"] = 0
    state["alarm_active"] = False
    state["audio_playing"] = False
    stop_alarm_audio(audio_placeholder)

    try:
        consecutive_failures = 0
        while cap.isOpened():
            if stop_stream:
                break

            success, frame = cap.read()

            if not success or frame is None:
                if is_webcam and consecutive_failures < 10:
                    consecutive_failures += 1
                    time.sleep(0.05)
                    continue
                break

            consecutive_failures = 0

            plotted, result, state = _process_frame(
                frame,
                conf,
                model,
                camera_name,
                location,
                tracking,
                tracker,
                audio_placeholder=audio_placeholder,
            )

            num_persons = len(result.boxes) if result.boxes is not None else 0
            best_conf = _best_confidence(result)
            if state["alarm_active"]:
                dashboard_placeholder.markdown(
                    f"""
                    <div style="background-color: #d32f2f; padding: 14px; border-radius: 8px; color: white; text-align: center; margin-bottom: 12px; border: 2px solid #b71c1c;">
                        <h2 style="margin: 0; color: white;">🚨 ALARM TRIGGERED — DROWNING DETECTED!</h2>
                        <p style="margin: 4px 0 0 0; font-size: 15px;">Source: <b>{camera_name}</b> | Persons: <b>{num_persons}</b> | Confidence: <b>{best_conf * 100:.1f}%</b> | Sustained Frames: <b>{state['consecutive_drowning']}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                dashboard_placeholder.markdown(
                    f"""
                    <div style="background-color: #1b5e20; padding: 10px; border-radius: 8px; color: white; text-align: center; margin-bottom: 12px; border: 1px solid #2e7d32;">
                        <h3 style="margin: 0; color: white;">🟢 STATUS: NORMAL (Monitoring Active)</h3>
                        <p style="margin: 4px 0 0 0; font-size: 14px;">Source: <b>{camera_name}</b> | Detected Persons: <b>{num_persons}</b> | Safe Frames: <b>{state['consecutive_safe']}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
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
        stop_alarm_audio(audio_placeholder)
        state["audio_playing"] = False


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

    audio_placeholder = st.empty()
    cameras = manager.get_frames()
    processed = []
    any_active_alarm = False

    for camera in cameras:
        frame = camera.get("frame")

        if frame is None:
            camera["drowning"] = False
            camera["alarm_triggered"] = False
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
            has_drown = _has_drowning(result)
            best_conf = _best_confidence(result)
            state = _update_incident_state(camera_name, has_drown, best_conf)

            camera["drowning"] = state["alarm_active"]
            camera["alarm_triggered"] = state["alarm_active"]
            camera["confidence"] = best_conf

            _write_active_recording(camera_name, frame)

            if state["alarm_active"]:
                any_active_alarm = True
                _handle_drowning(frame, result, camera_name, location, state, audio_placeholder)

        except Exception as exc:
            camera["online"] = False
            camera["error"] = str(exc)

        processed.append(camera)

    if not any_active_alarm:
        stop_alarm_audio(audio_placeholder)

    dashboard.statistics(processed)
    dashboard.draw(processed)

    if st.session_state.multi_camera_running:
        time.sleep(0.15)
        st.rerun()
