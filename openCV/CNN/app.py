# app_plate_only.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
from ultralytics import YOLO
import easyocr
import os

st.set_page_config(page_title="ANPR Number Plate Only", layout="wide")
st.title("🚗 License Plate Recognition (Number Plate Only)")

################################################################################
# Sidebar: Settings & Input
################################################################################
st.sidebar.header("Settings")

model_path = st.sidebar.text_input("YOLO Plate model path", "plate_detector.pt")
confidence = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.4)
ocr_allowlist = st.sidebar.text_input("OCR allowlist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

source_type = st.sidebar.radio(
    "Select input type",
    ["DroidCam (live feed)", "Upload video", "Upload image"]
)

droidcam_url = ""
uploaded_video = None
uploaded_image = None
if source_type == "DroidCam (live feed)":
    droidcam_url = st.sidebar.text_input("Enter DroidCam URL", "http://192.168.0.105:4747/video")
elif source_type == "Upload video":
    uploaded_video = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])
elif source_type == "Upload image":
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

################################################################################
# Load models
################################################################################
@st.cache_resource
def load_models(path):
    if path and os.path.exists(path):
        model = YOLO(path)
    else:
        model = YOLO("yolov8n.pt")  # fallback generic YOLO
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

model, reader = load_models(model_path)

if "plates_log" not in st.session_state:
    st.session_state["plates_log"] = []

################################################################################
# Helpers
################################################################################
def preprocess_plate_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def detect_plate_and_recognize(frame, model, reader, conf, allowlist):
    results = model(frame, imgsz=640, conf=conf)
    annotated = frame.copy()
    plates_info = []

    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            # Crop ONLY the plate region
            plate_crop = frame[y1:y2, x1:x2]
            plate_pre = preprocess_plate_for_ocr(plate_crop)

            text = ""
            conf_ocr = 0.0
            try:
                ocr_out = reader.readtext(plate_pre, detail=1, allowlist=allowlist)
                if ocr_out:
                    best = max(ocr_out, key=lambda x: x[2])
                    text, conf_ocr = best[1], best[2]
            except:
                pass

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

            plates_info.append({"text": text, "ocr_conf": conf_ocr})

    return annotated, plates_info

################################################################################
# UI Layout
################################################################################
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Recognized Plates")
    table_placeholder = st.empty()

with col1:
    st.header("Feed")
    start = st.button("Start")
    stop = st.button("Stop")

    if start:
        # --- DroidCam ---
        if source_type == "DroidCam (live feed)":
            cap = cv2.VideoCapture(droidcam_url)
            if not cap.isOpened():
                st.error("Failed to connect to DroidCam. Check URL.")
            else:
                stframe = st.empty()
                while cap.isOpened():
                    if stop: break
                    ret, frame = cap.read()
                    if not ret: break

                    annotated, plates = detect_plate_and_recognize(frame, model, reader, confidence, ocr_allowlist)
                    ts = datetime.utcnow().isoformat()
                    for p in plates:
                        st.session_state["plates_log"].append({
                            "timestamp": ts,
                            "text": p["text"],
                            "ocr_conf": p["ocr_conf"]
                        })

                    stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                    df = pd.DataFrame(st.session_state["plates_log"])
                    table_placeholder.dataframe(df.tail(10))
                    time.sleep(0.03)
                cap.release()
                st.success("Stream stopped")

        # --- Video Upload ---
        elif source_type == "Upload video" and uploaded_video is not None:
            tmp_path = f"uploaded_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_video.read())
            cap = cv2.VideoCapture(tmp_path)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                annotated, plates = detect_plate_and_recognize(frame, model, reader, confidence, ocr_allowlist)
                ts = datetime.utcnow().isoformat()
                for p in plates:
                    st.session_state["plates_log"].append({
                        "timestamp": ts,
                        "text": p["text"],
                        "ocr_conf": p["ocr_conf"]
                    })

                stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                df = pd.DataFrame(st.session_state["plates_log"])
                table_placeholder.dataframe(df.tail(10))
                time.sleep(0.03)
            cap.release()
            st.success("Video processing complete")

        # --- Image Upload ---
        elif source_type == "Upload image" and uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            annotated, plates = detect_plate_and_recognize(frame, model, reader, confidence, ocr_allowlist)
            ts = datetime.utcnow().isoformat()
            for p in plates:
                st.session_state["plates_log"].append({
                    "timestamp": ts,
                    "text": p["text"],
                    "ocr_conf": p["ocr_conf"]
                })

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            df = pd.DataFrame(st.session_state["plates_log"])
            table_placeholder.dataframe(df.tail(10))
            st.success("Image processing complete")

st.markdown("---")
st.markdown("Now detects **only number plates**, not entire vehicles.")
