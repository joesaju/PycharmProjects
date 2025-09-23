# app_plate_detection_fallback.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from ultralytics import YOLO
import easyocr
import re

st.set_page_config(page_title="Plate-only ANPR (YOLO + OpenCV fallback)", layout="wide")
st.title("🔎 License Plate Detection (plate-only) — YOLO + OpenCV fallback")

################################################################################
# Sidebar: settings & inputs
################################################################################
st.sidebar.header("Settings")
# Path to a plate-only YOLO model. If missing, app will use classical OpenCV plate detection.
yolo_plate_path = st.sidebar.text_input("YOLO plate model (plate_detector.pt)", "")
confidence = st.sidebar.slider("Detection confidence (YOLO)", 0.1, 0.9, 0.35)
ocr_allowlist = st.sidebar.text_input("OCR allowlist (characters)", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
min_plate_area = st.sidebar.number_input("Min plate area (pixels)", value=2000)
max_plate_area = st.sidebar.number_input("Max plate area (pixels)", value=200000)

source_type = st.sidebar.radio("Input", ["Sample image", "Upload image", "Upload video", "DroidCam (live)"])
sample_image_path = "/mnt/data/WhatsApp Image 2025-09-22 at 09.58.53_1f658a73.jpg"  # your uploaded image path
uploaded_image = None
uploaded_video = None
droidcam_url = ""

if source_type == "Upload image":
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
elif source_type == "Upload video":
    uploaded_video = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])
elif source_type == "DroidCam (live)":
    droidcam_url = st.sidebar.text_input("DroidCam URL (http://IP:PORT/video)", "http://192.168.0.105:4747/video")

st.sidebar.markdown("---")
st.sidebar.write("Tip: a plate-trained YOLO model (one class: plate) gives best detection. Fallback is classical detection.")

################################################################################
# Load models (YOLO if available, EasyOCR always)
################################################################################
@st.cache_resource
def load_models(yolo_path):
    yolo_model = None
    if yolo_path and os.path.exists(yolo_path):
        try:
            yolo_model = YOLO(yolo_path)
            st.sidebar.success("Loaded YOLO plate model.")
        except Exception as e:
            st.sidebar.error(f"Failed to load YOLO model: {e}")
            yolo_model = None
    else:
        st.sidebar.info("No YOLO plate model provided or file missing. Using OpenCV fallback.")
    # EasyOCR reader (CPU by default)
    reader = easyocr.Reader(["en"], gpu=False)
    return yolo_model, reader

yolo_model, ocr_reader = load_models(yolo_plate_path)

if "plates_log" not in st.session_state:
    st.session_state["plates_log"] = []

################################################################################
# Utilities: image preprocessing, crop deskew, OCR post-process
################################################################################
def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2

def preprocess_for_ocr(plate_bgr):
    # Convert to grayscale, denoise, equalize, threshold -> returns image ready for OCR
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    # Contrast limited adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # resize to decent height
    h, w = gray.shape
    target_h = 100
    scale = target_h / float(h) if h > 0 else 1.0
    resized = cv2.resize(gray, (max(100, int(w*scale)), target_h), interpolation=cv2.INTER_CUBIC)
    # slight blur and adaptive threshold
    blurred = cv2.bilateralFilter(resized, 9, 75, 75)
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th

def ocr_plate_image(plate_img_bgr, reader, allowlist):
    # Preprocess + OCR with allowlist; return best text + confidence
    pre = preprocess_for_ocr(plate_img_bgr)
    try:
        ocr_results = reader.readtext(pre, detail=1, allowlist=allowlist)
    except Exception as e:
        ocr_results = []
    if not ocr_results:
        return "", 0.0
    best = max(ocr_results, key=lambda x: x[2])
    text = best[1].upper()
    conf = float(best[2])
    # Clean basic noise: remove non-allowed chars, replace common confusions
    text = re.sub(r'[^' + re.escape(allowlist) + r']', '', text)
    text = text.replace('O', '0') if re.match(r'^[0-9]{1,}$', text) else text
    return text, conf

################################################################################
# Classical OpenCV plate detector (fallback)
# Idea: edge -> morphology -> contours -> rectangle proportional filters (aspect ratio + area)
################################################################################
def classical_plate_detector(frame_bgr, min_area=2000, max_area=200000):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # reduce noise while keeping edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # edge detection
    edged = cv2.Canny(gray, 30, 200)
    # close gaps with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    # find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        x, y, cw, ch = cv2.boundingRect(approx)
        area = cw * ch
        if area < min_area or area > max_area:
            continue
        aspect = cw / float(ch) if ch > 0 else 0
        # plates typically have wide aspect ratios; accept range ~2 to 7 (adjust per region)
        if 2.0 < aspect < 7.5 and len(approx) >= 4:
            # optionally run another check: mean brightness (plates often lighter background)
            crop = img[y:y+ch, x:x+cw]
            candidates.append((x, y, x+cw, y+ch, area))
    # sort by area desc (largest first)
    candidates = sorted(candidates, key=lambda x: x[4], reverse=True)
    return candidates  # list of bbox tuples

################################################################################
# YOLO plate detector wrapper (if available)
################################################################################
def yolo_plate_detector(frame_bgr, yolo_model, conf=0.35):
    results = yolo_model(frame_bgr, imgsz=640, conf=conf)
    bboxes = []
    if len(results) == 0:
        return bboxes
    r = results[0]
    if hasattr(r, "boxes") and r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf_score = float(box.conf[0]) if box.conf is not None else 0.0
            bboxes.append((x1, y1, x2, y2, conf_score))
    return bboxes

################################################################################
# Single-frame processing pipeline (detect plates -> OCR them only)
################################################################################
def process_frame_plate_only(frame_bgr):
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    plates_found = []

    # 1) Try YOLO plate detector if available
    if yolo_model is not None:
        boxes = yolo_plate_detector(frame_bgr, yolo_model, conf=confidence)
        for (x1, y1, x2, y2, det_conf) in boxes:
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
            plate_crop = frame_bgr[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue
            text, ocr_conf = ocr_plate_image(plate_crop, ocr_reader, ocr_allowlist)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (10, 200, 10), 2)
            label = f"{text} {ocr_conf:.2f}"
            cv2.putText(annotated, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            plates_found.append({"bbox": (x1, y1, x2, y2), "text": text, "ocr_conf": ocr_conf, "det_conf": det_conf})
        # if YOLO found plates, return them (prefer model)
        if len(plates_found) > 0:
            return annotated, plates_found

    # 2) Fallback: classical OpenCV detector
    candidates = classical_plate_detector(frame_bgr, min_area=min_plate_area, max_area=max_plate_area)
    for (x1, y1, x2, y2, area) in candidates:
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
        plate_crop = frame_bgr[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
        text, ocr_conf = ocr_plate_image(plate_crop, ocr_reader, ocr_allowlist)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 160, 255), 2)
        label = f"{text} {ocr_conf:.2f}"
        cv2.putText(annotated, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        plates_found.append({"bbox": (x1, y1, x2, y2), "text": text, "ocr_conf": ocr_conf, "area": area})
    return annotated, plates_found

################################################################################
# UI, streaming, uploads
################################################################################
col1, col2 = st.columns([2, 1])
with col2:
    st.header("Recognized plates log")
    log_placeholder = st.empty()
    if len(st.session_state["plates_log"]) > 0:
        df = pd.DataFrame(st.session_state["plates_log"])
        log_placeholder.dataframe(df.sort_values("timestamp", ascending=False).head(50))
    else:
        log_placeholder.info("No plates recognized yet")

with col1:
    st.header("Viewer")
    start = st.button("Start")
    stop = st.button("Stop")

    if start:
        # choose input
        if source_type == "Sample image":
            if not os.path.exists(sample_image_path):
                st.error(f"Sample image not found at {sample_image_path}")
            else:
                img = cv2.imread(sample_image_path)
                annotated, plates = process_frame_plate_only(img)
                ts = datetime.utcnow().isoformat()
                for p in plates:
                    st.session_state["plates_log"].append({"timestamp": ts, "text": p["text"], "ocr_conf": p.get("ocr_conf",0.0)})
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                st.success("Done (sample image)")

        elif source_type == "Upload image" and uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, plates = process_frame_plate_only(img)
            ts = datetime.utcnow().isoformat()
            for p in plates:
                st.session_state["plates_log"].append({"timestamp": ts, "text": p["text"], "ocr_conf": p.get("ocr_conf",0.0)})
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            st.success("Done (uploaded image)")

        elif source_type == "Upload video" and uploaded_video is not None:
            tmp_path = f"uploaded_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_video.read())
            cap = cv2.VideoCapture(tmp_path)
            stframe = st.empty()
            while cap.isOpened():
                if stop: break
                ret, frame = cap.read()
                if not ret: break
                annotated, plates = process_frame_plate_only(frame)
                ts = datetime.utcnow().isoformat()
                for p in plates:
                    st.session_state["plates_log"].append({"timestamp": ts, "text": p["text"], "ocr_conf": p.get("ocr_conf",0.0)})
                stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                # update log view
                df = pd.DataFrame(st.session_state["plates_log"])
                log_placeholder.dataframe(df.sort_values("timestamp", ascending=False).head(50))
                time.sleep(0.03)
            cap.release()
            st.success("Video processing complete")

        elif source_type == "DroidCam (live)":
            cap = cv2.VideoCapture(droidcam_url)
            if not cap.isOpened():
                st.error("Failed to open DroidCam URL. Check IP, port, app.")
            else:
                stframe = st.empty()
                while cap.isOpened():
                    if stop: break
                    ret, frame = cap.read()
                    if not ret: break
                    annotated, plates = process_frame_plate_only(frame)
                    ts = datetime.utcnow().isoformat()
                    for p in plates:
                        st.session_state["plates_log"].append({"timestamp": ts, "text": p["text"], "ocr_conf": p.get("ocr_conf",0.0)})
                    stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                    df = pd.DataFrame(st.session_state["plates_log"])
                    log_placeholder.dataframe(df.sort_values("timestamp", ascending=False).head(50))
                    time.sleep(0.03)
                cap.release()
                st.success("DroidCam stopped")

st.markdown("---")
st.markdown("""
**Notes & next steps (honest):**
- If you want high accuracy, **train a YOLOv8 plate detector** on plates from your region (fonts, sizes). The YOLO path box in the sidebar accepts such a model file.
- The OpenCV fallback works well on many photos but may fail on heavily occluded or extremely small plates.
- Tweak `min/max plate area` and aspect ratio checks if your camera crops or image sizes differ.
- Add deduplication (ignore same plate seen within N seconds) to reduce repeated logs.
""")
