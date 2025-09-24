"""
streamlit_anpr.py
Single-file Streamlit app for ANPR (YOLOv8 detection + EasyOCR / pytesseract OCR)
Features:
 - Live stream (webcam or DroidCam HTTP stream / RTSP / device index)
 - Upload image / upload video processing
 - Save snapshots and a CSV log of recognized plates
 - Basic preprocessing and OCR postprocessing
Notes:
 - For detection you should point "YOLO model path" to a trained plate detector (default: models/best_lp.pt)
 - If no custom model is available app will try to load a generic model (yolov8n.pt) but results will be poor.
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile, time, os, io
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# OCR: prefer easyocr, fallback to pytesseract
try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    _HAS_PYTESSERACT = False

# ---------------------
# Utils & helpers
# ---------------------
st.set_page_config(layout="wide", page_title="ANPR - Streamlit", initial_sidebar_state="expanded")

@st.cache_resource
def load_yolo_model(path: str):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model from {path}: {e}")
        return None

@st.cache_resource
def load_easyocr_reader(lang_list=['en']):
    if not _HAS_EASYOCR:
        return None
    return easyocr.Reader(lang_list, gpu=False)  # change gpu=True if you have CUDA

def preprocess_plate_for_ocr(crop_bgr):
    # returns an RGB numpy image suitable for OCR
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # increase contrast / denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 2)
    # morphological closing to fill gaps
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    # Convert to RGB for OCR libs that expect color arrays
    return cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

def ocr_with_easyocr(reader, image_rgb):
    if reader is None:
        return "", 0.0
    try:
        # easyocr expects RGB images
        results = reader.readtext(image_rgb)
        # choose highest-confidence text
        if not results:
            return "", 0.0
        # results: list of (bbox, text, conf)
        results_sorted = sorted(results, key=lambda r: r[2], reverse=True)
        text = results_sorted[0][1]
        conf = float(results_sorted[0][2])
        return text, conf
    except Exception as e:
        return "", 0.0

def ocr_with_pytesseract(image_rgb):
    if not _HAS_PYTESSERACT:
        return "", 0.0
    try:
        pil = Image.fromarray(image_rgb)
        # restrict chars (digits + uppercase letters) - adjust depending on region
        config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        txt = pytesseract.image_to_string(pil, config=config)
        txt = txt.strip().replace(" ", "")
        # pytesseract doesn't return confidence easily here; return 0.0 as placeholder
        return txt, 0.0
    except Exception:
        return "", 0.0

def normalize_plate_text(raw):
    if not raw:
        return ""
    txt = raw.upper().strip()
    # common corrections
    txt = txt.replace(" ", "")
    txt = txt.replace("O", "0") if any(c.isdigit() for c in txt) else txt
    # Replace similar looking letters/numbers
    mapping = str.maketrans({
        'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'O': '0'
    })
    txt2 = txt.translate(mapping)
    # Basic filter keep A-Z0-9 only
    txt2 = "".join([c for c in txt2 if c.isalnum()])
    return txt2

def annotate_frame(frame_bgr, plates):
    # draw boxes and text
    out = frame_bgr.copy()
    for p in plates:
        x1,y1,x2,y2 = p['box']
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{p['text']} (ocr:{p.get('ocr_conf',0):.2f} det:{p.get('det_conf',0):.2f})"
        cv2.putText(out, label, (x1, max(10,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return out

def parse_yolo_results(results, conf_th=0.25):
    """
    results: ultralytics results[0]
    returns list of boxes [(x1,y1,x2,y2), ...] and conf list
    """
    boxes = []
    try:
        r = results
        if hasattr(r, "boxes") and r.boxes is not None:
            xy = r.boxes.xyxy.cpu().numpy()  # Nx4
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else [0]*len(confs)
            for b, c, cl in zip(xy, confs, cls):
                if float(c) < conf_th:
                    continue
                x1,y1,x2,y2 = [int(max(0, v)) for v in b]
                boxes.append( (x1,y1,x2,y2, float(c), int(cl)) )
    except Exception:
        pass
    return boxes

# ---------------------
# UI
# ---------------------
st.title("🚗 Vehicle License Plate Recognition (ANPR) — Streamlit")
st.markdown("""
**Features:** Live stream (DroidCam/webcam/IP), upload image/video, YOLOv8 plate detection, EasyOCR/Tesseract OCR, preprocessing, logs & snapshots.
""")

# Sidebar controls
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("YOLO model path", value="models/best_lp.pt")
confidence_threshold = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.35, 0.05)
use_easyocr = st.sidebar.checkbox("Use EasyOCR (preferred)", value=_HAS_EASYOCR)
use_pytesseract = st.sidebar.checkbox("Use pytesseract as fallback", value=_HAS_PYTESSERACT)
reader_langs = st.sidebar.text_input("EasyOCR languages (comma)", value="en")
droidcam_example = "Examples: 0 (webcam), http://192.168.43.1:8080/video , rtsp://<ip>/path"
camera_source = st.sidebar.text_input("Camera source (index or URL)", value="0", help=droidcam_example)

if "log" not in st.session_state:
    st.session_state.log = pd.DataFrame(columns=["timestamp", "plate", "det_conf", "ocr_conf", "source"])

if "model" not in st.session_state:
    st.session_state.model = None

if "reader" not in st.session_state:
    st.session_state.reader = None

if st.sidebar.button("Load Model"):
    st.session_state.model = load_yolo_model(model_path)
    if st.session_state.model:
        st.sidebar.success("YOLO model loaded.")
    # load easyocr reader if requested
    if use_easyocr and _HAS_EASYOCR:
        langs = [l.strip() for l in reader_langs.split(",") if l.strip()]
        if len(langs)==0: langs=['en']
        st.session_state.reader = load_easyocr_reader(langs)
        if st.session_state.reader is not None:
            st.sidebar.success("EasyOCR reader ready.")
    elif use_easyocr and not _HAS_EASYOCR:
        st.sidebar.warning("EasyOCR not installed. Install 'easyocr' or uncheck option.")

# lazy load if model path exists and not loaded
if st.session_state.model is None:
    if os.path.exists(model_path):
        st.session_state.model = load_yolo_model(model_path)
        if st.session_state.model:
            st.sidebar.info(f"Auto-loaded model from {model_path}")

# top row: mode selection
mode = st.radio("Mode", ["Live stream", "Upload image", "Upload video"], horizontal=True)

col1, col2 = st.columns([2,1])

with col1:
    display_area = st.empty()
with col2:
    st.subheader("Recognized plates (log)")
    log_area = st.empty()
    st.write("Logs will be saved in session and can be downloaded.")
    if st.button("Download CSV"):
        csv = st.session_state.log.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="anpr_log.csv", mime="text/csv")

# Helpers for saving snapshot
OUTPUT_DIR = "anpr_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_event(plate, det_conf, ocr_conf, frame_rgb):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plate_{plate}_{ts}.jpg"
    path = os.path.join(OUTPUT_DIR, filename)
    Image.fromarray(frame_rgb).save(path)
    new = {"timestamp": datetime.now().isoformat(), "plate": plate, "det_conf": det_conf, "ocr_conf": ocr_conf, "source": path}
    st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame([new])], ignore_index=True)

# ---------------------
# Main workflows
# ---------------------
def process_frame(frame_bgr, model, reader, conf_th):
    plates = []
    # run detection
    try:
        results = model(frame_bgr)[0]  # run single-image inference
    except Exception as e:
        return frame_bgr, plates

    boxes = parse_yolo_results(results, conf_th)

    for (x1, y1, x2, y2, det_conf, cls) in boxes:
        # Ensure detection is a plate (class id == 0 if you trained only "plate")
        # If your model has multiple classes, replace with correct index/name check
        if hasattr(model, "names"):
            class_name = model.names.get(cls, "")
            if class_name.lower() not in ["plate", "license_plate", "number_plate"]:
                continue  # skip non-plate detections

        # crop with small margin
        h, w = frame_bgr.shape[:2]
        pad = int(0.02 * max(w, h))
        x1m = max(0, x1 - pad)
        y1m = max(0, y1 - pad)
        x2m = min(w, x2 + pad)
        y2m = min(h, y2 + pad)
        crop = frame_bgr[y1m:y2m, x1m:x2m]

        # preprocess for OCR
        pre = preprocess_plate_for_ocr(crop)
        text = ""
        oconf = 0.0

        if pre is not None:
            if use_easyocr and reader is not None:
                text, oconf = ocr_with_easyocr(reader, pre)
            if (not text or len(text.strip()) == 0) and use_pytesseract:
                text, oconf2 = ocr_with_pytesseract(pre)
                oconf = max(oconf, oconf2)

        text_norm = normalize_plate_text(text)

        plates.append({
            "box": (x1, y1, x2, y2),
            "text": text_norm,
            "raw_text": text,
            "ocr_conf": oconf,
            "det_conf": det_conf
        })

    annotated = annotate_frame(frame_bgr, plates)
    return annotated, plates


# Mode: Upload Image
if mode == "Upload image":
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)[:,:,::-1].copy()  # rgb->bgr
        if st.session_state.model is None:
            st.warning("YOLO model not loaded. Click 'Load Model' in the sidebar or provide a model at path.")
        annotated, plates = process_frame(frame, st.session_state.model or YOLO('yolov8n.pt'), st.session_state.reader, confidence_threshold)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        display_area.image(annotated_rgb, channels="RGB", use_column_width=True)
        if plates:
            for p in plates:
                st.write(f"Plate: **{p['text']}**  — det:{p['det_conf']:.2f} ocr:{p['ocr_conf']:.2f}")
                if st.button(f"Save snapshot {p['text']}", key=f"save_{p['text']}"):
                    save_event(p['text'], p['det_conf'], p['ocr_conf'], annotated_rgb)
            st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame(
                [{"timestamp": datetime.now().isoformat(), "plate": p['text'], "det_conf": p['det_conf'], "ocr_conf": p['ocr_conf'], "source": "upload_image"} for p in plates]
            )], ignore_index=True)
        else:
            st.info("No plates detected.")

# Mode: Upload Video
elif mode == "Upload video":
    uploaded = st.file_uploader("Upload a video (mp4, avi)", type=["mp4","avi","mov"])
    sample_rate = st.number_input("Process every Nth frame (1 = every frame)", value=5, min_value=1, max_value=60, step=1)
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tfile.write(uploaded.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        stframe = display_area
        stframe.image(np.zeros((480,640,3), dtype=np.uint8))
        frame_count = 0
        pbar = st.progress(0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
            annotated, plates = process_frame(frame, st.session_state.model or YOLO('yolov8n.pt'), st.session_state.reader, confidence_threshold)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, channels="RGB", use_column_width=True)
            for p in plates:
                st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame(
                    [{"timestamp": datetime.now().isoformat(), "plate": p['text'], "det_conf": p['det_conf'], "ocr_conf": p['ocr_conf'], "source": f"video:{uploaded.name}"}]
                )], ignore_index=True)
            idx += 1
            if total>0:
                pbar.progress(min(1.0, frame_count/total))
        cap.release()
        st.success("Video processed. Check logs on the right.")

# Mode: Live stream
else:
    st.markdown("**Live / DroidCam stream** — enter device index (0,1,..) or DroidCam HTTP/RTSP URL.")
    start = st.button("Start stream")
    stop = st.button("Stop stream")
    if start:
        source = camera_source.strip()
        # pick source: numeric index or URL
        try:
            src = int(source)
        except Exception:
            src = source  # url
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            st.error(f"Cannot open camera source: {source}. Double-check DroidCam URL or device index.")
        else:
            st.info("Streaming... click Stop stream to end.")
            frame_slot = display_area
            while cap.isOpened():
                if st.button("End stream (internal)", key="end_internal"):
                    break
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from stream. Stopping.")
                    break
                annotated, plates = process_frame(frame, st.session_state.model or YOLO('yolov8n.pt'), st.session_state.reader, confidence_threshold)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_slot.image(annotated_rgb, channels="RGB", use_column_width=True)
                # append log and optionally save snapshot
                for p in plates:
                    # save snapshot automatically
                    save_event(p['text'], p['det_conf'], p['ocr_conf'], annotated_rgb)
                # throttling small sleep
                time.sleep(0.03)
                # early stop if user pressed Stop on the sidebar
                if stop:
                    break
            cap.release()
            st.success("Stream ended.")

# show logs
log_area.dataframe(st.session_state.log.sort_values("timestamp", ascending=False).reset_index(drop=True).head(200))
