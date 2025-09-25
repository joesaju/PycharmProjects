"""
Vehicle License Plate Recognition (single-file Streamlit app)
Features:
 - Upload Image / Upload Video / Webcam / DroidCam (IP stream)
 - Uses custom YOLO model if you upload one (recommended), otherwise OpenCV contour detection fallback
 - OCR by EasyOCR (default) or pytesseract fallback
 - Validates recognized text against configurable license-plate regex (default: India)
 - Shows only the plate region (cropped) and overlays bounding box on the frame
 - Logs recognized plates with timestamp and confidence into an in-app table

Save this file as app.py and run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import os
import io
import re
from datetime import datetime
import pandas as pd

# Optional heavy imports: load lazily
YOLO_AVAILABLE = False
EASYOCR_AVAILABLE = False
PYTESSERACT_AVAILABLE = False

# Try import optional libs on demand in functions to avoid import-time errors
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# --------- Utility & detection functions ---------
@st.cache_resource
def load_easyocr_reader(lang_list=["en"]):
    global EASYOCR_AVAILABLE
    try:
        import easyocr
        reader = easyocr.Reader(lang_list, gpu=False)  # set gpu=True if you have a GPU and want faster OCR
        EASYOCR_AVAILABLE = True
        return reader
    except Exception as e:
        EASYOCR_AVAILABLE = False
        st.warning("easyocr not available: %s" % e)
        return None

@st.cache_resource
def load_yolo(path=None):
    """
    Load an Ultralytics YOLO model from a .pt or .onnx file if provided.
    If no path, returns None.
    """
    global YOLO_AVAILABLE
    if not path:
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        YOLO_AVAILABLE = True
        return model
    except Exception as e:
        YOLO_AVAILABLE = False
        st.warning(f"Could not load YOLO model: {e}")
        return None

def normalize_text(t: str):
    if t is None:
        return ""
    # uppercase, remove common non-alnum separators except keep spaces
    return re.sub(r'[^A-Z0-9 ]', '', t.upper()).strip()

def validate_plate_text(text: str, country="IN"):
    """
    Validate recognized plate against regex patterns.
    Default country = "IN" (India). If you want another country's format, pass custom regex in the UI.
    Returns (is_valid, matched_pattern or None)
    """
    if not text:
        return False, None
    t = normalize_text(text)
    # default patterns for India (flexible)
    patterns = []
    # Common full form: KA 01 AB 1234
    patterns.append(r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}$')
    # Without spaces: KA01AB1234
    patterns.append(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$')
    # flexible: 2 letters + digits + letters + digits (1-4)
    patterns.append(r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{1,4}$')
    # fallback generic (4-10 alnum)
    patterns.append(r'^[A-Z0-9]{4,10}$')

    for p in patterns:
        if re.match(p, t):
            return True, p
    return False, None

def draw_bbox(img, box, label=None, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def opencv_plate_detector(image_bgr, debug=False):
    """
    Detect plate using OpenCV heuristic (edges, contours, aspect-ratio).
    Returns (plate_crop_bgr, bbox) or (None, None) if not found.
    bbox = (x1,y1,x2,y2)
    Note: heuristic — best for relatively frontal, clean plates; less reliable than a trained detector.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Noise removal & contrast
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    edged = cv2.Canny(gray, 30, 200)
    # Close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    plate_candidate = None
    bbox = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        if len(approx) >= 4:
            x,y,wc,hc = cv2.boundingRect(approx)
            aspect_ratio = wc / float(hc) if hc>0 else 0
            area = wc*hc
            if debug:
                print("cnt approx len", len(approx), "ar", aspect_ratio, "area", area)
            # Plate aspect ratio heuristic: typically between ~2 and ~6 (varies)
            if 2.0 <= aspect_ratio <= 6.5 and area > 1000 and wc > 60 and hc > 15:
                # choose this candidate
                plate_candidate = img[y:y+hc, x:x+wc]
                bbox = (x, y, x+wc, y+hc)
                break
    if plate_candidate is None:
        return None, None
    # Optional refinement: straighten via thresholding
    return plate_candidate, bbox

def yolo_plate_detector(yolo_model, image_bgr, conf_thres=0.3):
    """
    Run YOLO (ultralytics) model on image. Expect the model to be trained to detect license plates.
    Returns list of detections as tuples (bbox, score, cls_name) where bbox=(x1,y1,x2,y2).
    """
    # Model expects RGB (Ultralytics usually handles BGR->RGB internally but safer to convert)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # run inference
    results = yolo_model.predict(source=img_rgb, conf=conf_thres, verbose=False)
    detections = []
    # results is a list; we use the first (image) result
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
            cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            # Try to get cls name if model has names
            cls_name = None
            try:
                cls_name = yolo_model.names[cls_id]
            except Exception:
                cls_name = str(cls_id)
            detections.append(((int(x1), int(y1), int(x2), int(y2)), conf, cls_name))
    return detections

def ocr_easyocr(reader, plate_image_bgr):
    """
    Use easyocr to OCR the plate image. Returns (text, confidence)
    """
    img_rgb = cv2.cvtColor(plate_image_bgr, cv2.COLOR_BGR2RGB)
    try:
        res = reader.readtext(img_rgb)
    except Exception as e:
        st.warning(f"easyocr failed: {e}")
        return "", 0.0
    if not res:
        return "", 0.0
    # res is list of (bbox, text, conf)
    # join texts ordered by bbox x
    res_sorted = sorted(res, key=lambda x: x[0][0][0])
    texts = [r[1] for r in res_sorted]
    confs = [r[2] for r in res_sorted]
    text = " ".join(texts).strip()
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return text, avg_conf

def ocr_pytesseract(plate_image_bgr):
    """
    Use pytesseract to OCR plate region. Returns (text, confidence). Note: pytesseract confidence parsing is hacky.
    """
    if not PYTESSERACT_AVAILABLE:
        return "", 0.0
    gray = cv2.cvtColor(plate_image_bgr, cv2.COLOR_BGR2GRAY)
    # Resize for better OCR
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(th)
    custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    try:
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=custom_config)
        text = " ".join([w for w in data['text'] if w.strip() != ""]).strip()
        # average confidence
        confs = [int(c) for c in data['conf'] if c.isdigit() or (isinstance(c, (int, float)))]
        avg_conf = float(np.mean(confs))/100.0 if confs else 0.0
        return text, avg_conf
    except Exception as e:
        st.warning(f"pytesseract error: {e}")
        return "", 0.0

# --------- Streamlit UI ---------
st.set_page_config(page_title="ANPR (License Plate Recognition)", layout="wide")
st.title("🚗 Vehicle License Plate Recognition (ANPR)")

st.markdown("""
**What this app does**
- Detect license plates (YOLO if you supply a model, otherwise OpenCV heuristic)
- OCR the plate (EasyOCR preferred)
- Validate recognized text against a license-plate regex (default: India)
- Supports: Image upload, Video upload, Webcam, DroidCam/IP camera
""")

# Sidebar: model & OCR settings
st.sidebar.header("Model & OCR settings")
uploaded_model = st.sidebar.file_uploader("Upload YOLO model (.pt/.onnx) — optional (better accuracy)", type=["pt","onnx"])
use_yolo = False
yolo_model = None
if uploaded_model is not None:
    # Save to temp path
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_model.name)[1])
    tfile.write(uploaded_model.getbuffer())
    tfile.flush()
    tfile_path = tfile.name
    st.sidebar.success(f"Saved model to {tfile_path}")
    yolo_model = load_yolo(tfile_path)
    if yolo_model is not None:
        use_yolo = True

ocr_choice = st.sidebar.selectbox("OCR engine", ["EasyOCR (recommended)", "pytesseract (system must have Tesseract installed)"])
if ocr_choice.startswith("EasyOCR"):
    reader = load_easyocr_reader(["en"])
else:
    reader = None
    if not PYTESSERACT_AVAILABLE:
        st.sidebar.warning("pytesseract not found in Python environment. Install or switch to EasyOCR.")

country_choice = st.sidebar.selectbox("Plate format template", ["India (default)", "Custom regex"])
custom_regex = None
if country_choice == "Custom regex":
    custom_regex = st.sidebar.text_input("Enter custom regex (use uppercase A-Z, 0-9). Example: ^[A-Z0-9]{4,10}$")

# Modes: Image / Video / Webcam / DroidCam
mode = st.selectbox("Input mode", ["Image", "Video", "Webcam", "DroidCam (IP camera)"])

# Placeholder for output image / video frame
output_placeholder = st.empty()

# Log of detected plates in this session
if 'detections' not in st.session_state:
    st.session_state['detections'] = []

# Helpers for logging
def log_detection(text, conf, bbox, source="image"):
    st.session_state['detections'].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plate": text,
        "confidence": round(conf, 3),
        "bbox": bbox,
        "source": source
    })

# Process single image
def process_image_file(image_bytes, source_label="upload"):
    # Read into OpenCV
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig = img.copy()
    detected_plate_crop = None
    detected_bbox = None
    detection_conf = 0.0
    detection_label = None

    # Try YOLO first if available
    if yolo_model:
        try:
            detections = yolo_plate_detector(yolo_model, img, conf_thres=0.25)
            if detections:
                # choose highest conf
                detections = sorted(detections, key=lambda x: x[1], reverse=True)
                bbox, conf, cls_name = detections[0]
                detected_bbox = bbox
                x1,y1,x2,y2 = bbox
                # ensure bbox within image bounds
                x1=max(0,x1); y1=max(0,y1); x2=min(img.shape[1]-1,x2); y2=min(img.shape[0]-1,y2)
                detected_plate_crop = img[y1:y2, x1:x2].copy()
                detection_conf = conf
                detection_label = cls_name
        except Exception as e:
            st.warning(f"YOLO detection failed: {e}")

    # Fallback to OpenCV contour detector
    if detected_plate_crop is None:
        plate_crop, bbox = opencv_plate_detector(img)
        if plate_crop is not None:
            detected_plate_crop = plate_crop
            detected_bbox = bbox
            detection_conf = 0.5  # heuristic
            detection_label = "opencv_plate"

    # If we found a plate region, OCR it
    recognized_text = ""
    ocr_conf = 0.0
    if detected_plate_crop is not None:
        # Preprocess plate crop before OCR (increase contrast)
        plate_gray = cv2.cvtColor(detected_plate_crop, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Maybe apply adaptive thresholding
        _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Use selected OCR
        if ocr_choice.startswith("EasyOCR") and reader is not None:
            try:
                recognized_text, ocr_conf = ocr_easyocr(reader, detected_plate_crop)
            except Exception as e:
                st.warning(f"EasyOCR error: {e}")
                recognized_text, ocr_conf = "", 0.0
        elif PYTESSERACT_AVAILABLE:
            try:
                recognized_text, ocr_conf = ocr_pytesseract(detected_plate_crop)
            except Exception as e:
                st.warning(f"pytesseract error: {e}")
                recognized_text, ocr_conf = "", 0.0
        else:
            st.warning("No OCR engine available. Install EasyOCR or pytesseract.")

        normalized = normalize_text(recognized_text)
        # Validate
        if custom_regex:
            valid = bool(re.match(custom_regex, normalized))
            matched = custom_regex if valid else None
        else:
            valid, matched = validate_plate_text(normalized, country="IN")
        # Log detection
        log_detection(normalized, ocr_conf if ocr_conf else detection_conf, detected_bbox, source=source_label)

        # Draw bbox on original image
        if detected_bbox:
            draw_bbox(orig, detected_bbox, label=f"{normalized} ({round(ocr_conf,3)})")

    # Display
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Frame / Image")
        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Detected plate (crop)")
        if detected_plate_crop is not None:
            st.image(cv2.cvtColor(detected_plate_crop, cv2.COLOR_BGR2RGB))
            st.write("OCR text:", normalize_text(recognized_text))
            st.write("OCR conf (est):", round(ocr_conf, 3))
            st.write("Valid format:", valid, " (pattern: %s)" % (matched if matched else "None"))
        else:
            st.write("No plate detected.")
    return

# Process video frames from capture (webcam, ip cam, or uploaded video)
def process_video_stream(cap, max_frames=10000, source_label="stream"):
    stframe = st.empty()
    info_area = st.empty()
    # Show controls
    stop_button = st.button("Stop stream")
    frame_skip = st.sidebar.number_input("Process every N frames (larger -> faster)", min_value=1, max_value=30, value=3, step=1)
    processed_frames = 0
    last_plate = None
    # Loop
    while cap.isOpened():
        if stop_button:
            break
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1
        display_frame = frame.copy()
        h,w = frame.shape[:2]
        # Process every Nth frame
        if processed_frames % frame_skip == 0:
            detected_plate_crop = None
            detected_bbox = None
            detection_conf = 0.0
            # YOLO?
            if yolo_model:
                try:
                    dets = yolo_plate_detector(yolo_model, frame, conf_thres=0.25)
                    if dets:
                        dets = sorted(dets, key=lambda x: x[1], reverse=True)
                        bbox, conf, cls = dets[0]
                        x1,y1,x2,y2 = bbox
                        x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
                        detected_plate_crop = frame[y1:y2, x1:x2].copy()
                        detected_bbox = (x1,y1,x2,y2)
                        detection_conf = conf
                except Exception as e:
                    st.warning(f"YOLO inference error: {e}")
            # Fallback
            if detected_plate_crop is None:
                plate, bbox = opencv_plate_detector(frame)
                if plate is not None:
                    detected_plate_crop = plate
                    detected_bbox = bbox
                    detection_conf = 0.45

            if detected_plate_crop is not None:
                # OCR
                if ocr_choice.startswith("EasyOCR") and reader is not None:
                    text, ocr_conf = ocr_easyocr(reader, detected_plate_crop)
                elif PYTESSERACT_AVAILABLE:
                    text, ocr_conf = ocr_pytesseract(detected_plate_crop)
                else:
                    text, ocr_conf = "", 0.0
                normalized = normalize_text(text)
                if custom_regex:
                    valid = bool(re.match(custom_regex, normalized))
                    matched = custom_regex if valid else None
                else:
                    valid, matched = validate_plate_text(normalized, country="IN")
                # Log and annotate if new or repeated with some cooldown
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if normalized and (normalized != last_plate):
                    last_plate = normalized
                    log_detection(normalized, ocr_conf if ocr_conf else detection_conf, detected_bbox, source=source_label)
                # Draw bbox on display frame
                if detected_bbox:
                    draw_bbox(display_frame, detected_bbox, label=f"{normalized} {round(ocr_conf,3)}")
                    # Show plate crop in small overlay
                    ph, pw = detected_plate_crop.shape[:2]
                    # resize small preview
                    preview = cv2.resize(detected_plate_crop, (int(pw*0.6), int(ph*0.6)))
                    # place preview at top-left
                    h_p, w_p = preview.shape[:2]
                    display_frame[5:5+h_p, 5:5+w_p] = preview

        # Show frame in Streamlit
        stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        # show info area
        df = pd.DataFrame(st.session_state['detections']).sort_values("timestamp", ascending=False).head(25)
        info_area.dataframe(df)
       

# UI handlers for each mode
if mode == "Image":
    uploaded_image = st.file_uploader("Upload an image file (jpg/png)", type=["png","jpg","jpeg"])
    if uploaded_image is not None:
        st.info("Processing image...")
        process_image_file(uploaded_image, source_label="image_upload")

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video file (mp4,mov,avi)", type=["mp4","mov","avi","mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        tfile.write(uploaded_video.getbuffer())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        st.info("Starting video processing. Click Stop stream to end.")
        process_video_stream(cap, source_label="uploaded_video")

elif mode == "Webcam":
    st.info("Using local webcam (index 0). Make sure your browser/OS grants camera access if required.")
    start = st.button("Start Webcam")
    if start:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam (index 0). Try different index or use DroidCam stream.")
        else:
            process_video_stream(cap, source_label="webcam")

elif mode == "DroidCam (IP camera)":
    st.info("Enter the video stream URL from your phone's DroidCam / IP webcam app (e.g. http://192.168.1.10:8080/video)")
    stream_url = st.text_input("Stream URL (http/rtsp)")
    start_stream = st.button("Start Stream")
    if start_stream and stream_url:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            st.error("Cannot open stream. Check URL and that your phone & PC are on the same network.")
        else:
            process_video_stream(cap, source_label="ip_stream")

# Show detection log and CSV export
st.header("Detections log")
if st.session_state['detections']:
    df = pd.DataFrame(st.session_state['detections']).sort_values("timestamp", ascending=False)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download log as CSV", data=csv, file_name="plate_detections.csv", mime="text/csv")
else:
    st.write("No detections yet.")

# Small help / tips
st.markdown("---")
st.markdown("**Tips to improve accuracy (production):**")
st.markdown("""
- Train / fine-tune a YOLOv8 (or YOLOv5/YOLOX) model specifically on your target-region plates — this is the single best improvement.
- Use EasyOCR for many languages and good accuracy. pytesseract can work but requires system Tesseract and tuning.
- Provide higher-resolution frames and frontal views of plates.
- Adjust regex for your country's plate format (use the Custom regex option).
- For heavy production usage, run on a GPU and keep inference frame rate moderate (process every N frames).
""")

st.markdown("**Notes:**")
st.markdown("""
- If you upload a YOLO model, it should be trained to detect license plates (one class). The app expects a model that returns bounding boxes around plates.
- The OpenCV fallback is heuristic-based and works best on relatively clean, frontal images.
- If you choose pytesseract but Tesseract is not installed on your system, OCR will fail. On Linux install: `sudo apt-get install tesseract-ocr`.
""")
