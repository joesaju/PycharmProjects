import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import easyocr
from PIL import Image
import tempfile
import os
import time

st.set_page_config(page_title="YOLO + OCR License Plate Recognition", layout="wide")

# ---------- Config ----------
YOLO_MODEL_PATH = r"D:\python\New folder\best_lp.pt"  # change if your model file is elsewhere
USE_EASYOCR_FALLBACK = True
TESSERACT_LANG = 'eng'  # change/add languages if you installed them
OCR_PSM = 7  # tesseract psm
# ----------------------------

# Load models (lazy load)
@st.cache_resource(show_spinner=False)
def load_yolo_model(path):
    model = YOLO(path)
    return model

@st.cache_resource(show_spinner=False)
def load_easyocr():
    return easyocr.Reader(['en'], gpu=False)  # set gpu=True if available

# helper: preprocess crop for OCR
def preprocess_plate(img):
    # img: BGR numpy array cropped plate
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize to improve OCR
    h, w = gray.shape
    scale = max(1, int(200 / max(h, w)))
    gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # bilateral / adaptive threshold
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# helper: OCR on image using pytesseract then fallback to easyocr
def read_plate_text(img_bgr, easy_reader=None):
    try:
        # pytesseract expects RGB/PIL
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        config = f'--psm {OCR_PSM} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        txt = pytesseract.image_to_string(pil, lang=TESSERACT_LANG, config=config)
        txt = txt.strip()
        if txt:
            return txt
    except Exception as e:
        st.debug(f"pytesseract error: {e}")

    # fallback to easyocr
    if easy_reader is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            result = easy_reader.readtext(gray)
            # join results with confidence threshold
            texts = [t[1] for t in result if t[2] > 0.3]
            if texts:
                return " ".join(texts)
        except Exception as e:
            st.debug(f"easyocr error: {e}")

    return ""

# Draw bounding boxes and labels on image
def annotate_frame(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4] if len(det) > 4 else None
        label = det[5] if len(det) > 5 else "plate"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{label} {conf:.2f}" if conf is not None else label
        cv2.putText(frame, text, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame

# Process one image or frame: return annotated image and list of (crop, text)
def process_frame(frame_bgr, model, easy_reader=None, conf_thresh=0.25):
    results = model(frame_bgr, imgsz=1280)  # ultralytics returns a Results list
    plates = []
    annotated = frame_bgr.copy()
    for r in results:
        boxes = r.boxes  # Boxes object
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
            if conf < conf_thresh:
                continue
            xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
            # class name if available
            cls_name = ""
            try:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
            except:
                cls_name = "plate"
            x1, y1, x2, y2 = map(int, xyxy)
            # clip coords
            h, w = frame_bgr.shape[:2]
            x1, x2 = max(0,x1), min(w, x2)
            y1, y2 = max(0,y1), min(h, y2)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            prep = preprocess_plate(crop)
            # try tesseract on preprocessed
            # convert preprocessed back to BGR for OCR function which expects color input
            prep_bgr = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)
            text = read_plate_text(prep_bgr, easy_reader if USE_EASYOCR_FALLBACK else None)
            plates.append({'coords': (x1,y1,x2,y2), 'conf':conf, 'class':cls_name, 'crop':crop, 'text':text})
            # draw annotation
            label = text if text else cls_name
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return annotated, plates

# ---------- Streamlit UI ----------
st.title("🚗 YOLO + OCR — License Plate Recognition (DroidCam / Uploads)")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Input options")
    stream_url = st.text_input("DroidCam / IP camera URL or webcam index (e.g. http://192.168.1.10:4747/video or 0)", value="0")
    use_stream = st.checkbox("Use live stream / camera", value=True)
    uploaded_file = st.file_uploader("Upload image or video", type=['jpg','jpeg','png','mp4','mov','avi'], accept_multiple_files=False)
    conf_thresh = st.slider("YOLO confidence threshold", 0.1, 0.9, 0.25, 0.05)
    run_button = st.button("Start / Process")
    st.markdown("**Notes:**\n- Provide your YOLO LP model at `models/best_lp.pt` or change path in code.\n- Install Tesseract separately (system package).")
    st.write("Tesseract text extraction settings: PSM=", OCR_PSM)

with col2:
    st.header("Output")
    output_placeholder = st.empty()

# Load models when needed
if run_button:
    if not os.path.exists(YOLO_MODEL_PATH):
        st.error(f"YOLO model not found at {YOLO_MODEL_PATH}. Please place your license-plate model there.")
    else:
        model = load_yolo_model(YOLO_MODEL_PATH)
        easy_reader = load_easyocr() if USE_EASYOCR_FALLBACK else None

        # Case A: Uploaded image or video
        if uploaded_file is not None and not use_stream:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.flush()
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                frame = cv2.imdecode(np.fromfile(tfile.name, dtype=np.uint8), cv2.IMREAD_COLOR)
                annotated, plates = process_frame(frame, model, easy_reader, conf_thresh)
                # show annotated
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                output_placeholder.image(annotated_rgb, caption="Annotated", use_column_width=True)
                st.write("Detected plates:")
                for i,p in enumerate(plates):
                    st.write(f"**Plate {i+1}:** text=`{p['text']}` conf={p['conf']:.2f}")
                    st.image(cv2.cvtColor(p['crop'], cv2.COLOR_BGR2RGB), width=300)
            else:
                # treat as video
                cap = cv2.VideoCapture(tfile.name)
                play = st.checkbox("Play processed video (streamed frames)")
                stframe = st.empty()
                results_list = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated, plates = process_frame(frame, model, easy_reader, conf_thresh)
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    stframe.image(rgb, channels="RGB", use_column_width=True)
                    # accumulate plates (first occurrences)
                    for p in plates:
                        results_list.append(p)
                    # small delay so UI is responsive
                    time.sleep(0.01)
                cap.release()
                st.write("Summary of detected plates (last frame captures):")
                for i,p in enumerate(results_list[-10:]):
                    st.write(f"{i+1}. {p['text']} (conf {p['conf']:.2f})")
        else:
            # Case B: Live stream / DroidCam or webcam
            # parse stream_url: if numeric, use webcam index
            try:
                src = int(stream_url)
            except:
                src = stream_url  # e.g. http://ip:port/video
            stframe = output_placeholder.empty()
            stop_stream = st.button("Stop stream")
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                st.error("Unable to open stream. Check URL / webcam index and make sure DroidCam stream is running and accessible from this machine.")
            else:
                st.info("Streaming... press 'Stop stream' to end.")
                while cap.isOpened():
                    if stop_stream:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("No frame grabbed — check the camera or stream. Retrying...")
                        time.sleep(0.5)
                        continue
                    annotated, plates = process_frame(frame, model, easy_reader, conf_thresh)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated_rgb, channels="RGB", use_column_width=True)
                    # show detections as small boxes + text below
                    if plates:
                        with st.expander("Detected plates (latest frame)"):
                            for i,p in enumerate(plates):
                                st.write(f"**Plate {i+1}:** text=`{p['text']}` conf={p['conf']:.2f}")
                                st.image(cv2.cvtColor(p['crop'], cv2.COLOR_BGR2RGB), width=260)
                    # small sleep to make UI responsive
                    time.sleep(0.02)
                cap.release()
                st.success("Stream stopped.")
