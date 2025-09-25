import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pytesseract
import tempfile

# Configure Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="YOLOv8 Vehicle License Plate Recognition", layout="wide")
st.title("🚗 YOLOv8 Vehicle License Plate Recognition (ANPR)")

# Load YOLOv8 model (custom-trained for license plates)
model = YOLO("yolov8_lp.pt")  # Replace with your custom-trained model path

stream_source = st.selectbox("Select Stream Source:", ["Webcam", "DroidCam", "Upload Photo", "Upload Video"])

def detect_plate_yolo(frame):
    results = model.predict(frame)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            plate_img = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = pytesseract.image_to_string(plate_img, config='--psm 8')
            cv2.putText(frame, text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame

if stream_source in ["Webcam", "DroidCam"]:
    if stream_source == "Webcam":
        cam_index = 0
    else:
        cam_index = st.text_input("Enter DroidCam IP (e.g., 192.168.1.5:4747/video):")
    run = st.checkbox('Start Livestream')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0 if stream_source=="Webcam" else f"http://{cam_index}/video")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to load camera.")
            break
        frame = detect_plate_yolo(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

elif stream_source == "Upload Photo":
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        result = detect_plate_yolo(image)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

elif stream_source == "Upload Video":
    uploaded_file = st.file_uploader("Choose a Video", type=["mp4", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_plate_yolo(frame)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
