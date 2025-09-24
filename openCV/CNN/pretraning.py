from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # start from small backbone for quick experiments
model.train(data=r"D:\python\New folder\License-Plate-Recognition-10\data.yaml", epochs=100, imgsz=640, batch=8, name='lp_yolov8', augment=True)
# result weights saved in ./runs/detect/lp_yolov8/weights/best.pt
