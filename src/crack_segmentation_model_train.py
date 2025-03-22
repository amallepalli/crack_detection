from ultralytics import YOLO

model = YOLO('YOLO11m-seg.pt')

results = model.train(data="crack-seg.yaml", epochs=50, imgsz=640, batch=8)
