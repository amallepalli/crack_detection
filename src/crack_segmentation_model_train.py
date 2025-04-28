from ultralytics import YOLO
def main():
    # Your training code here
    model = YOLO('yolo11l-seg.pt')
    results = model.train(data="C:/Users/adity/Projects/FTR Research/crack_detection/spalling-3/data.yaml", epochs=100, imgsz=640, batch=8)
if __name__ == '__main__':
    main()