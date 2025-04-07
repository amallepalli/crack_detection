'''
from roboflow import Roboflow
rf = Roboflow(api_key="Zkf96W12ifjZpAXEo72K")
project = rf.workspace("apu-jimbr").project("crack-uuasm")
version = project.version(3)
dataset = version.download("yolov11")
'''
from ultralytics import YOLO
def main():
    # Your training code here
    model = YOLO('yolo11l-seg.pt')
    results = model.train(data="C:/Users/adity/Projects/FTR Research/crack_detection/Crack-3/data.yaml", epochs=150, imgsz=640, batch=8)
if __name__ == '__main__':
    main()