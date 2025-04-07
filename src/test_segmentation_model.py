from ultralytics import YOLO
import cv2
model = YOLO("spalling_segmentation_model_01.pt")
image = cv2.imread("IMG_9775.jpg")
results = model.predict(image, save=True, show=False)
