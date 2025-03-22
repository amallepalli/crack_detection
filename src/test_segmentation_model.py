from ultralytics import YOLO
import cv2

model = YOLO("path to model")

results = model.predict("path to image", save=True, show=True)

cv2.waitKey(0)  
cv2.destroyAllWindows()
