from spall_crack_combined import detect_crack_and_spall
from ultralytics import YOLO
import cv2
from threading import Thread
from queue import Queue

crack_model = YOLO("C:/Users/adity/Projects/FTR Research/crack_segmentation_model_02.pt")      
spalling_model = YOLO("C:/Users/adity/Projects/FTR Research/spalling_segmentation_model_01.pt")
crack_thresh = 0
spalling_thresh = 0.2

cap = cv2.VideoCapture("crack_vid_test2.mp4")
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    'output3.mp4', 
    fourcc, 
    fps, 
    (frame_width, frame_height)
)
if not cap.isOpened():
    raise RuntimeError("Could not open video file")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run your two‐model overlay
    out_rgb = detect_crack_and_spall(frame, crack_model, spalling_model, crack_thresh, spalling_thresh)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    writer.write(out_bgr)
    cv2.imshow("Overlay", out_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()



