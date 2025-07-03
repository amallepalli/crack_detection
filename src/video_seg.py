from spall_crack_combined import detect_crack_and_spall
from ultralytics import YOLO
import cv2
from threading import Thread
from queue import Queue

crack_model = YOLO("C:/Users/adity/Projects/FTR Research/crack_segmentation_model_02.pt")      
spalling_model = YOLO("C:/Users/adity/Projects/FTR Research/spalling_segmentation_model_01.pt")
crack_thresh = 0
spalling_thresh = 0


# Setup Video Capture
cap = cv2.VideoCapture(0) #0 = default webcam

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
writer = cv2.VideoWriter(
    'output.mp4', 
    fourcc, 
    fps, 
    (frame_width, frame_height)
)

#Creating fixed sized Queue to hold "to be processed" frames 
in_q = Queue(maxsize=1)
out_q = Queue(maxsize=1)

# capture_loop grabes the frames as fast as cam can deliver
def capture_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # drop older frame if queue is full
        if in_q.full():
            _ = in_q.get_nowait()
        in_q.put(frame)

# infer_loop runs the combine mask logic
def infer_loop():
    while True:
        frame = in_q.get()
        result = detect_crack_and_spall(frame, crack_model, spalling_model, crack_thresh, spalling_thresh)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if out_q.full():
            _ = out_q.get_nowait()
        out_q.put(result)

#Initialize Threads
Thread(target=capture_loop, daemon=True).start()
Thread(target=infer_loop, daemon=True).start()

#Main loop always shows the latest processed frame
while True:
    if not out_q.empty():
        display = out_q.get()
        writer.write(display)
        cv2.imshow("Live Crack+Spall Overlay", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()




'''
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_overlay_2.mp4", fourcc, fps, (w, h))

frame_idx = 0
last_frame = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if last_frame is not None:
        to_write = last_frame
    else:
        to_write = frame


    if frame_idx % 5 == 0:
        result = detect_crack_and_spall(frame, crack_model, spalling_model)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        to_write = result
        last_frame = result

    cv2.imshow("Crack + Spall Overlay", to_write)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    out.write(to_write)
    frame_idx += 1
cap.release()
out.release()
cv2.destroyAllWindows()
'''