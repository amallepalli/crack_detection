from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def detect_crack_and_spall(frame, crack_model, spalling_model):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = frame.shape[:2] 

    #predict with crack model and get mask
    crack_mask = np.zeros((H, W), dtype=np.uint8)
    crack_result = crack_model.predict(frame, verbose=False)[0]
    if crack_result.masks is not None:
        crack_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for m in crack_result.masks.data:
            mask = m.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            crack_mask[mask_resized > 127] = 1  # Class 1 = crack

    spall_mask = np.zeros((H, W), dtype=np.uint8)
    spalling_result = spalling_model.predict(frame, verbose=False)[0]
    if spalling_result.masks is not None:
        spall_mask = np.zeros(frame  .shape[:2], dtype=np.uint8)
        for m in spalling_result.masks.data:
            mask = m.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            spall_mask[mask_resized > 127] = 2  # Class 2 = spalling

    # Combine masks
    final_mask = np.zeros((H, W), dtype=np.uint8)
    final_mask[crack_mask == 1] = 1
    final_mask[spall_mask == 2] = 2
    final_mask[(crack_mask == 1) & (spall_mask == 2)] = 3 # Class 3 = Both detections
     
    
    # mask overlay
    output_image = image_rgb.copy()
    mask_overlay = np.zeros_like(image_rgb)
    mask_overlay[final_mask == 1] = [255, 0, 0]     # Red for cracks
    mask_overlay[final_mask == 2] = [0, 255, 0]     # Green for spalling
    mask_overlay[final_mask == 3] = [255, 0, 255]   # Magenta for both

    output_image = cv2.addWeighted(image_rgb, 1.0, mask_overlay, 0.6, 0)
    return output_image

    '''
    # Show result
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Crack + Spalling Segmentation")
    plt.imshow(output_image)
    plt.axis('off')

    legend_elements = [
    Patch(facecolor='red', edgecolor='red', label='Crack'),
    Patch(facecolor='green', edgecolor='green', label='Spalling'),
    Patch(facecolor='magenta', edgecolor='magenta', label='Both Detected'),
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize='small')

    plt.tight_layout()
    plt.show()
    '''
crack_model = YOLO("/Users/amallepalli/Projects/FTR2025/crack_detection/crack_segmentation_model_02.pt")      
spalling_model = YOLO("/Users/amallepalli/Projects/FTR2025/crack_detection/spalling_segmentation_model_01.pt")  


# Setup Video Capture
cap = cv2.VideoCapture("/Users/amallepalli/Downloads/crack_vid.mp4") #0 = default webcam
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
