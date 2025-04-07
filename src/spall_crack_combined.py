from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def detect_crack_and_spall(image_path, crack_model, spalling_model):
    image_og = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)
    H, W = image_og.shape[:2] 

    #predict with crack model and get mask
    crack_mask = np.zeros((H, W), dtype=np.uint8)
    crack_result = crack_model.predict(image_og, verbose=False)[0]
    if crack_result.masks is not None:
        crack_mask = np.zeros(image_og.shape[:2], dtype=np.uint8)
        for m in crack_result.masks.data:
            mask = m.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            crack_mask[mask_resized > 127] = 1  # Class 1 = crack

    spall_mask = np.zeros((H, W), dtype=np.uint8)
    spalling_result = spalling_model.predict(image_og, verbose=False)[0]
    if spalling_result.masks is not None:
        spall_mask = np.zeros(image_og  .shape[:2], dtype=np.uint8)
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

crack_model = YOLO("crack_segmentation_model_01.pt")      
spalling_model = YOLO("spalling_segmentation_model_01.pt") 
image_path = "IMG_9775.jpg"
detect_crack_and_spall(image_path, crack_model, spalling_model)