from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time
import clip
from PIL import Image
import torch

clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
clip_clases = ["a photo of concrete crack", "a photo of concrete spalling", "a photo of undamaged concrete"]
clip_text = clip.tokenize(clip_clases).to("cuda")
def detect_crack_and_spall_clip(frame,
                           crack_model, spalling_model,
                           crack_thresh, spalling_thresh, clip_thresh):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_image = image_rgb.copy()

    H, W = frame.shape[:2] 

    #predict with crack model and get mask
    crack_mask = np.zeros((H, W), dtype=np.uint8)
    crack_result = crack_model.predict(frame, verbose=False)[0]
    crack_confs = crack_result.boxes.conf.cpu().numpy()                 
    crack_boxes = crack_result.boxes.xyxy.cpu().numpy().astype(int)

    if crack_result.masks is not None:
        for idx, m in enumerate(crack_result.masks.data):
            if crack_confs[idx] < crack_thresh:
                #print("crack made it", crack_confs[idx])
                continue

            # Crop for CLIP
            x1, y1, x2, y2 = crack_boxes[idx]
            crop = image_rgb[y1:y2, x1:x2]
            plt.imshow(crop)
            if crop.size == 0:
                continue
            
            #Clip Inference
            clip_input = clip_preprocess(Image.fromarray(crop)).unsqueeze(0).to("cuda")
            with torch.no_grad():
                img_feat = clip_model.encode_image(clip_input)
                txt_feat = clip_model.encode_text(clip_text)
                logits = (img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy().squeeze()
            best_idx = np.argmax(logits)
            best_conf = logits[best_idx]

            print(f"CLIP Crack Conf: {best_conf:.2f}, Best Index: {best_idx}")

            #Skip if clip below threshold and not crack class
            if best_conf < clip_thresh and best_idx != 0:
                continue

            # Resize and apply mask
            mask = m.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            crack_mask[mask_resized > 127] = 1  # Class 1 = crack

    spall_mask = np.zeros((H, W), dtype=np.uint8)
    spalling_result = spalling_model.predict(frame, verbose=False)[0]
    spalling_confs = spalling_result.boxes.conf.cpu().numpy()
    spalling_boxes = spalling_result.boxes.xyxy.cpu().numpy().astype(int)

    if spalling_result.masks is not None:
        for idx, m in enumerate(spalling_result.masks.data):
            if spalling_confs[idx] < spalling_thresh:
                #print("spalling made it", spalling_confs[idx])
                continue

            # Crop for CLIP
            x1, y1, x2, y2 = spalling_boxes[idx]
            crop = image_rgb[y1:y2, x1:x2]
            plt.imshow(crop)
            if crop.size == 0:
                continue

            #Clip Inference
            clip_input = clip_preprocess(Image.fromarray(crop)).unsqueeze(0).to("cuda")
            with torch.no_grad():
                img_feat = clip_model.encode_image(clip_input)
                txt_feat = clip_model.encode_text(clip_text)
                logits = (img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy().squeeze()
            best_idx = np.argmax(logits)
            best_conf = logits[best_idx]

            print(f"CLIP Spalling Conf: {best_conf:.2f}, Best Index: {best_idx}")

            #Skip if clip below threshold and not spalling class
            if best_conf < clip_thresh and best_idx != 1:
                continue

            # Resize and apply mask
            mask = m.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            spall_mask[mask_resized > 127] = 2  # Class 2 = spalling

    # Combine masks
    final_mask = np.zeros((H, W), dtype=np.uint8)
    final_mask[crack_mask == 1] = 1
    final_mask[spall_mask == 2] = 2
    final_mask[(crack_mask == 1) & (spall_mask == 2)] = 3 # Class 3 = Both detections
     
    
    # mask overlay
    mask_overlay = np.zeros_like(image_rgb)
    mask_overlay[final_mask == 1] = [255, 0, 0]     # Red for cracks
    mask_overlay[final_mask == 2] = [0, 255, 0]     # Green for spalling
    mask_overlay[final_mask == 3] = [255, 0, 255]   # Magenta for both

    output_image = cv2.addWeighted(image_rgb, 1.0, mask_overlay, 0.6, 0)

    for idx, conf in enumerate(crack_confs):
        if conf < crack_thresh:
            continue
        x1, y1, _, _ = crack_boxes[idx]
        cv2.putText(output_image,
                    f"Crack {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),
                    2)

    for idx, conf in enumerate(spalling_confs):
        if conf < spalling_thresh:
            continue
        x1, y1, _, _ = spalling_boxes[idx]
        cv2.putText(output_image,
                    f"Spalling {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2)

    ys, xs = np.where(final_mask == 3)
    if ys.size > 0:
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.putText(output_image,
                    "Both",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 255),
                    2)
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
    return output_image

crack_model = YOLO("C:/Users/adity/Projects/FTR Research/crack_segmentation_model_02.pt")      
spalling_model = YOLO("C:/Users/adity/Projects/FTR Research/spalling_segmentation_model_01.pt")
crack_thresh = 0
spalling_thresh = 0
clip_thresh = 0.3
frame = cv2.imread("Screenshot 2025-07-01 004652.png")  # Replace with your image path
start_time = time.time()
output = detect_crack_and_spall_clip(frame, crack_model, spalling_model, crack_thresh, spalling_thresh, clip_thresh)
end_time = time.time()
print(f"Combined Mask runtime: {end_time - start_time:.2f} seconds")