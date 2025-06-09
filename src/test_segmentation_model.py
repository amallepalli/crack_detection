from ultralytics import YOLO
import cv2
model = YOLO("/Users/amallepalli/Projects/FTR2025/crack_detection/crack_segmentation_model_02.pt")
image = cv2.imread("/Users/amallepalli/Projects/FTR2025/crack-seg/train/images/1411.rf.169023ce46b72b27d898ecbc9ac6ecf5.jpg")
results = model.predict(image, save=True, show=False)

masks = results[0].masks

# Now you can access masks.data, which is a tensor of shape [num_masks, height, width]
# To convert it to numpy arrays:
if masks is not None:
    mask_array = masks.data.cpu().numpy()  # (num_instances, height, width)
    print(mask_array)
else:
    print("No masks detected!")