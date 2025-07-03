from ultralytics import YOLO
import cv2
model = YOLO("C:/Users/adity/Projects/FTR Research/spalling_segmentation_model_01.pt")
image = cv2.imread("Screenshot 2025-07-01 004652.png")  # Replace with your image path
results = model.predict(image, save=True, show=False)

masks = results[0].masks

# Now you can access masks.data, which is a tensor of shape [num_masks, height, width]
# To convert it to numpy arrays:
if masks is not None:
    mask_array = masks.data.cpu().numpy()  # (num_instances, height, width)
    print(mask_array)
else:
    print("No masks detected!")