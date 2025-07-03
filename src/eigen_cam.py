from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

crack_model = YOLO("C:\Programming\crack_detection\models\crack_segmentation_model_02.pt")      
spalling_model = YOLO("C:\Programming\crack_detection\models\spalling_segmentation_model_01.pt")
image_path = r"C:\Programming\crack_detection\test3.jpg"

img = cv2.imread(image_path)
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255

target_layers = [crack_model.model.model[-4]]
cam = EigenCAM(crack_model, target_layers, task='seg')
grayscale_cam = cam(rgb_img)[0, :, :]
crack_cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

target_layers = [spalling_model.model.model[-4]]
cam = EigenCAM(spalling_model, target_layers, task='seg')
grayscale_cam = cam(rgb_img)[0, :, :]
spalling_cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

# Create comprehensive visualization
plt.figure(figsize=(6, 12))

plt.subplot(2, 1, 1)
plt.title("Crack Model - Eigen-CAM", fontsize=14, fontweight='bold')
plt.imshow(crack_cam_image)
plt.axis('off')

plt.subplot(2, 1, 2)
plt.title("Spalling Model - Eigen-CAM", fontsize=14, fontweight='bold')
plt.imshow(spalling_cam_image)
plt.axis('off')

plt.tight_layout()
plt.show()