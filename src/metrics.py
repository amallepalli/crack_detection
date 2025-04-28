import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score


def load_txt_mask(txt_path, image_shape):
    """
    Converts YOLO segmentation txt file to a binary mask.
    
    txt_path: path to the label (.txt) file
    image_shape: (height, width) of the corresponding image
    """
    mask = np.zeros(image_shape, dtype=np.uint8)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        coords = list(map(float, parts[1:]))

        # Group into (x, y) pairs
        xy = np.array(coords).reshape(-1, 2)

        # Scale normalized coords back to pixel values
        xy[:, 0] *= image_shape[1]  # scale x by width
        xy[:, 1] *= image_shape[0]  # scale y by height

        # Round and convert to integers
        polygon = np.round(xy).astype(np.int32)

        # Fill the polygon on the mask
        cv2.fillPoly(mask, [polygon], 1)

    return mask

crack_image_folder = "crack_detection/crack-seg/test/images"   # your original test images
crack_label_folder = "crack_detection/crack-seg/test/labels"   # your txt label files

spall_image_folder = "crack_detection/spalling-3/valid/images"
spall_label_folder = "crack_detection/spalling-3/valid/labels"

# Get list of image and label paths
crack_image_paths = sorted(glob.glob(os.path.join(crack_image_folder, "*.jpg")))
crack_label_paths = sorted(glob.glob(os.path.join(crack_label_folder, "*.txt")))

spall_image_paths = sorted(glob.glob(os.path.join(spall_image_folder, "*.jpg")))
spall_label_paths = sorted(glob.glob(os.path.join(spall_label_folder, "*.txt")))

# This will store your ground truth masks
crack_ground_truth_masks = []

# Loop through all images and labels
for img_path, lbl_path in zip(crack_image_paths, crack_label_paths):
    # Load image to get size
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Create the ground truth mask
    gt_mask = load_txt_mask(lbl_path, (height, width))

    crack_ground_truth_masks.append(gt_mask)

print(f"Successfully created {len(crack_ground_truth_masks)} crack ground truth masks.")

crack_model = YOLO("crack_segmentation_model_02.pt")
crack_predicted_masks = []
for img_path in crack_image_paths:
    # Load image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Predict using crack model
    results = crack_model.predict(img_rgb, imgsz=(img.shape[1], img.shape[0]))

    # Check if any masks were predicted
    if results[0].masks is None:
        # No detections: create an empty mask
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    else:
        # Get predicted masks
        pred_masks = results[0].masks.data.cpu().numpy()  # (N_objects, H, W)

        # Combine multiple masks into one
        combined_mask = np.any(pred_masks > 0.5, axis=0).astype(np.uint8)

    crack_predicted_masks.append(combined_mask)

print(f"Predicted {len(crack_predicted_masks)} crack masks.")


# This will store your ground truth masks
spall_ground_truth_masks = []

# Loop through all images and labels
for img_path, lbl_path in zip(spall_image_paths, spall_label_paths):
    # Load image to get size
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Create the ground truth mask
    gt_mask = load_txt_mask(lbl_path, (height, width))

    spall_ground_truth_masks.append(gt_mask)

print(f"Successfully created {len(spall_ground_truth_masks)} spall ground truth masks.")

spall_model = YOLO("spalling_segmentation_model_01.pt")
spall_predicted_masks = []
for img_path in spall_image_paths:
    # Load image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Predict using spall model
    results = spall_model.predict(img_rgb, imgsz=(img.shape[1], img.shape[0]))

    # Check if any masks were predicted
    if results[0].masks is None:
        # No detections: create an empty mask
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    else:
        # Get predicted masks
        pred_masks = results[0].masks.data.cpu().numpy()  # (N_objects, H, W)

        # Combine multiple masks into one
        combined_mask = np.any(pred_masks > 0.5, axis=0).astype(np.uint8)

    spall_predicted_masks.append(combined_mask)

print(f"Predicted {len(spall_predicted_masks)} spall masks.")


# Squeeze predicted masks if needed
crack_predicted_masks = [np.squeeze(mask) for mask in crack_predicted_masks]
spall_predicted_masks = [np.squeeze(mask) for mask in spall_predicted_masks]


def get_metrics(true_masks, pred_masks, model_type):

    # Flatten all ground truth and predicted masks
    y_true = np.concatenate([mask.flatten() for mask in true_masks])
    y_pred = np.concatenate([mask.flatten() for mask in pred_masks])

    # Now compute the metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)

    # Print final metrics
    print(f"Evaluation Metrics for {model_type} Segmentation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score (Dice): {f1:.4f}")
    print(f"IoU: {iou:.4f}")

get_metrics(crack_ground_truth_masks, crack_predicted_masks, "Crack")
get_metrics(spall_ground_truth_masks, spall_predicted_masks, "Spalling")

