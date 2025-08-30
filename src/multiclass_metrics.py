import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from collections import Counter

# ---------------------------
# 1) Helpers
# ---------------------------

def load_txt_mask_for_class(txt_path, image_shape, target_class_id=None):
    """
    Convert a YOLO segmentation .txt into a binary mask for a specific class.
    If target_class_id is None, includes all classes.
    Expected line format: <class_id> x1 y1 x2 y2 ... (normalized coords)
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if not os.path.exists(txt_path):
        return mask  # no labels -> empty mask

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        # class id can appear as float in some exports; coerce safely
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue

        if (target_class_id is not None) and (cls_id != target_class_id):
            continue

        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0 or len(coords) < 6:
            continue  # malformed polygon

        xy = np.array(coords, dtype=np.float32).reshape(-1, 2)
        xy[:, 0] *= W
        xy[:, 1] *= H
        polygon = np.round(xy).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)

    return mask


def combine_pred_masks_for_class(result, target_class_id, out_shape):
    """
    From a single YOLO result, return a binary mask for a target class id.
    result: Ultralytics result object (single image)
    """
    H, W = out_shape
    if result.masks is None or result.boxes is None or result.boxes.cls is None:
        return np.zeros((H, W), dtype=np.uint8)

    pred_masks = result.masks.data.cpu().numpy()  # (N, Hm, Wm), float [0..1]
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    idxs = np.where(cls_ids == target_class_id)[0]
    if idxs.size == 0:
        return np.zeros((H, W), dtype=np.uint8)

    combined = np.zeros(pred_masks.shape[1:], dtype=np.float32)
    for i in idxs:
        combined = np.maximum(combined, pred_masks[i])

    if combined.shape != (H, W):
        combined = cv2.resize(combined, (W, H), interpolation=cv2.INTER_NEAREST)

    return (combined > 0.5).astype(np.uint8)


def eval_seg_metrics(y_true_list, y_pred_list, tag):
    y_true = np.concatenate([m.flatten() for m in y_true_list])
    y_pred = np.concatenate([m.flatten() for m in y_pred_list])

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    iou       = jaccard_score(y_true, y_pred, zero_division=0)

    print(f"\nEvaluation Metrics for {tag}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"IoU      : {iou:.4f}")


def find_class_id_by_name(names_dict, name_substr):
    """Find class id in model.names by substring match (case-insensitive)."""
    name_substr = name_substr.lower()
    for k, v in names_dict.items():
        if name_substr in str(v).lower():
            return int(k)
    return None


def guess_gt_class_id(label_folder, sample=100):
    """
    Guess the class id used for the single-class dataset in a label folder by
    counting the most frequent first token (class id) across up to 'sample' files.
    """
    paths = sorted(glob.glob(os.path.join(label_folder, "*.txt")))[:sample]
    counter = Counter()
    for p in paths:
        try:
            with open(p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cid = int(float(parts[0]))
                            counter[cid] += 1
                        except ValueError:
                            pass
        except FileNotFoundError:
            continue
    if not counter:
        # default to 0 if empty (no labels found)
        return 0
    return counter.most_common(1)[0][0]


def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)

# ---------------------------
# 2) Paths (reuse your structure)
# ---------------------------

crack_image_folder = "crack_detection/crack-seg/test/images"
crack_label_folder = "crack_detection/crack-seg/test/labels"

spall_image_folder = "crack_detection/spalling/valid/images"
spall_label_folder = "crack_detection/spalling/valid/labels"

crack_image_paths = list_images(crack_image_folder)
crack_label_paths = sorted(glob.glob(os.path.join(crack_label_folder, "*.txt")))

spall_image_paths = list_images(spall_image_folder)
spall_label_paths = sorted(glob.glob(os.path.join(spall_label_folder, "*.txt")))

assert len(crack_image_paths) == len(crack_label_paths), "Crack images/labels mismatch"
assert len(spall_image_paths)  == len(spall_label_paths), "Spall images/labels mismatch"

# ---------------------------
# 3) Load single multi-class model
# ---------------------------

multi_model = YOLO("crack_and_spalling_segmentation_01.pt")  # <-- your single model weights

# Auto-detect model class IDs (e.g., {0:'crack', 1:'spalling'})
CRACK_MODEL_ID = find_class_id_by_name(multi_model.names, "crack")
SPALL_MODEL_ID = find_class_id_by_name(multi_model.names, "spall")

if CRACK_MODEL_ID is None or SPALL_MODEL_ID is None:
    raise ValueError(f"Could not find 'crack'/'spall' in model.names: {multi_model.names}")

print("Model class map:", multi_model.names)
print(f"Detected model IDs -> CRACK: {CRACK_MODEL_ID}, SPALL: {SPALL_MODEL_ID}")

# Auto-detect GT class IDs per dataset (handles single-class exports)
CRACK_GT_ID = guess_gt_class_id(crack_label_folder)
SPALL_GT_ID = guess_gt_class_id(spall_label_folder)

print(f"Guessed GT IDs -> CRACK_GT_ID: {CRACK_GT_ID}  (from {crack_label_folder})")
print(f"Guessed GT IDs -> SPALL_GT_ID: {SPALL_GT_ID}  (from {spall_label_folder})")

# ---------------------------
# 4) Evaluate on the *same* datasets as your dual-model runs
#    (A) Crack dataset, measure CRACK class only
# ---------------------------

cr_gt_masks = []
cr_pr_masks = []

for img_path, lbl_path in zip(crack_image_paths, crack_label_paths):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    gt_mask = load_txt_mask_for_class(lbl_path, (H, W), target_class_id=CRACK_GT_ID)
    cr_gt_masks.append(gt_mask)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = multi_model.predict(img_rgb, imgsz=(W, H), verbose=False)[0]
    pr_mask = combine_pred_masks_for_class(res, target_class_id=CRACK_MODEL_ID, out_shape=(H, W))
    cr_pr_masks.append(pr_mask)

print("\nCRACK sanity counts:",
      "GT_pos_px =", sum(int(m.sum()) for m in cr_gt_masks),
      "PR_pos_px =", sum(int(m.sum()) for m in cr_pr_masks))
eval_seg_metrics(cr_gt_masks, cr_pr_masks, "Multi-class model on CRACK dataset (CRACK class)")

# ---------------------------
#    (B) Spall dataset, measure SPALL class only
# ---------------------------

sp_gt_masks = []
sp_pr_masks = []

for img_path, lbl_path in zip(spall_image_paths, spall_label_paths):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    gt_mask = load_txt_mask_for_class(lbl_path, (H, W), target_class_id=SPALL_GT_ID)
    sp_gt_masks.append(gt_mask)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = multi_model.predict(img_rgb, imgsz=(W, H), verbose=False)[0]
    pr_mask = combine_pred_masks_for_class(res, target_class_id=SPALL_MODEL_ID, out_shape=(H, W))
    sp_pr_masks.append(pr_mask)

print("\nSPALL sanity counts:",
      "GT_pos_px =", sum(int(m.sum()) for m in sp_gt_masks),
      "PR_pos_px =", sum(int(m.sum()) for m in sp_pr_masks))
eval_seg_metrics(sp_gt_masks, sp_pr_masks, "Multi-class model on SPALL dataset (SPALL class)")
