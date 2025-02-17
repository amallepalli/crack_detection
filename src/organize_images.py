import os
import random
import shutil
import glob
crack_base_dir = "crack_detection/dataset/Crack"
non_crack_base_dir = "crack_detection/dataset/Non-Crack"
train_base_dir = "crack_detection/dataset/train"
val_base_dir = "crack_detection/dataset/val"
test_base_dir = "crack_detection/dataset/test"

categories = ["crack", "non-crack"]
train_ratio = 4/7
val_ratio = 2/7
test_ratio = 1/7

crack_images = glob.glob(os.path.join(crack_base_dir, "*.jpg"))
non_crack_images = glob.glob(os.path.join(non_crack_base_dir, "*.jpg"))
random.shuffle(crack_images)
random.shuffle(non_crack_images)

total_images = len(crack_images)
train_split = int(total_images * train_ratio)
val_split = train_split + int(total_images * val_ratio)

crack_train_images = crack_images[:train_split]
crack_val_images = crack_images[train_split:val_split]
crack_test_images = crack_images[val_split:]

non_crack_train_images = non_crack_images[:train_split]
non_crack_val_images = non_crack_images[train_split:val_split]
non_crack_test_images = non_crack_images[val_split:]

def move_images(image_list, dest_dir):
    for img_path in image_list:
        shutil.move(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

# Move Crack images
move_images(crack_train_images, os.path.join(train_base_dir, "crack"))
move_images(crack_val_images, os.path.join(val_base_dir, "crack"))
move_images(crack_test_images, os.path.join(test_base_dir, "crack"))

# Move Non-Crack images
move_images(non_crack_train_images, os.path.join(train_base_dir, "non-crack"))
move_images(non_crack_val_images, os.path.join(val_base_dir, "non-crack"))
move_images(non_crack_test_images, os.path.join(test_base_dir, "non-crack"))

print("Dataset successfully split and organized!")

