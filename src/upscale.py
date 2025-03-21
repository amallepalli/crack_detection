import os
import random
import cv2

# Input and output directories

input_folder = "/Users/amallepalli/Downloads/accepted/"  # Change this to your actual folder
output_folder = "/Users/amallepalli/Downloads/spalling"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)


# Get list of images in the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Select 285 random images
selected_images = random.sample(image_files, min(73, len(image_files)))

# Resize settings
new_size = (224, 224)

# Process each selected image
for img_name in selected_images:
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)
    os.remove(img_path)

    if img is None:
        print(f"Skipping {img_name}, unable to read.")
        continue

    # Resize using bicubic interpolation
    upscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

    # Save the upscaled image
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, upscaled_img)

print(f"Upscaling complete! {len(selected_images)} images saved to {output_folder}.")
