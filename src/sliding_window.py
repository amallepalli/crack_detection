import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
# code inspired from https://github.com/KangchengLiu/Crack-Detection-and-Segmentation-Dataset-for-UAV-Inspection
def highlight_cracks(im, im_size, model, class_names, step_size):
    im = cv2.resize(im, im_size)
    output_image = im.copy()
    row = 0
    for i in range(2 * int(im.shape[0] / step_size) + 1):
        col = 0
        for j in range(2 * int(im.shape[1] / step_size) + 1):
            sliding_window = im[row:row+step_size, col:col+step_size]
            if sliding_window.shape[0] == 0 or sliding_window.shape[1] == 0:
                continue
            image_resized = cv2.resize(sliding_window, im_size)
            image = np.expand_dims(image_resized, axis=0)
            pred = model.predict(image)
            output_class = class_names[np.argmax(pred)]
            if(output_class == "crack"):
                color = (255, 26, 26) # red
            else:
                color = (153, 255, 153) # green
            colored_window = np.zeros_like(sliding_window, dtype=np.uint8)
            colored_window[:] = color
            output_image[row:row+step_size, col:col+step_size] = cv2.addWeighted(output_image[row:row+step_size, col:col+step_size], 0.83, colored_window, 0.17, 0)
            col += int(step_size/2)
        row += int(step_size/2)
    return output_image

model = tf.keras.models.load_model("crack_detection/crack_detection_model_01.keras")
image_path = "C:/Users/adity/Downloads/concrete_crack.jpeg"
class_names = ["crack", "non-crack"]
step_size = 25
im_size = (224, 224)
im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
print(im.shape)
if im is None:
    raise ValueError("Error: Image not found. Check the file path.")
output_img = highlight_cracks(im, im_size, model, class_names, step_size)

plt.imshow(output_img)
plt.show()

