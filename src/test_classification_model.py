import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("crack_detection/crack_classification_model_02.keras")
class_names = ['crack', 'non-crack', 'spalling']

def predict_image(img_path):
    image = cv2.imread(img_path)
    image_resized = cv2.resize(image, (224, 224))
    image = np.expand_dims(image_resized, axis=0)
    print(image.shape)
    pred = model.predict(image)
    print(pred)
    output_class=class_names[np.argmax(pred)]
    confidence = np.max(pred)
    print(f"Predicted Class: {output_class} with confidence {confidence:.2f}")

# Test on a sample image
predict_image("IMG_9764.jpg")
#00001 - correct non crack

#00012 - correct crack

#00450 - crack with low confidence

#00121 - crack but detects as non-crack