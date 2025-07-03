import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import tensorflow as tf

model = tf.keras.models.load_model("crack_classification_model_01.keras")

tf.random.set_seed(0)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_dataset = datagen.flow_from_directory(
    "crack_detection/dataset/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

y_true = test_dataset.classes
y_pred_prob = model.predict(test_dataset)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.class_indices.keys())
display.plot(cmap=plt.cm.Blues)
plt.show()
