import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore

tf.random.set_seed(0)

#Loading and preprocessing the data
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
'''
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "crack_detection/dataset/train",
    shuffle=True,
    batch_size=32,
    image_size=(224, 224)
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "crack_detection/dataset/val",
    shuffle=True,
    batch_size=32,
    image_size=(224, 224)
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "crack_detection/dataset/test",
    shuffle=True,
    batch_size=32,
    image_size=(224, 224)
)

class_names = train_dataset.class_names
print(f"Class Names: {class_names}")
'''
train_dataset = datagen.flow_from_directory(
    "crack_detection/dataset/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)


# Load validation data
val_dataset = datagen.flow_from_directory(
    "crack_detection/dataset/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load test data
test_dataset = datagen.flow_from_directory(
    "crack_detection/dataset/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
class_names = list(train_dataset.class_indices.keys())  # Convert keys to a list
print(f"Class Names: {class_names}")

model = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)

for layer in model.layers:
    layer.trainable = False


x = model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x) #Extra layer for better learning
x = Dense(64, activatoin="relu")(x) #Extra later to help learn complex patterns
predictions = Dense(3, activation="softmax")(x)
model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Create a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.2,          # Reduce learning rate by this factor
    patience=5,           # Number of epochs with no improvement before reducing
    min_lr=1e-6           # Minimum learning rate
)

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    steps_per_epoch=len(train_dataset),
    validation_steps=len(val_dataset),
    callbacks=[lr_scheduler]
)


# Function to plot training and validation accuracy/loss
def plot_training_history(history):
    # Extract accuracy and loss from history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r*-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Call the function after training
plot_training_history(history)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save("crack_detection/crack_detection_model_02.keras")