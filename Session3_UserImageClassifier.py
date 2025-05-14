# Session 3: Build Your Own Image Classifier
import tensorflow as tf
from tensorflow.keras import layers, models

# Assume images/ contains subfolders for each class with images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'images/', image_size=(128,128), batch_size=32)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128,128,3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=3)