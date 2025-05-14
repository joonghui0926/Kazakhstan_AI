# Session 2: CNN & Filter Visualization
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1) / 255.0

# Define simple CNN
model = models.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)

# Visualize first-layer filters
filters, biases = model.layers[0].get_weights()
n_filters = filters.shape[-1]
fig, axes = plt.subplots(1, n_filters, figsize=(12,3))
for i in range(n_filters):
    axes[i].imshow(filters[:,:,0,i], cmap='gray')
    axes[i].axis('off')
plt.show()