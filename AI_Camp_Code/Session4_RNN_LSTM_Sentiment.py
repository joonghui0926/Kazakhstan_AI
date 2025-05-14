# Session 4: RNN/LSTM Sentiment Analysis
import tensorflow as tf
from tensorflow.keras import layers, models

# Load IMDB dataset
(x_train, y_train), _ = tf.keras.datasets.imdb.load_data(num_words=10000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)

model = models.Sequential([
    layers.Embedding(10000, 16, input_length=200),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=64)