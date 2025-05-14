# Session 6: Implement Scaled Dot-Product Attention
import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(dk)
    weights = tf.nn.softmax(scaled, axis=-1)
    return tf.matmul(weights, v), weights

# Example
q = tf.constant(np.random.rand(1,5,64), dtype=tf.float32)
k = tf.constant(np.random.rand(1,5,64), dtype=tf.float32)
v = tf.constant(np.random.rand(1,5,64), dtype=tf.float32)
output, attn = scaled_dot_product_attention(q, k, v)
print("Attention shape:", attn.shape)