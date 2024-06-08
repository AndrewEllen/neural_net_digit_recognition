import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# The MNIST labelled dataset for handwritten numbers
mnist = tf.keras.datasets.mnist

# x_training_data is the image data. y_training_data is the labelled data.
# x_test_data is the test image data. y_test_data is the test labelled data.
(x_training_data, y_training_data), (x_test_data, y_test_data) = mnist.load_data()

# Normalizing the training and test data
x_training_data = tf.keras.utils.normalize(x_training_data, axis=1)
x_test_data = tf.keras.utils.normalize(x_test_data, axis=1)

model = tf.keras.models.Sequential()

# Flatten means you're flattening the input shape, in this case it's a 28x28 pixel image.
# Flatten turns it into a flat layer instead of a grid of 28x28 pixels. So one big line of 784 pixels.
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# A dense layer is the most basic layer. In a dense layer each neuron is connected to each other neuron of the other layer.
# relu is rectified linear unit
model.add(tf.keras.layers.Dense(128, activation='relu'))