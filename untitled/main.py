import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf


def train_model():
    model = tf.keras.models.Sequential()

    # Flatten means you're flattening the input shape, in this case it's a 28x28 pixel image.
    # Flatten turns it into a flat layer instead of a grid of 28x28 pixels. So one big line of 784 pixels.
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # A dense layer is the most basic layer. In a dense layer each neuron is connected to each other neuron of the other layer.
    # relu is rectified linear unit
    # softmax is essentially the confidence value for the output which is the probability of the digit being the right number.
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # ten is the number of neurons. Only 10 are needed as there are only ten digits
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fitting the model (training it). Epochs are how many times it runs the training
    model.fit(x_training_data, y_training_data, epochs=3)

    # Saving the model
    model.save('handwritten_number_recognition_model.keras')


def test_model():
    loss, accuracy = model.evaluate(x_test_data, y_test_data)

    print(loss)
    print(accuracy)


def run_model():
    image_number = 1
    while os.path.isfile(f"test_data/digit{image_number}.png"):
        try:
            # Taking last channel of the png. Only need the shape data no pixel colour data needed.
            img = cv2.imread(f"test_data/digit{image_number}.png")[:, :, 0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            # argmax gives the neuron with the highest activation
            print(f"The digit is {np.argmax(prediction)}")

            # Setting tkinter to be used for plot display. Without this it wasn't working for me
            matplotlib.use('TkAgg')
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()

        except Exception as exception:
            print("An Error occurred")
            print(exception)
        finally:
            # Looping onto the next number
            image_number += 1

if __name__ == "__main__":
    # Setting up global variables for training data and the model

    # The MNIST labelled dataset for handwritten numbers
    global mnist
    mnist = tf.keras.datasets.mnist

    # x_training_data is the image data. y_training_data is the labelled data.
    # x_test_data is the test image data. y_test_data is the test labelled data.
    global x_training_data, y_training_data, x_test_data, y_test_data
    (x_training_data, y_training_data), (x_test_data, y_test_data) = mnist.load_data()

    # Normalizing the training and test data
    x_training_data = tf.keras.utils.normalize(x_training_data, axis=1)
    x_test_data = tf.keras.utils.normalize(x_test_data, axis=1)

    # If model doesn't exist then train it
    if not os.path.isfile('handwritten_number_recognition_model.keras'):
        train_model()
    global model
    model = tf.keras.models.load_model('handwritten_number_recognition_model.keras')
    run_model()
