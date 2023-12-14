"""
To run this project make sure that you:
    - install Python >=3.10
    - install Keras
    - install Numpy
    - install Scikit-learn
    - install Matplotlib

Project created by:
    Kajetan Welc
    Daniel Wirzba
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


class Estimator:
    _estimator_type = "classifier"
    classes_ = []

    def __init__(self, model, classes):
        self.model = model
        self.classes_ = classes

    def predict(self, x):
        y_prob = self.model.predict(x)
        y_pred = y_prob.argmax(axis=1)
        return y_pred


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

input_shape = train_images[0].shape
output_shape = len(np.unique(train_labels))

model1 = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
        keras.layers.Softmax()
    ]
)

model2 = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10),
        keras.layers.Softmax()
    ]
)

model1.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model2.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model1.fit(train_images, train_labels, epochs=10)
model2.fit(train_images, train_labels, epochs=10)

classifier = Estimator(model1, class_names)

ConfusionMatrixDisplay.from_estimator(
    estimator=classifier, X=test_images, y=test_labels)

plt.show()

test_loss_1, test_acc_1 = model1.evaluate(
    test_images,
    test_labels,
    verbose=2
)
print("\nTest accuracy for model A:", test_acc_1)

test_loss_2, test_acc_2 = model2.evaluate(
    test_images,
    test_labels,
    verbose=2
)
print("\nTest accuracy for model B:", test_acc_2)
