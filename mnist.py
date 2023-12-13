"""
To run this project make sure that you:
    - download Python 3.10

    Project created by:
        Kajetan Welc
        Daniel Wirzba
"""

import keras
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


class Estimator:
    _estimator_type = ''
    classes_ = []

    def __init__(self, model, classes):
        self.model = model
        self._estimator_type = 'classifier'
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

class_names = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']

input_shape = train_images[0].shape
output_shape = len(np.unique(train_labels))

model_size_1 = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
    ]
)

model_size_2 = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(10),
        keras.layers.Softmax()
    ]
)

model_size_1.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model_size_2.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model_size_1.fit(train_images, train_labels, epochs=10)
model_size_2.fit(train_images, train_labels, epochs=10)

classifier = Estimator(model_size_1, class_names)

ConfusionMatrixDisplay.from_estimator(estimator=classifier, X=test_images, y=test_labels)

plt.show()

test_loss_1, test_acc_1 = model_size_1.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc_1)
test_loss_2, test_acc_2 = model_size_2.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc_2)
