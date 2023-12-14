"""
To run this project make sure that you:
    - install Python >=3.10
    - install Keras

Project created by:
    Kajetan Welc
    Daniel Wirzba
"""

import keras

animals_cifar = keras.datasets.cifar10

(train_images, train_labels), (test_images,
                               test_labels) = animals_cifar.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
        keras.layers.Softmax(),
    ]
)

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)
