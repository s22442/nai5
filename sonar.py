"""
To run this project make sure that you:
    - download Python 3.10

    Project created by:
        Kajetan Welc
        Daniel Wirzba
"""

import keras
import numpy as np
from sklearn.model_selection import train_test_split

sonar = np.genfromtxt("resources/sonar.csv", delimiter=",", dtype=str)

sonar_X = sonar[:, :-1].astype(float)
sonar_y = sonar[:, -1]

sonar_y = np.where(sonar_y == "R", 0, 1)

sonar_X_train, sonar_X_test, sonar_y_train, sonar_y_test = train_test_split(
    sonar_X, sonar_y, test_size=0.4
)

DROPOUT = 0.2

sonar_model = keras.Sequential(
    [
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(2),
        keras.layers.Softmax(),
    ]
)

sonar_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

sonar_model.fit(sonar_X_train, sonar_y_train, epochs=100)

sonar_y_test_pred = sonar_model.predict(sonar_X_test)

test_loss, test_acc = sonar_model.evaluate(
    sonar_X_test,
    sonar_y_test,
)
print("\nTest accuracy:", test_acc)
