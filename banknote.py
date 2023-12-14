"""
To run this project make sure that you:
    - install Python >=3.10
    - install Keras
    - install Numpy
    - install Scikit-learn

Project created by:
    Kajetan Welc
    Daniel Wirzba
"""

import keras
import numpy as np
from sklearn.model_selection import train_test_split

banknote = np.genfromtxt(
    "resources/data_banknote_authentication.csv",
    delimiter=",",
    dtype=str
)

banknote_X = banknote[:, :-1].astype(float)
banknote_y = banknote[:, -1]

banknote_y = np.where(banknote_y == "0", 0, 1)

banknote_X_train, banknote_X_test, banknote_y_train, banknote_y_test = train_test_split(
    banknote_X, banknote_y
)

DROPOUT = 0.2

banknote_model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(2),
        keras.layers.Softmax(),
    ]
)

banknote_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

banknote_model.fit(banknote_X_train, banknote_y_train, epochs=100)

banknote_y_test_pred = banknote_model.predict(banknote_X_test)

test_loss, test_acc = banknote_model.evaluate(
    banknote_X_test,
    banknote_y_test,
)
print("\nTest accuracy:", test_acc)
