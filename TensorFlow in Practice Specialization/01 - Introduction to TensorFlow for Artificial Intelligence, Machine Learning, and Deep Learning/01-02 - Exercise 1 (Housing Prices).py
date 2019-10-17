# We have to predict the house price accoring to the following formula:
# price = 50 + 50*number_of_bedrooms

import tensorflow as tf
import numpy as np
from tensorflow import keras


def get_price(number_of_bedrooms):
    return (50 + 50 * number_of_bedrooms) / 100


# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10], float)
    ys = np.array(get_price(xs), float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)[0]


prediction = house_model([7.0])
print(prediction)
