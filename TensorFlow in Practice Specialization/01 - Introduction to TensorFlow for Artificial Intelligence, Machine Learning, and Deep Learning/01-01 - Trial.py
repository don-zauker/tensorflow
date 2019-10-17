# Aim of this trial is to find the function that implements the following one:
#
#    float hw_function(float x){
#        float y = (2 * x) - 1;
#        return y;
#    }
# This trial will be splitted ion two parts. The first

# I will train the model with Both SKLEARN and TENSORFLOW

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def hw_function(x):
    return(2*x) - 1


xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# SKLEARN

print("\n##### USING SKLEARN\n")

model = LinearRegression()

X = np.reshape(xs, (-1, 1))
y = np.reshape(ys, (-1, 1))

X_test = np.array([10, 20, -2])
y_test = np.array([19.0, 39, -5])

model.fit(X, y)

prediction = model.predict(np.reshape(X_test, (-1, 1)))

print ("MAE:", mean_absolute_error(y_test, prediction))
print ("Expected: [[-5.]]", "Got:", model.predict(np.reshape([-2], (-1, 1))))

# Tensorflow
print("\n##### USING TENSORFLOW")

model = Sequential()
model.add(Dense(units=1, input_shape=[1]))

# SGD: Stochastic Gradient Descent


model.compile(optimizer="sgd", loss="mean_squared_error")

model.fit(xs, ys, epochs=500,)
print ("Expected: [[19.]]", "Got:", model.predict([10.0]))
