import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# NN work better on low data value so we get scale them. Since the max value is 255 we will divide by 255
training_images = training_images / 255.0
test_images = test_images / 255.0

# lets to create the NN 3 layers: Input layer 28x28, hidden layer with 128 nodes, output layer with 10 as the number
# of the classes

# Flatten just takes that square and turns it into a 1 dimensional set
model = keras.Sequential([keras.layers.Flatten(),
                          keras.layers.Dense(units=128, activation=tf.nn.relu),
                          keras.layers.Dense(units=10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Let's get evaluate how the model works with unseen data
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Loss:", test_loss, "Accuracy:", test_acc)

classification = model.predict(test_images)
print(classification[0])
print(test_labels[0])
