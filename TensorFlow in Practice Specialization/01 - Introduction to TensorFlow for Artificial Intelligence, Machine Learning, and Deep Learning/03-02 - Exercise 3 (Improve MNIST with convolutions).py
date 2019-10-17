import tensorflow as tf


def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    # model fitting
    history = model.fit(
        training_images, training_labels, epochs=20, callbacks=[myCallback()]
    )
    print("Accuracy [loss, accuracy]:", model.evaluate(test_images, test_labels))
    # model fitting
    return history.epoch, history.history['acc'][-1]


print(train_mnist_conv())
