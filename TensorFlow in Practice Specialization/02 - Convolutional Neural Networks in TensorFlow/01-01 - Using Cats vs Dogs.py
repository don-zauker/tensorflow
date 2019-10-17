import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

basedir = '/tmp/cats_and_dogs_filtered'

train_dir = os.path.join(basedir, 'train')
validation_dir = os.path.join(basedir, 'validation')

train_dogs_dir = os.path.join(train_dir, 'dogs')
train_cats_dir = os.path.join(train_dir, 'cats')

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')

train_dog_fnames = os.listdir(train_dogs_dir)
train_cat_fnames = os.listdir(train_cats_dir)

print(train_dog_fnames[:10])
print(train_cat_fnames[:10])

# Let's find out the total number of cat and dog images
# in the train and validation directories:
print('total training dog images :', len(train_dog_fnames))
print('total training cat images :', len(train_cat_fnames))

print('total validation dog images :', len(os.listdir(validation_dogs_dir)))
print('total validation cat images :', len(os.listdir(validation_cats_dir)))

# Now let's take a look at a few pictures to get a better sense
# of what the cat and dog datasets look like.
# First, configure the matplot parameters

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[0:pic_index+8]]

next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[0:pic_index+8]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

#plt.show()

# In order to train a neural network to handle the images,
# you'll need them to be in a uniform size.
# We've chosen 150x150 for this, and you'll see the code that
# preprocesses the images to that shape shortly.
# But before we continue, let's start defining the model:

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
])

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    target_size=(150, 150),
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size=20,
    target_size=(150, 150),
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_steps=50,
    verbose=2
)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
