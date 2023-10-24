# PREP DATA

import numpy
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for purposes of reproducibility
seed = 21

# Loading in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

# ----------------------------------------------------------------------------
# CREATE MODEL

model = keras.Sequential()

# Create a layer with 32 channels of 3x3 kernels.
# padding='same' means we're not changing the size of the image.
# We can swap out and compare relu against sigmoid later if we want
model.add(keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))  # Focuses in on changes in the image via pooling
model.add(keras.layers.Dropout(0.2))  # Drop-out layer to prevent overfitting and drops 20% of connections
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

# FLATTEN THE CONVOLUTIONAL LAYERS
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))

# CREATE DENSE LAYER
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())

# Softmax activation function selects neuron with highest probability.
# This is the model's FINAL layer.
model.add(keras.layers.Dense(class_num, activation='softmax'))

# ----------------------------------------------------------------------------
# COMPILE AND PRINT

# Compile the model via the Adam algorithm (this may be changed to Nadam or RMSProp later)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'val_accuracy'])

# Print out model summary.
print(model.summary())

# ----------------------------------------------------------------------------
# TRAIN MODEL

numpy.random.seed(seed)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)

# ----------------------------------------------------------------------------
# EVALUATE THE MODEL
"""
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""
pd.DataFrame(history.history).plot()
plt.show()
