import numpy
from tensorflow import keras
from keras.utils import np_utils
import pandas as pd
import matplotlib.pyplot as plt

# Prep the data

# Load in the data

# Normalize the inputs from 0-255 to between 0 and 1 by dividing by 255

# One-hot encode outputs

# Create the model
model = keras.Sequential()

# Create the convolution layers

# Flatten the convolutional layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))

# Create a Dense layer
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())

# Create the final layer with the softmax activation function

# Compile the model

# Print out model summary.
print(model.summary())

# Train the model

# Evaluate the model

# Plot the graph of the model
pd.DataFrame(history.history).plot()
plt.show()