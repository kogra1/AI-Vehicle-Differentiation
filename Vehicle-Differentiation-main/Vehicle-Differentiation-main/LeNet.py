

import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
import matplotlib.pyplot as plt

TLDir = 'Military and Civilian Vehicles Classification/Images'
imgSize = 32    # Number of pixels of one dimension of the image.

# Load annotations into memory using Pandas
train_df = pd.read_csv('Military and Civilian Vehicles Classification/Images/train_labels.csv')
test_df = pd.read_csv('Military and Civilian Vehicles Classification/Images/test_labels.csv')

# Create an ImageDataGenerator and specify the dataframe and image directory
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TLDir,
    x_col='filename',
    y_col='class',
    target_size=(imgSize, imgSize),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TLDir,
    x_col='filename',
    y_col='class',
    target_size=(imgSize, imgSize),
    batch_size=32,
    class_mode='categorical'
)
print(train_generator.classes)
# Create a LeNet model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(imgSize, imgSize, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,                              # RESET THIS TO 10 EPOCHS FOR DEMONSTRATION!
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Load the image you want to test
img = cv2.imread(TLDir+'/1(1).jpg')

# Resize the image to the same size as your training data
img = cv2.resize(img, (imgSize, imgSize))

# Displaying an img:
cv2.imshow('Image', img)  # Create a window with the img in it titled 'Image'
cv2.waitKey(0)            # Wait until key is pressed, then destroy window
cv2.destroyAllWindows()

# Preprocess the image
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

# Use the predict method of the model to predict the class of the image
prediction = model.predict(img)

# The output of the predict method is a one-hot encoded vector
# To get the class label, find the index of the highest value in the vector
class_idx = np.argmax(prediction[0])

# Print the predicted class label

print('Predicted class:', train_generator.classes[class_idx])
print('History: ', history)

# Display the history graph
pd.DataFrame(history.history).plot()
plt.title('Evaluation Metrics')
plt.show()
