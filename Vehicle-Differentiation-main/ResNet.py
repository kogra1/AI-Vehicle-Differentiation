import keras
import tensorflow as tf
import pandas as pd
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers.convolutional import MaxPooling2D
from keras.datasets import cifar10
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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

# Create a ResNet model
def residual_block(inputs, filters, kernel_size=3, strides=1):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='s1ame')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x

inputs = Input(shape=(imgSize, imgSize, 3))
x = Conv2D(32, kernel_size=3, strides=1, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2)(x)
x = residual_block(x, 32)
x = MaxPooling2D(pool_size=2)(x)
x = residual_block(x, 32)
x = MaxPooling2D(pool_size=2)(x)
x = residual_block(x, 32)
x = MaxPooling2D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(6, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)


# Print the predicted class label
def switch(label):
    if label == 0:
        return "Civilian"
    elif label == 1:
        return "Military"

counter = 0
predictions = [0] * 1000
imsize = (12,8)

for image in train_generator.filepaths:
    img = cv2.imread(image)
    img = cv2.resize(img, (imgSize, imgSize))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction[0])

    predictions[counter] = train_generator.classes[class_idx]
    """
    # Display Test Images
    pimg = np.asarray(Image.open(image))
    plt.imshow(pimg)
    plt.xlabel(switch(train_generator.classes[class_idx]))
    #plt.axis("off")
    #if (counter % 10 == 0) & (counter < 100):
        #plt.show()
    """
    counter+=1
    if counter == 999:
        break

numRight = 0
i = 0;
for pred in predictions:
    if pred == train_generator.classes[i]:
        numRight+=1

print("SYSTEM ACCURACY: ", numRight/1000*100, "%")

"""
# Display the history graph
pd.DataFrame(history.history).plot()
plt.ylabel('Decimal Value')
plt.xlabel('Epoch')
plt.title('Resnet Evaluation Metrics')
plt.show()
"""