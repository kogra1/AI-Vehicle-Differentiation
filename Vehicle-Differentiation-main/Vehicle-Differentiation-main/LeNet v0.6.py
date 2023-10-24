import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


TLDir = 'Military and Civilian Vehicles Classification/Images'
imgSize = 32    # Number of pixels of one dimension of the image.
batchSize = 32

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
    batch_size=batchSize,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TLDir,
    x_col='filename',
    y_col='class',
    target_size=(imgSize, imgSize),
    batch_size=batchSize,
    class_mode='categorical'
)
#(train_generator.classes)
#print(train_generator.filepaths.__class__)

# Create a LeNet model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='tanh', input_shape=(imgSize, imgSize, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='tanh'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='tanh'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='tanh'))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#print(train_generator.filepaths)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)
"""
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
"""
# Print the predicted class label
def switch(label):
    if label == 0:
        return "Civilian Aircraft"
    elif label == 1:
        return "Civilian Car"
    elif label == 2:
        return "Military Aircraft"
    elif label == 3:
        return "Military Helicopter"
    elif label == 4:
        return "Military Tank"
    elif label == 5:
        return "Military Truck"
#print('Predicted class:', train_generator.classes[class_idx])
#print('History: ', history)

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


    #print(prediction[0])
    predictions[counter] = train_generator.classes[class_idx]
    #print('Predicted class:', train_generator.classes[class_idx])


    # Display Test Images
    pimg = np.asarray(Image.open(image))
    plt.imshow(pimg)
    plt.xlabel(switch(train_generator.classes[class_idx]))
    #plt.axis("off")
    if (counter % 100 == 0) & (counter < 100):
        plt.show()

    counter+=1
    if counter == 999:
        break

numRight = 0
i = 0;
for pred in predictions:
    if pred == train_generator.classes[i]:
        numRight+=1

#print(len(predictions))
#print(predictions)
print("SYSTEM ACCURACY: ", numRight/1000*100, "%")
"""
# Display the history graph
pd.DataFrame(history.history).plot()
plt.ylabel('Decimal Value')
plt.xlabel('Epoch')
plt.title('LeNet Evaluation Metrics')
plt.show()
"""