from tensorflow import keras
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from GenerateData import Data
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip,RandomZoom,RandomRotation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
data = Data()
IMG_HEIGHT = data.img_height
IMG_WIDTH = data.img_width
train_ds = data.train_ds
val_ds = data.val_ds
learning_rate = 1.00e-3

class_names = data.class_names
data_augmentation = Sequential(
  [
    RandomFlip("horizontal_and_vertical",
                                                 input_shape=(  IMG_HEIGHT,
                                                                IMG_WIDTH,3)),
    RandomRotation(0.2),
    RandomZoom(0.1),

  ]
)
batch_size =1
model = Sequential([
  data_augmentation,
]) 
model.add(Rescaling(1./255))
model.add(Conv2D(16, (8, 8), activation='relu'))
model.add(MaxPooling2D((5, 5)))
model.add(Conv2D(32, (2, 2),  activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(300, activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
model.add(Dense(len(class_names)))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
with tf.device('/GPU:0'):
    epochs = 150
    history = model.fit( train_ds,  validation_data=val_ds,  epochs=epochs)
img = tf.keras.preprocessing.image.load_img(os.getcwd()+"\\Testing\\BedRoom\\BedRoom1.JPG", target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
)
