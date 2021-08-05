from tensorflow import keras
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip,RandomZoom,RandomRotation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')


batch_size =32
img_height,img_width= (180,180)
data_dir = os.path.join(os.path.join(os.getcwd(),"data"),"train")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


path = 'saved_model/my_model'


categories =  ["BathRoom","BedRoom","Kitchen","LivingRoom"]
DIR= os.path.join(os.getcwd(),os.path.join("data","test"))
print(DIR)

if(os.path.isdir(os.path.join(os.getcwd(),path))):
    model = tf.keras.models.load_model(path)
    print('else')
    # Check its architecture
    model.summary()  
  

else:
    model = Sequential([

    Conv2D(128, (3,3), activation='relu', input_shape=(150,150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(momentum=0.9),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.2),
 
    Dense(4, activation='softmax')
    ])
    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['acc'])
    with tf.device('/GPU:0'):
        epochs=10
        history = model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=epochs
        )



total = 0
num_correct = 0
class_names = ["BathRoom","BedRoom","Kitchen","LivingRoom"]

for cat in categories:
    for file in os.listdir(os.path.join(DIR,cat)):
        total +=1
        img = tf.keras.preprocessing.image.load_img(os.path.join(os.path.join(DIR,cat),file), target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(f"This image {class_names[np.argmax(score)]}.\n")
        print(f"The true label is {cat}\n")
        if cat == class_names[np.argmax(score)]:
          num_correct +=1

print(f"Total Correct:{num_correct}\nTotal Tests:{total}\nAccuracy: {num_correct/total}")
