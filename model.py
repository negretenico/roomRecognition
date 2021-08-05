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


train_datagen = ImageDataGenerator(
      featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical')


# data = Data()
# IMG_HEIGHT = data.img_height
# IMG_WIDTH = data.img_width
# train_ds = data.train_ds
# val_ds = data.val_ds
learning_rate = 1.00e-3

batch_size =1
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
        epochs = 50
        history = model.fit_generator(
      train_generator,  
      epochs=epochs,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
        model.save('saved_model/my_model')



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
