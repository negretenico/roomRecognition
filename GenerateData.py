import numpy as np
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image


class Data:
    def normalize_img(self,image, label):
          """Normalizes images: `uint8` -> `float32`."""
          return tf.cast(image, tf.float32) / 255., label

    def __init__(self):
        data_dir = os.getcwd() +"\\Images"

        self.img_height,self.img_width = 180,180
        batch_size = 32
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                color_mode = "grayscale",
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=batch_size)
        self.class_names = self.train_ds.class_names
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            color_mode = "grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size)
        
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    

#convertes from jpg to JPEG
# def convert_to_JPEG(dir):
#     for cat in categories:
#         #path for the folder
#         path = os.path.join(dir,cat)
#         print(path)
#         count = 0
#         for filename in os.listdir(path):
#             if filename.endswith(".jpg"):
#                 try:
#                     im = Image.open(os.path.join(path,filename))
#                     name = filename[:-4]+".JPEG"
#                     rgb_im = im.convert('RGB')
#                     rgb_im.save(os.path.join(path,name))
#                     count +=1
#                 except:
#                     print("Error occured")
#         print(f"{count} converted from png/jpg to JPEG in {cat}")
