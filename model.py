import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

from GenerateData  import  DataGenerator
class CNN:
    def __init__(self):
        self.dg = DataGenerator()
        self.names = self.dg.categories
        pickle_in = open("trainX.pickle", "rb")
        self.images = pickle.load(pickle_in)

        pickle_in = open("testY.pickle", "rb")
        self.labels = pickle.load(pickle_in)
        self.data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(  self.dg.IMG_SIZE,
                                                                self.dg.IMG_SIZE,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
        self.batch_size =32
        self.model = Sequential([
            self.data_augmentation,
            layers.experimental.preprocessing.Rescaling(1. / 255),
            layers.Conv2D(8, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.names))
        ])
        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    def training(self,epochs):
        history =  self.model.fit(self.images, np.array(self.labels), batch_size=self.batch_size, epochs=epochs, validation_split=0.3)
        return history
    def evaluateModel(self):
        self.model.evaluate(self.images,np.array(self.labels))

    def viewPredictions(self):
        predictions = self.model.predict(self.images)
        plt.figure(figsize=(150, 150))
        classes = np.argmax(predictions, axis=1)
        self.images = self.images.squeeze()
        for i in range(len(classes)):
            plt.grid(False)
            plt.imshow( self.images[i], cmap=plt.cm.binary)
            plt.xlabel("Actual: " + self.names[self.labels[i]])
            plt.title("Prediction: " + self.names[classes[i]])
            plt.show()


def plot(epochs,history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()




model = CNN()
epochs = 50
history = model.training(epochs)
plot(epochs,history)

model.viewPredictions()