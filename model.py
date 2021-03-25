import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, \
    GlobalAveragePooling2D

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
        self.batch_size =64
        self.model = Sequential([ self.data_augmentation])

        self.model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(self.dg.IMG_SIZE, self.dg.IMG_SIZE, 3)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(96, (3, 3), activation='relu', padding='same', strides=2))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(192, (3, 3), activation='relu', padding='same', strides=2))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(192, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(192, (1, 1), padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(10, (1, 1), padding='valid'))

        self.model.add(GlobalAveragePooling2D())

        self.model.add(Activation('softmax'))

        self.model.summary()
        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    def training(self,epochs):
        history =  self.model.fit(self.images, np.array(self.labels), batch_size=self.batch_size, epochs=epochs, validation_split=0.3)
        return history
    def evaluate(self):
        self.model.evaluate(self.images,np.array(self.labels))



    def viewPredictions(self):
        predictions = self.self.model.predict(self.images)
        plt.figure(figsize=(150, 150))
        classes = np.argmax(predictions, axis=1)
        self.images = self.images.squeeze()
        for i in range(len(classes)):
            plt.grid(False)
            plt.imshow( self.images[i], cmap=plt.cm.binary)
            plt.xlabel("Actual: " + self.names[self.labels[i]])
            plt.title("Prediction: " + self.names[classes[i]])
            plt.show()
    def save(self, path):
        self.self.model.save(path)
    def predict(self,images):
        return self.model.predict(images)

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



def test(model):
    pickle_in = open("testingImages.pickle", "rb")
    images = pickle.load(pickle_in)
    pickle_in = open("testingLabels.pickle", "rb")
    labels = pickle.load(pickle_in)
    names = ['Basements', 'BathRoom', 'BedRoom', 'DiningRoom', 'Kitchen', 'Living Room']
    predictions = model.predict(images)
    classes = np.argmax(predictions, axis=1)
    for i in range(len(classes)):
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual: " + names[labels[i]])
        plt.title("Prediction: " + names[classes[i]])
        plt.show()

# model = tf.keras.models.load_model('saved_model/new_model')
model = CNN()
epochs = 300
history = model.training(epochs)
plot(epochs,history)
#self.model.viewPredictions()
model.model.save('saved_model/new_model1')
test(model)

