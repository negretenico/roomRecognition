import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random


class DataGenerator():
    def __init__(self):
        self.DATADIR = os.getcwd() + '\\Images'
        self.categories = ['Basement', 'BathRoom', 'Bed Room', 'Dinning Room', 'Kitchen', 'Living Room']
        self.trainingData = []
        self.generateTrainingData()
        random.shuffle(self.trainingData)
        self.labels = []
        self.imgs = []
        self.IMG_SIZE = 150

    def showImages(self):
        for cat in self.categories:
            # path for the folder
            path = os.path.join(self.DATADIR, cat)
            print(path)
            for images in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, images), cv2.IMREAD_GRAYSCALE)
                self.IMG_SIZE = 150
                newImg = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                plt.imshow(newImg, cmap="gray")
                plt.show()

    def generateTrainingData(self):
        for cat in self.categories:
            # path for the folder
            path = os.path.join(self.DATADIR, cat)
            classNum = self.categories.index(cat)
            for images in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, images), cv2.IMREAD_GRAYSCALE)
                    IMG_SIZE = 150
                    newImg = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    self.trainingData.append([newImg, classNum])
                except Exception as e:
                    pass

    def generateImagesAndLabels(self):
        for img, label in self.trainingData:
            self.labels.append(label)
            self.imgs.append(img)
        self.imgs = np.array(self.imgs).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)

        pickle_out = open("trainX.pickle", "wb")
        pickle.dump(self.labels, pickle_out)
        pickle_out.close()

        pickle_outY = open("testY.pickle", "wb")
        pickle.dump(self.labels, pickle_outY)
        pickle_out.close()


generator = DataGenerator()

generator.generateImagesAndLabels()
