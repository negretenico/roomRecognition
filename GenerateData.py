import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

class DataGenerator():
    def __init__(self):
        self.DATADIR = os.getcwd() + '\\Images'
        self.TESTING = os.getcwd()+'\\Testing'
        self.categories = ['Basements', 'BathRoom', 'BedRoom', 'DiningRoom', 'Kitchen', 'Living Room']
        self.trainingData = []
        self.testingData = []
        self.generateTrainingData()
        self.generateTestingData()
        random.shuffle(self.trainingData)
        random.shuffle(self.testingData)
        self.labels = []
        self.imgs = []
        self.testingLabels = []
        self.testingImgs = []
        self.IMG_SIZE = 150

        self.generateImagesAndLabels()
        self.imagesAndLabelsForTesting()


    def showImages(self):
        for cat in self.categories:
            # path for the folder
            path = os.path.join(self.DATADIR, cat)
            print(path)
            for images in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, images))
                self.IMG_SIZE = 150
                newImg = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                plt.imshow(newImg)
                plt.show()

    def generateTrainingData(self):
        for cat in self.categories:
            # path for the folder
            path = os.path.join(self.DATADIR, cat)
            classNum = self.categories.index(cat)
            for images in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, images))
                    IMG_SIZE = 150
                    newImg = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    self.trainingData.append([newImg, classNum])
                except Exception as e:
                    pass

    def generateImagesAndLabels(self):
        for img, label in self.trainingData:
            self.labels.append(label)
            self.imgs.append(img)
        self.imgs = np.array(self.imgs)

        pickle_out = open("trainX.pickle", "wb")
        pickle.dump(self.imgs, pickle_out)
        pickle_out.close()

        pickle_outY = open("testY.pickle", "wb")
        pickle.dump(self.labels, pickle_outY)
        pickle_out.close()
    def imagesAndLabelsForTesting(self):
        for img, label in self.testingData:
            self.testingLabels.append(label)
            self.testingImgs.append(img)
        self.testingImgs = np.array(self.testingImgs)

        pickle_out = open("testingImages.pickle", "wb")
        pickle.dump(self.testingImgs, pickle_out)
        pickle_out.close()

        pickle_outY = open("testingLabels.pickle", "wb")
        pickle.dump(self.testingLabels, pickle_outY)
        pickle_out.close()
    def generateTestingData(self):

        for cat in self.categories:
            # path for the folder
            path = os.path.join(self.TESTING, cat)
            classNum = self.categories.index(cat)
            for images in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, images))
                    IMG_SIZE = 150
                    newImg = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    plt.imshow(newImg)
                    self.testingData.append([newImg, classNum])
                except Exception as e:
                    pass
dg = DataGenerator()
