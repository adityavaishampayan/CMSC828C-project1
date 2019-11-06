#!/usr/bin/env python3

# importing required libraries
from matplotlib import pyplot as plt
from utils import mnist_reader
from future.utils import iteritems
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib
from sklearn.neighbors import KNeighborsClassifier


class Dataset(object):

    def __init__(self):
        pass

    def load(self, folder_path, data_type):
        """
        This function loads the data-set
        :param folder_path: path to data-set folder
        :param data_type: train or test data
        :return: data and labels
        """
        train_data, test_data = mnist_reader.load_mnist(folder_path, kind=data_type)
        return train_data, test_data

    def normalize(self, data_vector):
        """
        This function normalizes the data
        :param data_vector: data to be normalised
        :return: normalised data
        """
        data_vector.astype('float32')
        normalised_data = (data_vector / 255)
        return normalised_data


data_set = Dataset()
x_train, y_train = data_set.load('data/fashion', 'train')
x_test, y_test = data_set.load('data/fashion', 't10k')

x_train_norm = data_set.normalize(x_train)
x_test_norm = data_set.normalize(x_test)

#############################
#IMPLEMENTING for KNN
############################

#Import knearest neighbors Classifier model

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(x_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#
# X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
# X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
#
# X_train = X_train.astype('float32') #images loaded in as int64, 0 to 255 integers
# X_test = X_test.astype('float32')
# # Normalization
# X_train /= 255
# X_test /= 255

# plt.figure(figsize=(12,10))# Showing the Input Data after Normalizing
# x, y = 4, 4
# for i in range(15):
#     plt.subplot(y, x, i+1)
#     plt.imshow(X_train[i].reshape((28,28)),interpolation='nearest')
# plt.show()

# some_item = X_train[9000]
# # some_item_image = some_item.reshape(28, 28)
# # plt.imshow(some_item_image, cmap = matplotlib.cm.binary,interpolation="nearest")
# # plt.axis("off")
# # plt.show()