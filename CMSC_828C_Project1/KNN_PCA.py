# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:23:24 2019

@author: Aditya's HP Omen 15
"""

#!/usr/bin/env python3

# importing required libraries
from utils import mnist_reader
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


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
        data_vector.astype("float32")
        normalised_data = data_vector / 255
        return normalised_data

def prep_data():
    """
    This function preps the data set for further application
    :return: normalised test and train data
    """
    data_set = Dataset()
    x_train, y_train = data_set.load("data/fashion", "train")
    x_test, y_test = data_set.load("data/fashion", "t10k")
    
    shuffle_index = np.random.permutation(60000)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

    x_train_norm = data_set.normalize(x_train)
    x_test_norm = data_set.normalize(x_test)

    return x_train_norm, x_test_norm, y_train, y_test


def run_PCA(train_data, test_data):
    """
    This function performs PCA on data set and reduces its dimensionality
    :param train_data: train data for PCA dimensionality reduction
    :param test_data: test data for PCA dimensionality reduction
    :return: train and test data after applying PCA
    """
    pca = PCA()
    pca.fit(train_data)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print("no. of dimensions: ", d)
    pca = PCA(n_components=187)
    x_train_pca = pca.fit_transform(train_data)
    x_test_pca = pca.fit_transform(test_data)

    return x_train_pca, x_test_pca


def run_incremental_PCA(train_data, test_data, n_batches=50):
    """
    :param train_data: train_data: train data for incremental PCA dimensionality reduction
    :param test_data: test data for incremental PCA dimensionality reduction
    :param n_batches: default parameter
    :return: train and test data after applying incremental PCA
    """

    inc_pca = IncrementalPCA(n_components=187)
    for X_batch in np.array_split(train_data, n_batches):
        inc_pca.partial_fit(X_batch)
    x_train_pca_inc = inc_pca.transform(train_data)

    for X_batch in np.array_split(test_data, n_batches):
        inc_pca.partial_fit(X_batch)
    x_test_pca_inc = inc_pca.transform(test_data)

    return x_train_pca_inc, x_test_pca_inc

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

def cf_matrix(model, X, Y):
    #y_train_pred = cross_val_predict(knn, x_train_pca, y_train_data, cv=3)
    y_train_pred = cross_val_predict(model, X, Y, cv=3)
    conf_mx =confusion_matrix(Y, y_train_pred)
    print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    return conf_mx, y_train_pred

def cf_matrix_norm(cfm):
    row_sums = cfm.sum(axis=1, keepdims=True)
    norm_conf_mx = cfm / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    return norm_conf_mx
    
def plot_images(a,b,x_train, y_train, y_pred):
    cl_a, cl_b = 6, 2
    X_aa = x_train[(y_train == cl_a) & (y_pred == cl_a)]
    X_ab = x_train[(y_train == cl_a) & (y_pred == cl_b)]
    X_ba = x_train[(y_train == cl_b) & (y_pred == cl_a)]
    X_bb = x_train[(y_train == cl_b) & (y_pred == cl_b)]
    plt.figure(figsize=(8,8))
    plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
    plt.show()
 
   
def main():
    x_train, x_test, y_train_data, y_test_data = prep_data()
    x_train_pca, x_test_pca = run_incremental_PCA(x_train, x_test)
     
    start = time.time()
    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    #Train the model using the training sets
    knn.fit(x_train_pca, y_train_data)
     
    print("Training time:", (time.time() - start))
     
    #Predict the response for test dataset
    y_pred = knn.predict(x_test_pca)
     
    print("Testing time:", (time.time() - start))
 
    #Import scikit-learn metrics module for accuracy calculation
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test_data, y_pred))
    print("classification report: \n")
    print(metrics.classification_report(y_test_data, y_pred))

    accuracy = metrics.accuracy_score(y_test_data, y_pred)
    print("accuracy of the classifier is: ", accuracy )
    average_accuracy = np.mean(y_test_data == y_pred) * 100
    print("The average_accuracy is {0:.1f}%".format(average_accuracy))
    
    
    cf, y_train_pred = cf_matrix(knn, x_train_pca, y_train_data)
    norm_cf  = cf_matrix_norm(cf)
    plt.matshow(norm_cf, cmap=plt.cm.gray)
    plt.show()
    
    cl_a, cl_b = 6, 2
    plot_images(cl_a,cl_b,x_train, y_train_data, y_train_pred)
    
   


if __name__ == "__main__":
    main()

    
# =============================================================================
#     plt.figure(figsize=(9,9))
#     example_images = np.r_[x_train[:12000:600], x_train[13000:30600:600], x_train[30600:60000:590]]
#     plot_digits(example_images, images_per_row=10)
#     plt.show()
# =============================================================================
    
# =============================================================================
#     X_aa = x_train[(y_train_data == cl_a) & (y_train_pred == cl_a)]
#     X_ab = x_train[(y_train_data == cl_a) & (y_train_pred == cl_b)]
#     X_ba = x_train[(y_train_data == cl_b) & (y_train_pred == cl_a)]
#     X_bb = x_train[(y_train_data == cl_b) & (y_train_pred == cl_b)]
#     plt.figure(figsize=(8,8))
#     plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
#     plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
#     plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
#     plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
#     plt.show()
#     
# =============================================================================
# =============================================================================
#     start = time.time()
#     model.fit(x_train_pca, y_train_data)
#     print("Training time:", (time.time() - start))
# 
#     start = time.time()
#     print("Train accuracy:", model.accuracy(x_train_pca, y_train_data))
#     print(
#         "Time to compute train accuracy:",
#         (time.time() - start),
#         "Train size:",
#         len(y_train_data),
#     )
# 
#     start = time.time()
#     print("Test accuracy:", model.accuracy(x_test_pca, y_test_data))
#     print(
#         "Time to compute test accuracy:",
#         (time.time() - start),
#         "Test size:",
#         len(y_test_data),
#     )
# 
# =============================================================================
