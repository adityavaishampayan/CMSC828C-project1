#!/usr/bin/env python3

# importing required libraries
from utils import mnist_reader
Sfrom sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
from sklearn import metrics


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


if __name__ == "__main__":

    x_train, x_test, y_train_data, y_test_data = prep_data()
    x_train_pca, x_test_pca = run_incremental_PCA(x_train, x_test)

    start = time.time()
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=7)

    # Train the model using the training sets
    knn.fit(x_train_pca, y_train_data)

    print("Training time:", (time.time() - start))

    # Predict the response for test dataset
    y_pred = knn.predict(x_test_pca)

    print("Testing time:", (time.time() - start))

    # Import scikit-learn metrics module for accuracy calculation
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test_data, y_pred))


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
