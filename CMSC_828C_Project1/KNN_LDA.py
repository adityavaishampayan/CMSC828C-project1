#!/usr/bin/env python3

# importing required libraries
import numpy as np
import time
from utils import mnist_reader
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
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
    This function preps the dataset for further application
    :return: normalised test and train data
    """
    data_set = Dataset()
    x_train, y_train = data_set.load("data/fashion", "train")
    x_test, y_test = data_set.load("data/fashion", "t10k")

    x_train_norm = data_set.normalize(x_train)
    x_test_norm = data_set.normalize(x_test)

    return x_train_norm, x_test_norm, y_train, y_test


def run_LDA(train_data, test_data, y_train, y_test):
    """
    This function performs LDA on dataset and reduces its dimensionality
    :param train_data: train data for LDA dimensionality reduction
    :param test_data: test data for LDA dimensionality reduction
    :param y_train: training data labels
    :param y_test: testing data labels
    :return: train and test data with reduced dimensionality
    """
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(train_data)
    x_test_scaled = sc.transform(test_data)
    lda = LDA(n_components=1)
    x_train_LDA = lda.fit_transform(x_train_scaled, y_train)
    x_test_LDA = lda.transform(x_test_scaled)

    return x_train_LDA, x_test_LDA


def main():
    """
    This is the main function that calls sub functions
    :return: none
    """

    x_train, x_test, y_train_data, y_test_data = prep_data()
    x_LDA_train, x_LDA_test = run_LDA(x_train, x_test, y_train_data, y_test_data)

    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model using the training sets
    knn.fit(x_LDA_train, y_train_data)

    print("Training time:", (time.time() - start))

    # Predict the response for test dataset
    y_pred = knn.predict(x_LDA_test)

    print("Testing time:", (time.time() - start))

    # Import scikit-learn metrics module for accuracy calculation
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test_data, y_pred))


if __name__ == "__main__":
    main()


# =============================================================================
#     start = time.time()
#     model.fit(x_LDA_train, y_train_data)
#     print("Time required for training:", float(time.time() - start))
#
#     start = time.time()
#     print("Training accuracy:", model.accuracy(x_LDA_train, y_train_data))
#     print(
#         "Time required for computing train accuracy:",
#         float(time.time() - start),
#         "Training data size:",
#         len(y_train_data),
#     )
#
#     start = time.time()
#     print("Testing accuracy:", model.accuracy(x_LDA_test, y_test_data))
#     print(
#         "Time required for computing test accuracy:",
#         float(time.time() - start),
#         "Testing dataset size:",
#         len(y_test_data),
#     )
# =============================================================================


# =============================================================================
# if __name__ == '__main__':
#     model = Bayes()
#     t0 = datetime.now()
#     model.fit(x_train_LDA, y_train)
#     print("Training time:", (datetime.now() - t0))
#
#     t0 = datetime.now()
#     print("Train accuracy:", model.accuracy(x_train_LDA, y_train))
#     print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(y_train))
#
#     t0 = datetime.now()
#     print("Test accuracy:", model.accuracy(x_test_LDA, y_test))
#     print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(y_test))
# =============================================================================
