#!/usr/bin/env python3

# Standard library imports
import sys
import time

# Adds higher directory to python modules path.
sys.path.append("..") 

# Third party imports
from future.utils import iteritems
from scipy.stats import multivariate_normal as mvn
import numpy as np
from sklearn import metrics

# Local application imports
from utils import mnist_reader


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


class Bayes(object):
    def __init__(self):
        self.priors = dict()
        self.gaussian = dict()

    @staticmethod
    def mean(x):
        """
        returns mean of the data
        :param x: data vector
        :return: mean
        """
        mean_x = np.mean(x, axis=0)
        return mean_x

    @staticmethod
    def covariance(x):
        """
        returns covariance of the data
        :param x: data vector
        :return: covariance of the data
        """
        cov_x = np.cov(x.T)
        return cov_x

    def prior(self, labels):
        """
        this function calculates the priors for each category
        :param labels: category labels
        :return: None
        """
        for category in labels:
            self.priors[category] = {len(labels[labels == category]) / len(labels)}
        return 0

    def fit(self, data, y):
        """
        calculates associated mean and covariance of each class in the data-set
        :param data: data
        :param y : data labels
        :return: None
        """
        smoothing_factor = 1e-2
        samples, feature_length = data.shape

        labels = set(y)
        for category in labels:
            current_data = data[y == category]
            self.gaussian[category] = {
                "mean": current_data.mean(axis=0),
                "cov": np.cov(current_data.T)
                + np.eye(feature_length) * smoothing_factor,
            }
            self.priors[category] = float(len(y[y == category])) / len(y)
        return 0

    def predict(self, data):
        """
        this function predicts the class of an unknown feature vector
        :param data: feature vectors whose class has to be determined
        :return: class of the feature vector
        """
        samples, feature_length = data.shape
        k = len(self.gaussian)
        p = np.zeros((samples, k))

        for category, g in iteritems(self.gaussian):
            mean, covariance = g["mean"], g["cov"]
            p[:, category] = mvn.logpdf(data, mean=mean, cov=covariance) + np.log(
                self.priors[category]
            )

        return np.argmax(p, axis=1)

    def accuracy(self, data, labels):
        """
        returns the accuracy/ score of the prediction
        :param data: data
        :param labels: labels for each feature vector
        :return: score of the prediction
        """
        prediction = self.predict(data)
        return np.mean(prediction == labels)


def prep_data():
    """
    This function preps the dataset for further application
    :return: normalised test and train data
    """
    data_set = Dataset()
    x_train, y_train = data_set.load("../data/fashion", "train")
    x_test, y_test = data_set.load("../data/fashion", "t10k")

    shuffle_index = np.random.permutation(60000)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    
    x_train_norm = data_set.normalize(x_train)
    x_test_norm = data_set.normalize(x_test)

    return x_train_norm, x_test_norm, y_train, y_test


def main():

    # preparing the data set
    x_train, x_test, y_train_data, y_test_data = prep_data()

    # create the Bayes classifer
    model = Bayes()
    start = time.time()
    model.fit(x_train, y_train_data)
    print("Time required for training:", float(time.time() - start))

    start = time.time()
    print("Training accuracy:", model.accuracy(x_train, y_train_data))
    print(
        "Time required for computing train accuracy:",
        float(time.time() - start),
        "Training data size:",
        len(y_train_data),
    )

    start = time.time()
    print("Testing accuracy:", model.accuracy(x_test, y_test_data))
    print(
        "Time required for computing test accuracy:",
        float(time.time() - start),
        "Testing data set size:",
        len(y_test_data),
    )
    
    # Predict the response for test dataset
    y_pred = model.predict(x_test)
    print("Testing time:", (time.time() - start))
    
    # calculating accuracy of the classifier
    accuracy = metrics.accuracy_score(y_test_data, y_pred)
    print("accuracy of the classifier is: ", accuracy)

    # classification report includes precision, recall, F1-score
    print("classification report: \n")
    print(metrics.classification_report(y_test_data, y_pred))

    # average accuracy
    average_accuracy = np.mean(y_test_data == y_pred) * 100
    print("The average_accuracy is {0:.1f}%".format(average_accuracy))


if __name__ == "__main__":
    main()

