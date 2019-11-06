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
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


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

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train_norm)
x_test_scaled = sc.transform(x_test_norm)

lda = LDA(n_components=1)
x_train_LDA = lda.fit_transform(x_train_scaled, y_train)
x_test_LDA = lda.transform(x_test_scaled)


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
                'mean': current_data.mean(axis=0),
                'cov': np.cov(current_data.T) + np.eye(feature_length) * smoothing_factor
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
            mean, covariance = g['mean'], g['cov']
            p[:, category] = mvn.logpdf(data, mean=mean, cov=covariance) + np.log(self.priors[category])

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


if __name__ == '__main__':
    model = Bayes()
    t0 = datetime.now()
    model.fit(x_train_LDA, y_train)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.accuracy(x_train_LDA, y_train))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(y_train))

    t0 = datetime.now()
    print("Test accuracy:", model.accuracy(x_test_LDA, y_test))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(y_test))
