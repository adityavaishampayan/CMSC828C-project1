#!/usr/bin/env python3

# Standard library imports
import sys
import time

# Adds higher directory to python modules path
sys.path.append("..") 

# importing required libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

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


def prep_data():
    """
    This function preps the dataset for further application
    :return: normalised test and train data
    """
    data_set = Dataset()
    x_train, y_train = data_set.load("../data/fashion", "train")
    x_test, y_test = data_set.load("../data/fashion", "t10k")
    
    shuffle_index = np.random.RandomState(seed=42).permutation(60000)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

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


def plot_digits(instances, images_per_row=10, **options):
    """
    This function plots the images
    :param instances:
    :param images_per_row: images per row
    :param options:
    :return: plots the image
    """
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
    """
    This function calculates the confusion matrix given the data and the labels
    :param model: type of model e.g. knn
    :param X: training data
    :param Y: data labels
    :return: confusion matrix and predictions on test data
    """
    y_train_pred = cross_val_predict(model, X, Y, cv=3)
    conf_mx = confusion_matrix(Y, y_train_pred)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    return conf_mx, y_train_pred


def cf_matrix_norm(cfm):
    """
    This function normalizes the confusion matrix
    :param cfm: confusion matrix
    :return: normalized confusion matrix
    """
    row_sums = cfm.sum(axis=1, keepdims=True)
    norm_conf_mx = cfm / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    return norm_conf_mx


def plot_images(a,b,x_train, y_train, y_pred):
    """
    This function plots the images in a 5x5 grid
    :param a: true class label
    :param b: predicted class label
    :param x_train: training data
    :param y_train: training data labels
    :param y_pred: prediction on the test data
    :return: None
    """
    cl_a = a
    cl_b = b
    x_aa = x_train[(y_train == cl_a) & (y_pred == cl_a)]
    x_ab = x_train[(y_train == cl_a) & (y_pred == cl_b)]
    x_ba = x_train[(y_train == cl_b) & (y_pred == cl_a)]
    x_bb = x_train[(y_train == cl_b) & (y_pred == cl_b)]
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plot_digits(x_aa[:25], images_per_row=5)
    plt.subplot(222)
    plot_digits(x_ab[:25], images_per_row=5)
    plt.subplot(223)
    plot_digits(x_ba[:25], images_per_row=5)
    plt.subplot(224)
    plot_digits(x_bb[:25], images_per_row=5)
    plt.show()

def cross_val(model, X, Y):
    scores = cross_val_score(model, X, Y,
                             scoring="accuracy", cv=10)
    return scores

def display_scores(scores):
    print("Scores: " + str(scores) + "\n")
    print("Mean: " + str(scores.mean()) + "\n")
    print("Standard deviation: " + str(scores.std()) + "\n")
        
    
def main():
    # preparing the data set
    x_train, x_test, y_train_data, y_test_data = prep_data()
    
    # running LDA on the data set
    x_LDA_train, x_LDA_test = run_LDA(x_train, x_test, y_train_data, y_test_data)
    
    start = time.time()
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Train the model using the training sets
    knn.fit(x_LDA_train, y_train_data)
    print("Training time:", (time.time() - start))
    
    #running cross validation on dataset
    corss_val_score = cross_val(knn, x_LDA_train, y_train_data)
    print("displaying cross validation scores: ")
    display_scores(corss_val_score)
    
    # Predict the response for test dataset
    y_pred = knn.predict(x_LDA_test)
    print("Testing time:", (time.time() - start))

    # Import scikit-learn metrics module for accuracy calculation
    print("Accuracy:", metrics.accuracy_score(y_test_data, y_pred))

    # calculating accuracy of the classifier
    accuracy = metrics.accuracy_score(y_test_data, y_pred)
    print("accuracy of the classifier is: ", accuracy)

    # classification report includes precision, recall, F1-score
    print("classification report: \n")
    print(metrics.classification_report(y_test_data, y_pred))

    # average accuracy
    average_accuracy = np.mean(y_test_data == y_pred) * 100
    print("The average_accuracy is {0:.1f}%".format(average_accuracy))

    # calculating the confusion matrix
    cf, y_train_pred = cf_matrix(knn, x_LDA_train, y_train_data)

    # normalizing the confusion matrix and plotting it
    norm_cf = cf_matrix_norm(cf)
    plt.matshow(norm_cf, cmap=plt.cm.gray)
    plt.show()
    
    # try K=1 through K=25 and record testing accuracy
    k_range = range(1, 100)
    
    # We can create Python dictionary using [] or dict()
    scores = dict()
    plot_scores = []
    
    # We use a loop through the range 1 to 26
    # We append the scores in the dictionary
    for k in k_range:
        print("k = ", k)
        knn = KNeighborsClassifier(n_neighbors=k)
        start = time.time()
        knn.fit(x_LDA_train, y_train_data)
        y_pred = knn.predict(x_LDA_test)
        print("time required: " + str(time.time() - start) + "\n")
        scores.update({k:metrics.accuracy_score(y_test_data, y_pred)})
        plot_scores.append(metrics.accuracy_score(y_test_data, y_pred))

    print(scores)
    
    # plot the relationship between K and testing accuracy
    # plt.plot(x_axis, y_axis)
    plt.plot(k_range, plot_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()
    

if __name__ == "__main__":
    main()

