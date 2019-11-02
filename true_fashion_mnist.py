
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from utils import mnist_reader
from matplotlib import pyplot as plt


# In[2]:


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


# In[3]:


X_train = X_train.astype('float32') #images loaded in as int64, 0 to 255 integers
X_test = X_test.astype('float32')
# Normalization
X_train /= 255
X_test /= 255


# In[4]:


plt.figure(figsize=(12,10))# Showing the Input Data after Normalizing
x, y = 4, 4
for i in range(15):  
    plt.subplot(y, x, i+1)
    plt.imshow(X_train[i].reshape((28,28)),interpolation='nearest')
plt.show()


# In[5]:


import matplotlib
import matplotlib.pyplot as plt
some_item = X_train[9000]
some_item_image = some_item.reshape(28, 28)
plt.imshow(some_item_image, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()


# In[6]:


y_train[9000]


# In[7]:


from future.utils import iteritems
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D)*smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    #X, Y = get_data(10000)
    #Ntrain = len(Y) // 2
    #Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    #Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    Xtrain = X_train
    Xtest = X_test
    Ytrain =y_train
    Ytest = y_test
    
    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    # plot the mean of each class
    for c, g in iteritems(model.gaussians):
        plt.imshow(g['mean'].reshape(28, 28))
        plt.title(c)
        plt.show()


# In[8]:


# with PCA

from sklearn.decomposition import PCA


# In[9]:


pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1


# In[10]:


d


# In[11]:


pca = PCA(n_components = 187)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.fit_transform(X_test)


# In[12]:


############################
# AFTER PCA
############################

from future.utils import iteritems
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D)*smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    #X, Y = get_data(10000)
    #Ntrain = len(Y) // 2
    #Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    #Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    Xtrain = X_train_reduced
    Xtest = X_test_reduced
    Ytrain =y_train
    Ytest = y_test
    
    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    # plot the mean of each class
#     for c, g in iteritems(model.gaussians):
#         plt.imshow(g['mean'].reshape(28, 28))
#         plt.title(c)
#         plt.show()


# In[13]:


from sklearn.decomposition import IncrementalPCA
n_batches = 50
#n_bacthes should be greater than or equal to dimesions

inc_pca = IncrementalPCA(n_components=187)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
X_train_mnist_reduced_inc = inc_pca.transform(X_train)

for X_batch in np.array_split(X_test, n_batches):
    inc_pca.partial_fit(X_batch)
X_test_mnist_reduced_inc = inc_pca.transform(X_test)


# In[14]:


############################
# AFTER INCREMENTAL PCA
############################

from future.utils import iteritems
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D)*smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    #X, Y = get_data(10000)
    #Ntrain = len(Y) // 2
    #Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    #Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    Xtrain = X_train_mnist_reduced_inc
    Xtest = X_test_mnist_reduced_inc
    Ytrain =y_train
    Ytest = y_test
    
    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    # plot the mean of each class
#     for c, g in iteritems(model.gaussians):
#         plt.imshow(g['mean'].reshape(28, 28))
#         plt.title(c)
#         plt.show()


# In[15]:


#####################################

# LDA

#####################################
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)
X_train_LDA = lda.fit_transform(X_train, y_train)
X_test_LDA = lda.transform(X_test)


# In[16]:


############################
# AFTER implementing LDA
############################

from future.utils import iteritems
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D)*smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    #X, Y = get_data(10000)
    #Ntrain = len(Y) // 2
    #Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    #Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    Xtrain = X_train_LDA
    Xtest = X_test_LDA
    Ytrain = y_train
    Ytest = y_test
    
    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    # plot the mean of each class
#     for c, g in iteritems(model.gaussians):
#         plt.imshow(g['mean'].reshape(28, 28))
#         plt.title(c)
#         plt.show()


# In[17]:


#############################
#IMPLEMENTING for KNN
############################

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


# In[18]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[19]:


import pickle


# In[20]:


filename = 'finalized_model.sav'
pickle.dump(knn, open(filename, 'wb'))


# In[ ]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

