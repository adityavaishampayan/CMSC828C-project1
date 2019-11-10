[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# CMSC828C-project-1

Implementated Maximum Likelihood Estimation (MLE) with Gaussian assumption followed by Bayes rule for classification. Also implemented the same after applying PCA and LDA for dimensionality reduction

## Getting Started


### Prerequisites

What things you need to install the software and how to install them

To run this project one needs to set up a virtual environment. 
#### For Anaconda users it can be done as follows:

```
conda update conda
```
```
conda create -n yourenvname python=x.x anaconda
```
Press y to proceed. This will install the Python version and all the associated anaconda packaged libraries at “path_to_your_anaconda_location/anaconda/envs/yourenvname”

To activate the virtual environment:
```
source activate yourenvname
```
Install git for cloning the repo:
```
conda install -c anaconda git
```
#### For ubuntu users:

Install dependencies
```
sudo apt install python-pip
```
Create virtual environment directory
```
mkdir /home/yourfoldername
cd /home/yourfoldername
```
Install virtualenv
```
pip install virtualenv
```
Create a virtual environment
```
virtualenv envname
```

### Installing

Once in the virtual environemnet:

* To clone this repo in your virtual environment:
```
git clone https://github.com/adityavaishampayan/CMSC828C-project1.git
```
* cd to the CMSC828C-project1 folder and then install the requirements file 
```
pip install -r requirements.txt
```
This will install all the required libraries to run this project

# Data

This project required the Fashion_MNIST dataset to run. The data has already be added in this repo.

However to clone the original repository one can use the following command:
```
git clone git@github.com:zalandoresearch/fashion-mnist.git
```

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.


## Content

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

## Labels

Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot | 

## Authors

* **Aditya Vaishampayan** - [AdityaVaishampayan](https://github.com/adityavaishampayan)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## 5. References
[1] [sklearn.LinearDiscriminantAnalysis](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis.fit)  
[2] [sklearn.PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
[3] [sklearn.GaussianNB](http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)  
[4] [How to Get 97% on MNIST with KNN](https://steven.codes/blog/ml/how-to-get-97-percent-on-MNIST-with-KNN/)  
[5] [fit() and transform() methods](https://stackoverflow.com/questions/23838056/what-is-the-difference-between-transform-and-fit-transform-in-sklearn)  
[6] [PCA returned negative values](https://stackoverflow.com/questions/34725726/is-it-possible-apply-pca-on-any-text-classification)  
[7] [tutorial on Bayes basic](https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/)  
[8] [Naive Bayes on MNIST dataset](https://github.com/bikz05/ipython-notebooks/blob/master/machine-learning/naive-bayes-mnist-sklearn.ipynb)  
[9] [MNIST HW u-brown](http://cs.brown.edu/courses/csci1950-f/fall2009/docs/wk04.pdf)
