"""
Basic Python3.7 script that uses random forest estimator from the scikit-learn machine learning module to train and test image classification model
Written by David Easton, 4/29/2020
"""
# import major libraries
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# import scikit learn funcs and algorithms
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

from .funcs import *

# assign dataset to variable
train_data = scipy.io.loadmat('./datasets/train_32x32.mat')
test_data = scipy.io.loadmat('./datasets/test_32x32.mat')

# Extract X (4D matrix of 32x32 images) and Y (vector of class labels)
train_images = train_data['X']
train_labels = train_data['y']
test_images = test_data['X']
test_labels = test_data['y']

# test image display
# index = 25
# plt.imshow(images[:,:,:,index])
# plt.show()
# print(labels[index])

# vectorize 
train_images = vectorizeMatrix4D(train_images)
train_labels = vectorizeLabels(train_labels)
test_images = vectorizeMatrix4D(test_images)
test_labels = vectorizeLabels(test_labels)


# shuffle dataset with sklearn shuffle to avoid pre-distribution biases
# TODO: split to two calls for different random state?
train_images, train_labels, test_images, test_labels = shuffle(train_images, train_labels, test_images, test_labels)

# implement random forest classifier
classifier = RandomForestClassifier()

# Currently, we don't need to split because Stanford gives us two data sets. But we could use this line to split "data" alone for 
#   a faster model run on a server with fewer resources
# training_images, testing_images, training_labels, testing_labels = train_test_split(images, labels)
classifier.fit(train_images, train_labels)

# TODO: add accuracy score