"""
Basic Python3.7 script for image classification model
    ML Estimator Strategies:
        - Random Forest Classifier
    Data Set:
        - Stanford's SVHN Database (32 x 32)
Written by David Easton, 4/29/2020
"""

import sys, os

# # import major libraries
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# import scikit learn funcs and algorithms
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

from support.funcs import *
#from support.variables import *

def classify():
    # assign dataset to variable
    train_data = scipy.io.loadmat('./datasets/train_32x32.mat')
    test_data = scipy.io.loadmat('./datasets/test_32x32.mat')

    # Extract X (4D matrix of 32x32 images) and Y (vector of class labels)
    train_images = train_data['X']
    train_labels = train_data['y']
    test_images = test_data['X']
    test_labels = test_data['y']

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

    # Assess accuracy score
    model_estimate = classifier.predict(test_images)
    print(f"Accuracy: %d", accuracy_score(test_labels, model_estimate))