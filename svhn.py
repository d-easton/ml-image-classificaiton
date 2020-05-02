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
from sklearn.model_selection import train_test_split

from helpers.funcs import *
from helpers.variables import ModelSize

def classify_svhn(dataset, estimator, model_size):

    #TODO: multiple estimators
    # introduce random forest classifier and other vars
    classifier = RandomForestClassifier()
    train_data, test_data = []
    train_images, train_labels, test_images, test_labels = []
    
    # based on the recieved model size, open relevant svhn files and prepare dataset variables      # assign dataset to variable
    if model_size == ModelSize.SMALL:
        # For small model sizes, do all the normal steps independently
        train_data = scipy.io.loadmat('./datasets/train_32x32.mat')

        # Extract X (4D matrix of 32x32 images) and Y (vector of class labels)
        train_images = train_data['X']
        train_labels = train_data['y']

        # vectorize
        train_images = vectorizeMatrix4D(train_images)
        train_labels = vectorizeLabels(train_labels)

        # shuffle and split trianing dataset into testing subset and smaller testing subset
        train_images, train_labels = shuffle(train_images, train_labels)
        train_images, test_images, train_labels, test_images = train_test_split(train_images, train_labels, test_size=.15)
    else:
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

    # Train RFC classifier
    classifier.fit(train_images, train_labels)

    # if data model is large, open additonal training dataset and train further
    if model_size == ModelSize.LARGE:
        extra_data = scipy.io.loadmat('./datasets/extra_32x32.mat')
        
        # Extract X (4D matrix of 32x32 images) and Y (vector of class labels)
        extra_images = extra_data['X']
        extra_labels = extra_data['y']

        # vectorize & shuffle
        extra_images = vectorizeMatrix4D(extra_images)
        extra_labels = vectorizeLabels(extra_labels)
        extra_images, extra_labels = shuffle(extra_images, extra_labels)

        # subject RFC classifier to addional training
        classifier.fit(extra_images, extra_labels)
        
    # Assess accuracy score
    model_estimate = classifier.predict(test_images)
    print(f"Accuracy: %d", accuracy_score(test_labels, model_estimate))