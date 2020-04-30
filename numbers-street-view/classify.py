"""
Basic Python3.7 script that uses scikit-learn machine learning module to train and test image classification model
Written by David Easton, 4/29/2020
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from .funcs import *

# assign dataset to variable
data = scipy.io.loadmat('./datasets/train_32x32.mat')

# Extract X (4D matrix of 32x32 images) and Y (vector of class labels)
images = data['X']
labels = data['y']

# test image display
# index = 25
# plt.imshow(images[:,:,:,index])
# plt.show()
# print(labels[index])

# vectorize 
images = vectorizeMatrix4D(images)
labels = vectorizeLabels(labels)

# shuffle dataset with sklearn shuffle to avoid pre-distribution biases
images, labels = shuffle(images, labels)