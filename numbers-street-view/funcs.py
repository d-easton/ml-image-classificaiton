"""
Support functions for image classificaiton script
"""

import numpy

def vectorizeMatrix4D(matrix):
    """
    Take in a numpy matrix of 4 dimensions, return reshaped vector (one dimensional), then return transpose
    """
    matrix = matrix.reshape(matrix.shape[0]*matrix.shape[1]*matrix.shape[2]*matrix.shape[3])
    return matrix.T

def vectorizeLabels(labels):
    """
    Take in the label numpy array of 1 dimension, return 1 dimension reshaped vector -- no transpose
    """
    return labels.reshape(labels.shape[0],)