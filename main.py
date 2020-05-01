"""
Machine learning for simple image classification using Python and scikit-learn
Written by David Easton, 4/29/2020
"""
import sys, argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify images in a given dataset using a given estimator strategy')
    parser.add_argument("-d", "--dataset", required=True, metavar="<dataset filename>",  dest="dataset", 
        help="image corpus to serve as dataset for model training and testing \n Supported:\n  - SVHN\n  - MNIST") # TODO: Support for nargs > 1?
    parser.add_argument("-e", "--estimator", required=True, metavar="<sklearn estimator strategy>",  dest="estimator", 
        help="scikit-learn estimator algorithm used to classify images in dataset \n Supported:\n  - rfc (Random Forest Classifier)") # TODO: Support for nargs > 1?

    args = parser.parse_args()

    dataset = args.dataset
    estimator = args.estimator
    #print("d = {0} || e = {1}".format(dataset, estimator))
