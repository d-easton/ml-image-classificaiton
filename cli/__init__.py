"""
Command line support package
Written by David Easton, 5/1/2020
"""

import sys, argparse

from .errors import UnsupportedArgumentError

supported_datasets = [
    "svhn",
    "mnist"
]

def main():
    # for arg in sys.argv:
    #     if arg not in supported_datasets and arg not in supported_estimators:
    #         pass
    parser = argparse.ArgumentParser(description='Classify images in a given dataset using a given estimator strategy')
    parser.add_argument("-d", "--dataset", required=True, metavar="<dataset filename>",  dest="dataset", 
        help="image corpus to serve as dataset for model training and testing \n Supported:\n  - SVHN\n  - MNIST") # TODO: Support for nargs > 1?
    parser.add_argument("-e", "--estimator", required=True, metavar="<sklearn estimator strategy>",  dest="estimator", 
        help="scikit-learn estimator algorithm used to classify images in dataset \n Supported:\n  - rfc (Random Forest Classifier)") # TODO: Support for nargs > 1?

    args = parser.parse_args()

    dataset = args.dataset.lower().strip()
    estimator = args.estimator.lower().strip()

    if dataset == "svhn":
        if estimator == "rfc":
            #print(f"inputs well recieved -- d: {dataset}  e: {estimator}")
            return dataset, estimator
        else:
            raise UnsupportedArgumentError(f"unsupported estimator '{estimator}' for dataset {dataset}") 

    elif dataset == "mnist":
         raise UnsupportedArgumentError(f"support for '{dataset}' coming soon")
    else:
        raise UnsupportedArgumentError(f"unsupported dataset '{dataset}'")