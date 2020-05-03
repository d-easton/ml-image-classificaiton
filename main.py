"""
Machine learning for simple image classification using Python and scikit-learn
Written by David Easton, 4/29/2020
"""

from helpers.variables import ModelSize
import sys
from cli import main as process_args
from cli.errors import UnsupportedArgumentError
from svhn import classify_svhn

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from svhn.classify import *

if __name__ == "__main__":
    try:
        args = process_args()
        if "svhn" in args:
            if "rfc" in args:
                print("Inputs well recieved")
                classify_svhn("svhn", "rfc", ModelSize.SMALL)
        else:
            raise UnsupportedArgumentError(f"arguments unrecognized following parse")

    except UnsupportedArgumentError as err:
        print(f"Error (Unsupported Argument): {err}")
        sys.exit(1)
    
