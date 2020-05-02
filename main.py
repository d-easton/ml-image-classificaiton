"""
Machine learning for simple image classification using Python and scikit-learn
Written by David Easton, 4/29/2020
"""

import sys
from cli import main as process_args
from cli.errors import UnsupportedArgumentError

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from svhn.classify import *

if __name__ == "__main__":
    try:
        process_args()
    except UnsupportedArgumentError as err:
        print(f"Error (Unsupported Argument): {err}")
        sys.exit(1)
    
