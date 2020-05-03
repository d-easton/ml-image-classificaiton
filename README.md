# ml-image-classificaiton
Image classification toolbox using machine learning.

## Getting started
### Packages
Use of this package requires the supporting packages included in the dependencies section of this README.
These packages can be independently with pip or, following the creation of a virtual environment, can be installed via the included requirements.txt file.
### Datasets
Image classification models need large quantities of data to train with. The size of these files means you need to download the necessary datasets independently.
Executing the configure script in the CLI will automatically take care of this for you.
'''
python 3.6.9 configure.py
'''
If you'd like to independently download the datasets, they can be found at the following sites:
- http://ufldl.stanford.edu/housenumbers/

The datasets/ directory should be populated with *.mat files once the datasets have successfully been downloaded.
## Dependencies
- Python (3.6.9)
- numpy
- Sci-kit learn
- matplotlib

## Datasets
- Street View House Numbers (SVHN), Stanford University: http://ufldl.stanford.edu/housenumbers/
- MNIST Database of handwritten numbers (MNIST): http://yann.lecun.com/exdb/mnist/

## Estimators
Estimators are included as part of the scikit-learn library. At this time, the following estimator models are supported
- 
