"""
Machine learning for simple image classification using Python and scikit-learn
Written by David Easton, 4/29/2020
"""
import sys, getopt

def main(argv):
    dataset = ''
    estimator = ''
    try:
        opts, args = getopt.getopt(argv, "hd:e:", ["dataset=","estimator="])
    except getopt.GetoptError:
      print 'main.py -d <supported image dataset> -e <supported sklearn estimator>'
      sys.exit(2)