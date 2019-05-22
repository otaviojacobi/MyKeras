import numpy as np

VALID_ACTIVATION_FUNCTIONS = ['linear', 'sigmoid']

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def linear(x, derivative=False):
    return 1 if derivative else x