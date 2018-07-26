'''
Created on 2018年6月24日

@author: yxmfi
'''
import numpy as np

def sigmoid(X):    
    return 1.0 / (1 + np.exp(-X))

def sigmoid_prime(X):
    return sigmoid(X) * (1 - sigmoid(X))

