'''
Created on 2018年5月30日
@author: pc
'''

import numpy as np
import pandas as pd
from sklearn import datasets
import math

def sigmoid(X):    
    return 1.0 / (1 + np.exp(-X))

def biGradAscForLR(features, labels, cycleNum):
    '''Batch Gradient Ascent'''
    featureMat = np.mat(features)
    labelMat = np.mat(labels).transpose()
    
    m,n = np.shape(featureMat)
    
    weight = np.ones((n,1))
    alpha = 0.001
    
    for i in range(cycleNum):        
        error = labelMat - sigmoid(featureMat * weight)
        grad = featureMat.transpose() * error
        weight = weight + alpha * grad    
    return weight

def biStocasticGradAscForLR(features, labels):
    '''Stochastic Gradient Ascent'''
    featureMat = np.mat(features)
    labelMat = np.mat(labels).transpose()
    
    m,n = np.shape(featureMat)
    
    weight = np.ones((n,1))
    alpha = 0.001
        
    for i in range(m):
        x = featureMat[i]
        y = labelMat[i]        
        error = y[0,0] - sigmoid(np.dot(x,weight))        
        
        grad = x.transpose() * error
        weight = weight + alpha * grad    
    return weight
