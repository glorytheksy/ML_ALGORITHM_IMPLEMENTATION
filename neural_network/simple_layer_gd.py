'''
Created on 2018年6月25日

@author: yxmfi
'''
import numpy as np
import pandas as pd
import activation_function

history_weight = []
data = pd.read_csv("C:/Users/yxmfi/Desktop/temp_file" + u"/jupyter博客/testSet.csv",encoding='utf-8',error_bad_lines=False)
features = data.drop(['label'], axis = 1)
targets = data['label']

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        h = np.dot(weights, x)
        output = activation_function.sigmoid(h)
        error = y - output
        error_term = error * activation_function.sigmoid_prime(h)
        del_w += error_term * x
    weights += learnrate * del_w / n_records