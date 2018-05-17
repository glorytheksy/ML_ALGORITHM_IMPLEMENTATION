'''
Created on 2018年5月16日
@author: glorythesky
'''
import numpy as np

def gini(data,label_col):
    labels = data[label_col]
    total_size = np.size(labels)
    if (0 == total_size):
        return 1    
    label_num_dict = {}
    for label in labels:
        if label not in label_num_dict:
            label_num_dict[label] = 1
        else:
            label_num_dict[label] = label_num_dict[label] + 1                
    return 1 - np.sum([ (float(value) / float(total_size))**2 for key,value in label_num_dict.items() ])
    
def gini_divide(data, feature, branch_condition, label_col):
    labels = data[label_col]
    total_size = np.size(labels)
    if (0 == total_size):
        return 1     
    data_1 = data[branch_condition == data[feature]]
    data_2 = data[branch_condition != data[feature]]    

    return ( float(data_1.shape[0]) / float(total_size) ) * gini(data_1, label_col) + ( float(data_2.shape[0]) / float(total_size) ) * gini(data_2, label_col)