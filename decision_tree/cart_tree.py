import numpy as np
import pandas as pd
from IPython.display import display 
import impurity

class cart_tree(object):
                
    def __init__(self):
        self.__lt = None
        self.__rt = None
        self.__depth = None
        self.__feature = None
        self.__branch_condition = None
        self.__label = None
        self.__size = None
        
    def fit(self, data, label_col, stop_depth=None , stop_impurity=None, stop_num=None):
        self.fit_with_depth(data, 1, label_col, stop_depth, stop_impurity, stop_num)
    
    def fit_with_depth(self, data, depth, label_col, stop_depth=None , stop_impurity=None, stop_num=None):  
        labels = data[label_col]
                
        # stop growing when impurity is enough
        if (impurity.gini(labels) < stop_impurity):
            self._stop_proc(depth, data, label_col)
            return
        # stop growing when depth is enouph
        if (stop_depth is not None and self.__depth + 1 >= depth):
            self._stop_proc(depth, data, label_col)
            return
        # stop when sample size is less than given value 
        if (np.size(labels) <= stop_num):
            self._stop_proc(depth, data, label_col)
            return
        
        self.__depth = depth
        self.__label = self._get_node_label(data, label_col)
        self.__feature, self.__branch_condition  = self._get_branch(data, label_col)
        
        # stop growing if one of the son nodes is empty
        if (0 == np.size(data[[self.__feature] == self.__branch_condition]) or 0 == np.size(data[[self.__feature] != self.__branch_condition])):
            self.__lt = None
            self.__rt = None
            self.__size = 1
            return
                    
        self.__lt = cart_tree()
        self.__rt = cart_tree()        
        self.__lt.fit(data[[self.__feature] == self.__branch_condition], self.__depth+1, label_col, stop_depth , stop_impurity, stop_num)
        self.__rt.fit(data[[self.__feature] != self.__branch_condition], self.__depth+1, label_col, stop_depth , stop_impurity, stop_num)           
        self.__size = self.__lt.size() + self.__rt.size()
     
    def predict(self, X):             
        return [self.line_predict(line) for line in X]
            
    def line_predict(self, x):
        if (None == self.__lt or None == self.__rt):
            return self.__label
            
        if x[self.__feature] == self.__branch_condition:
            return self.__lt.line_predict(x)
        else:
            return self.__rt.line_predict(x)
                            
    def _get_branch(self, data, label_col):    
        features = data.drop([label_col, 'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'], axis = 1) 

        self_feature = '1'
        self_branch_condition = '1'
        mini_gini = 1

        for feature in features.keys():
            feature_value_dict = {}
            for feature_value in data[feature]:
                print feature_value_dict

                if feature_value in feature_value_dict:
                    continue
                else:
                    feature_value_dict[feature_value] = 1
                    temp_gini = impurity.gini_divide(data, feature, feature_value, label_col)
                    if temp_gini < mini_gini:
                        self_feature = feature
                        self_branch_condition = feature_value
                        mini_gini = temp_gini
                    else:
                        pass
        return self_feature, self_branch_condition
                    
    def _get_node_label(self, data, label_col):
        labels = data[label_col]
        label_num_dict = {}
        for label in labels:
            if label not in label_num_dict:
                label_num_dict[label] = 1
            else:
                label_num_dict[label] = label_num_dict[label] + 1
        max_label_value_num = 0
        max_label_value = ''
        for label_value,label_value_num in label_num_dict.items():
            if (label_value_num > max_label_value_num):
                max_label_value = label_value
                max_label_value_num = label_value_num            
        return max_label_value
    
    
    def _stop_proc(self, depth, data, label_col):
        self.__lt = None
        self.__rt = None
        self.__depth = depth
        self.__feature = None
        self.__label = self._get_node_label(data, label_col)
        self.__size = 1
        return 
    
    
    @property
    def size(self):
        return self._size
        
    @property
    def lt(self):
        return self._lt
    
    @property
    def rt(self):
        return self._rt


    
    