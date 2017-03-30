# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:29:52 2017

@author: rami1


This module could perform hyper-parameter selection by cross-validation
See: http://scikit-learn.org/stable/modules/cross_validation.html.
"""
from sklearn import preprocessing
from data_manager import DataManager
from zRegressor import Regressor
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
import numpy as np

datadir = '../public_data/'                        # Change this to the directory where you put the input data
dataname = 'bikes'
basename = datadir  + dataname
import data_converter as dc
import data_io
import matplotlib.pyplot as plt
import eval
reload(data_io)
 



if __name__ == "__main__":
    data = data_io.read_as_df(basename, 'train')
    X_train = data.drop('target', axis=1)
    X_train = X_train.drop('date_(m/d/y)', axis=1).values
    #X_train = X_train.drop('visibility' , axis=1).values
    #print X_train.shape
    #np.delete(X_train, 2, 0)                      
   # X_train = X_train.drop('visibility')
    # The target values encoded as categorical variables
    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train_std = std_scale.transform(X_train)
    
    y_train = data['target'].values  
    clf = Regressor()
    clf.fit(X_train_std, y_train)
    y_trainPred = clf.predict(X_train_std)
    print eval.rmsle(y_trainPred, y_train)
        
    sns.regplot(x = y_train, y = y_trainPred, color = 'Orange')


   
    
    
 
