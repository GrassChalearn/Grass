# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:58:43 2017

@author: swann.raynal
"""


from sys import argv
from sklearn.base import BaseEstimator
from data_manager import DataManager # The class provided by binome 1
# Note: if zDataManager is not ready, use the mother class DataManager
from sklearn.decomposition import PCA


class Preprocessor(BaseEstimator):
      def suppr():
         data=data[:,0]
         return data
   
    def __init__(self):
        self.transformer = PCA(n_components=2)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        return self.transformer.transform(X)
    
if __name__=="__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data/"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'bikes'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print (D)
    
    Prepro = Preprocessor()
 
    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print (D)