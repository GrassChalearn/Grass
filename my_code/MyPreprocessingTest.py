# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 23:42:18 2017

@author: rami1
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:13:27 2017

@author: swann.raynal
"""
import sys
sys.path.append("../sample_code/")
from data_manager import DataManager
from MyPreprocessing import Preprocessor
input_dir = "../public_data"
output_dir = "../res"

basename = 'bikes'
D = DataManager(basename, input_dir) # Load data
print("*** Original data ***")
print D

Prepro = Preprocessor()
 
# Preprocess on the data and load it back into D
D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
# Here show something that proves that the preprocessing worked fine
print("*** Transformed data ***")
print D

# Preprocessing gives you opportunities of visualization:
# Scatter-plots of the 2 first principal components
# Scatter plots of pairs of features that are most relevant
import matplotlib.pyplot as plt
X = D.data['X_train']
Y = D.data['Y_train']
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()