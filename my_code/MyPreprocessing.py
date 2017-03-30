# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:58:43 2017

@author: swann.raynal
"""


from sys import argv
from sklearn.base import BaseEstimator
from data_manager import DataManager
from sklearn.decomposition import PCA


class Preprocessor(BaseEstimator):
    
    def __init__(self):
        self.transformer = PCA(n_components=3)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        return self.transformer.transform(X)
        
        
    def suppr(self, X):
        moment=X[:,0] 
        station=X[:,1]
        date=X[:,2]
        temperature=X[:,5]
        vent=X[:,8]
        humidite=X[:,6] 
        nuage=X[:,9]
        precipitation=X[:,10]
        pca=[humidite, nuage, precipitation] #combinaison des variables très corrélées
        pro = Preprocesor()
        Y=[moment, station, date, temperature, vent, pro.fit_transform(pca)] #Nouveau tableau ne contenant que les variables utiles
        return Y
        
   
