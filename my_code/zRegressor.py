from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import gaussian_process
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model

import pickle


class Regressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        #self.clf = DecisionTreeRegressor()
        #self.clf = BaggingRegressor()
        #self.clf = AdaBoostRegressor()
        #self.clf = ExtraTreesRegressor()
        #self.clf = GradientBoostingRegressor()
        self.clf = linear_model.BayesianRidge()
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
