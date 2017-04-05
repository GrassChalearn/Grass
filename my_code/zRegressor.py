from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import gaussian_process
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn import linear_model
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import IsolationForest
#from sklearn.ensemble import RandomForestRegressor 
#from sklearn.multioutput import MultiOutputRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.neighbors import RadiusNeighborsRegressor
import numpy as np
import eval
#import matplotlib.pyplot as plt
#import seaborn as sns



import pickle


class Regressor(BaseEstimator):
    
    #rng = check_random_state(0)
    
    def __init__(self):
        pass

    def fit(self, X, y):
        self.clf = BaggingRegressor(base_estimator=None, n_estimators=200, max_samples=1.0, max_features=1.0, 
                                    bootstrap=True, bootstrap_features=False, oob_score=True)
        #self.clf = DecisionTreeRegressor()
        #self.clf = AdaBoostRegressor()
        #self.clf = ExtraTreesRegressor()
        #self.clf = GradientBoostingRegressor()
        #self.clf = linear_model.BayesianRidge()
        #self.clf = gaussian_process.GaussianProcessRegressor()
        #self.clf =  IsolationForest()
        #self.clf =  RandomForestRegressor
        #self.clf =  sklearn.linear_model.PassiveAggressiveRegressor
        #self.clf = MultiOutputRegressor()
        #self.clf = KNeighborsRegressor(n_neighbors=2)
        #self.clf = RadiusNeighborsRegressor()
        #self.clf = linear_model.Lasso(alpha = 0.1)
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
    
    def predictColumn (self, column, X_train, data, y_train ,i):
        y_train = data['target'].values 
        clf = Regressor()
        h = [row[i] for row in X_train]
        h = np.reshape(h , (95236, 1)) 
        clf.fit(h,y_train)
        y_trainPred = clf.predict(h)
        print eval.rmsle(y_trainPred, y_train)
        
    def predictColumns (self, tab, X_train, data, y_train):
        y_train = data['target'].values 
        clf = Regressor()
        #h = np.reshape(h , (95236, 1)) 
        clf.fit(tab,y_train)
        y_trainPred = clf.predict(tab)
        #sns.regplot(x = y_trainPred, y = y_train, data = data, color = 'Orange')
        return eval.rmsle(y_trainPred, y_train)
        
    def predictDataColumn (self, X_train,data, y_train):
        clf = Regressor()
        print clf.predictColumn ('Moment_of_the_day', X_train,data, y_train,0)
        print clf.predictColumn ('station_id', X_train,data, y_train,1)
        print clf.predictColumn ('average_duration_(s)', X_train,data, y_train, 2)
        print clf.predictColumn ('season', X_train,data, y_train, 3)
        print clf.predictColumn ('temperature_(F)', X_train,data, y_train, 4)
        print clf.predictColumn ('humidity', X_train,data, y_train, 5)
        print clf.predictColumn ('visibility', X_train,data, y_train, 6)
        print clf.predictColumn ('wind', X_train,data, y_train, 7)
        print clf.predictColumn ( 'cloud cover', X_train,data, y_train, 8)
        print clf.predictColumn ( 'precipitation_(mm)', X_train,data, y_train, 9)
        
        
    def CombinaisonDataColumn (self, X_train,data, y_train):
        clf = Regressor()
        y0 = np.reshape([row[0] for row in X_train] , (95236, 1))
        y1 = np.reshape([row[1] for row in X_train] , (95236, 1))
        y2 = np.reshape([row[2] for row in X_train] , (95236, 1))
        y3 = np.reshape([row[3] for row in X_train] , (95236, 1))
        y4 = np.reshape([row[4] for row in X_train] , (95236, 1))
        y5 = np.reshape([row[5] for row in X_train] , (95236, 1))
        y6 = np.reshape([row[6] for row in X_train] , (95236, 1))
        y7 = np.reshape([row[7] for row in X_train] , (95236, 1))
        y8 = np.reshape([row[8] for row in X_train] , (95236, 1))
        y9 = np.reshape([row[9] for row in X_train] , (95236, 1))
        y10 = np.reshape([row[10] for row in X_train] , (95236, 1))
        Y = [y0,y1,y2,y3, y4, y5, y6, y7, y8, y9,y10]
        return Y
        
    def functXtrain(self,X_train):
        y0 = np.reshape([row[0] for row in X_train] , (95236, 1))
        y1 = np.reshape([row[1] for row in X_train] , (95236, 1))
        y2 = np.reshape([row[2] for row in X_train] , (95236, 1))
        #y3 = np.reshape([row[3] for row in X_train] , (95236, 1))
        #y4 = np.reshape([row[4] for row in X_train] , (95236, 1))
        y5 = np.reshape([row[5] for row in X_train] , (95236, 1))
        y6 = np.reshape([row[6] for row in X_train] , (95236, 1))
        #y7 = np.reshape([row[7] for row in X_train] , (95236, 1))
        y8 = np.reshape([row[8] for row in X_train] , (95236, 1))
        #y9 = np.reshape([row[9] for row in X_train] , (95236, 1))
        y10 = np.reshape([row[10] for row in X_train] , (95236, 1))
        a = []
        for i  in range(0,95236) :
                b = []
                b.append(y0[i][0])
                b.append(y1[i][0])
                b.append(y2[i][0])
                #b.append(y3[i][0])
                #b.append(y4[i][0])
                b.append(y5[i][0])
                b.append(y6[i][0])
                #b.append(y7[i][0])
                b.append(y8[i][0])
                #b.append(y9[i][0])
                b.append(y10[i][0])
                a.append(b)
        return a
    
    def functX(self,X):
        y0 = np.reshape([row[2] for row in X] , (95236, 1))
        y1 = np.reshape([row[3] for row in X] , (95236, 1))
        y2 = np.reshape([row[10] for row in X] , (95236, 1))
        a = []
        for i  in range(0,95236) :
                b = []
                b.append(y0[i][0])
                b.append(y1[i][0])
                b.append(y2[i][0])
                a.append(b)
        return a
    
    def functXvalid(self,X):
        y0 = np.reshape([row[0] for row in X] , (5098, 1))
        y1 = np.reshape([row[1] for row in X] , (5098, 1))
        y2 = np.reshape([row[2] for row in X] , (5098, 1))
        y5 = np.reshape([row[5] for row in X] , (5098, 1))
        y6 = np.reshape([row[6] for row in X] , (5098, 1))
        y8 = np.reshape([row[8] for row in X] , (5098, 1))
        y10 = np.reshape([row[2] for row in X] , (5098, 1))
        a = []
        for i  in range(0,5098) :
                b = []
                b.append(y0[i][0])
                b.append(y1[i][0])
                b.append(y2[i][0])
                b.append(y5[i][0])
                b.append(y6[i][0])
                b.append(y8[i][0])
                b.append(y10[i][0])
                a.append(b)
        return a
    
    def functXtest(self,X):
        y0 = np.reshape([row[0] for row in X] , (10350, 1))
        y1 = np.reshape([row[1] for row in X] , (10350, 1))
        y2 = np.reshape([row[2] for row in X] , (10350, 1))
        y5 = np.reshape([row[5] for row in X] , (10350, 1))
        y6 = np.reshape([row[6] for row in X] , (10350, 1))
        y8 = np.reshape([row[8] for row in X] , (10350, 1))
        y10 = np.reshape([row[10] for row in X] , (10350, 1))
        a = []
        for i  in range(0,10350) :
                b = []
                b.append(y0[i][0])
                b.append(y1[i][0])
                b.append(y2[i][0])
                b.append(y5[i][0])
                b.append(y6[i][0])
                b.append(y8[i][0])
                b.append(y10[i][0])
                a.append(b)
        return a
    
    
def generateCombinaison (k, n) :
        if k <= 0:
            print 'on ne prend pas en compte ces cas : k<=0'
        elif k == 1 :
            Y = []
            for i in range(0,n+1):
                Y.append([i])
            return Y
        else :
            Y = []
            for i in range(0,n+1):
                l = generateCombinaison(k-1,n)
                for x in l:
                    if i not in x:
                        x.append(i)
                        x.sort()
                        if x not in Y:
                            Y.append(x)
            return Y
 
           