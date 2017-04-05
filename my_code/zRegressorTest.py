# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:29:52 2017

@author: rami1


This module could perform hyper-parameter selection by cross-validation
See: http://scikit-learn.org/stable/modules/cross_validation.html.
"""
import sys
sys.path.append("../sample_code/")
import seaborn as sns
from sklearn import preprocessing
from data_manager import DataManager
from zRegressor import Regressor
from MyPreprocessing import Preprocessor
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
from zRegressor import generateCombinaison
import matplotlib.pyplot as plt
import seaborn as sns 
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
    X_train = data.drop('target', axis=1).values
    #print X_train
    #X_train = X_train.drop('date_(m/d/y)', axis=1).values
    
    #X_train = X_train.drop('visibility' , axis=1).values
    #print X_train.shape
    #np.delete(X_train, 2, 0)                      
   # X_train = X_train.drop('visibility')
    # The target values encoded as categorical variables
    #std_scale = preprocessing.StandardScaler().fit(X_train)
    #X_train_std = std_scale.transform(X_train)
    
    y_train = data['target'].values 
    #a = data['humidity'].values 
    clf = Regressor()
    #clf.fit(X_train_std, y_train)
    #y_trainPred = clf.predict(X_train_std)
    #print clf.predictColumn('humidity', X_train, data, y_train ,5)
    #print ('humidity : {}'.format( clf.predictColumn('humidity', data, y_train)))
    #print eval.rmsle(y_trainPred, y_train)
    #print eval.rmsle(y_train, y_train)
    #print eval.rmsle(a, y_train)
    #sns.regplot(x = y_train, y = y_trainPred, color = 'Orange')
    # clf.predictDataColumn (data, y_train)

    #print X_train[0]
    # eval.rmsle(y_trainPred, y_train)
    
    #print clf.predictColumn ('Moment_of_the_day', X_train, data, y_train ,0)
    #print clf.predictDataColumn (X_train,data, y_train)
    
    pre = Preprocessor()
    #X_train1 = pre.suppr(X_train)
    #X_train1 = np.reshape(X_train1, (95236, 5))
    #clf.fit(X_train1, y_train)
    #y_trainPred = clf.predict(X_train1)
    # eval.rmsle(y_trainPred, y_train)
    
    #print generateCombinaison (1, 9)
    #print generateCombinaison (2, 9)
    #print generateCombinaison (3, 9)
    
    #tabRes = []
    #Y = clf.CombinaisonDataColumn(X_train,data, y_train)
    #gC = generateCombinaison (1, 9)
    #for el1 in gC :
        #print el1[0]
     #   i = el1[0]
      #  a = Y[i]
        #print a
       # b = clf.predictColumns(a, X_train, data, y_train)
        #tabRes.append(b)
    
    #print tabRes
    
    #tabRes = []
    #Y = clf.CombinaisonDataColumn(X_train,data, y_train)
    #gC = generateCombinaison (2, 9)
    #for el1 in gC:
     #   a = []
      #  for i in range(0,2):
       #     a.append(Y[el1[i]])
        #b = clf.predictColumns(a, X_train, data, y_train)
        #tabRes.append(b)
        
    #print tabRes
    #print Y
    #Y0 = clf.CombinaisonDataColumn(X_train,data, y_train)
    #Y0= np.reshape(Y0 , (95236, 10))
    #a = []
    #for tab in Y0:
     #   b = tab[0]
     #   for x in b:
      #      a.append(x)
        
    #print a
    #print clf.predictColumns(Y, X_train, data, y_train)
    #print a
    #np.reshape(a,(1,10))
    #np.concatenate( a, axis=0 )
    #print a
    
    #tabRes = []
    #Y0 = clf.CombinaisonDataColumn(X_train,data, y_train)
    #gC = generateCombinaison (2, 9)
    #e1 = []
    #min = 0.5
    #Y0= np.reshape(Y0 , (95236, 10))
    #for d in gC :    
     #   a = []
      #  if d[0] != d[1] :
       #     tab1 = Y0[d[0]]
        #    tab2 = Y0[d[1]]
            #print tab2[1]
         #   for i  in range(0,95236) :
          #      b = []
           #     b.append(tab1[i][0])
            #    b.append(tab2[i][0])
             #   a.append(b)
                #print a
           # res = clf.predictColumns (a, X_train, data, y_train)
            #if res <= min :
             #   e1 = a
              #  min = res
               # tabRes.append(res)
    #print tabRes
    #print min
    #print e1[0]
    
    
    
    #res = clf.functXtrain(X_train)
    #resA = clf.predictColumns (X_train, X_train, data, y_train)
    #print resA
    #X_scaled = preprocessing.scale(res)
    #X_normalized = preprocessing.normalize(res, norm='l2')
    #res1 = []
    #for x in res :
        #a = (((x[0]+1)*1.655) + (x[1]*0.085) + (x[2]*0.01) + (x[3]*0.07) + (x[4]*0.09) + (x[5]*0.8) + (x[6]*10+ 7) )/7
        #res1.append(int(a))
        
    #print res1
    #print sns.regplot(x = y_train, y = y_train, data = data, color = 'Orange')
    #print y_train
    #print np.array(res1)
    #print sns.regplot(x = np.array(res1), y = y_train, data = data, color = 'Red')
    #print eval.rmsle(np.array(res1), y_train)
        
    #res2 = []
    #for x in res :
        #a = x[2]
        #res2.append(int(a))
        
    #print y_train
    #print np.array(res2)
    #print sns.regplot(x = np.array(res2), y = y_train, data = data, color = 'Green')
    #print eval.rmsle(np.array(res2), y_train)
    print clf.predictColumns (X_train, X_train, data, y_train)