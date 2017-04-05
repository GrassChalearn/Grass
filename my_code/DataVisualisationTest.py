# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 11:57:11 2017

@author: rami1
"""
import sys
sys.path.append("../my_code/")
import os
#os.chdir("C:/Users/Desktop/Rayane/competition-6-1-data-competition-6-1-data-input")
os.chdir("../public_data/")
f=open("bikes_train.data","r")
import numpy as np
M= np.loadtxt('bikes_train.data')
#print(M)
f.close()
#f=open("bikes_train.solution","r")
S= np.loadtxt('bikes_train.solution')
#print(S)
f.close()
from DataVisualisation import Visualisation

visu = Visualisation()
visu.display1(M,S)