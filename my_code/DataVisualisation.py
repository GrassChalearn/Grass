import os
#os.chdir("C:/Users/Desktop/Rayane/competition-6-1-data-competition-6-1-data-input")
#os.chdir("../public_data/")
#f=open("bikes_train.data","r")
import numpy as np
#M= np.loadtxt('bikes_train.data')
#print(M)
#f.close()
#f=open("bikes_train.solution","r")
#S= np.loadtxt('bikes_train.solution')
#print(S)
#f.close()
import matplotlib.pyplot as plt


class Visualisation() :
    
    def display1(self,M,S):
        plt.plot(M[0:95235,10],S[0:95235],'o')
        plt.ylabel('Le nombre de velos')
        plt.xlabel('Les precipitations')
        plt.title('L\'evolution du nombre de velos en fonction des precipitations')
        plt.grid()
        plt.show()