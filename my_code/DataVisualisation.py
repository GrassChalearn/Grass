import os
#os.chdir("C:/Users/Desktop/Rayane/competition-6-1-data-competition-6-1-data-input")
os.chdir("../public_data/")
f=open("bikes_train.data","r")
import numpy as np
M= np.loadtxt('bikes_train.data')
print(M)
f.close()
f=open("bikes_train.solution","r")
S= np.loadtxt('bikes_train.solution')
print(S)
f.close()
#Representation des données
import matplotlib.pyplot as plt
plt.plot(M[0:95235,10],S[0:95235],'o')
plt.ylabel('Le nombre de vélos')
plt.xlabel('Les précipitations')
plt.title('L\'évolution du nombre de vélos en fonction des précipitations')
plt.grid()
plt.show()