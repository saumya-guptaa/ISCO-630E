# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# importing the dataset
data = pd.read_csv('microchip_data.csv')
data=data.sample(frac=1)
data=data.reset_index(drop=True)
Y=data.iloc[:,2].values
X=data.iloc[:,:2].values

# Separate into training and testing data
train_size = int((X.shape[0] * 7) / 10) # 70% train data
X_train_indices = random.sample(range(0, X.shape[0]), train_size)
#print(X_train_indices)
X_train, X_test, Y_train, Y_test = [], [], [], []
for i in range(X.shape[0]) :
    if(i in X_train_indices) :
        X_train.append(X[i])
        Y_train.append(Y[i])
    else :
        X_test.append(X[i])
        Y_test.append(Y[i])

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

#calculation of phi
phi=sum(Y_train)/len(Y_train)
print("phi: "+str(phi))

#calculation of mu0 and mu1
mu0=[0,0]
mu1=[0,0]
c0=0
c1=0
for i in range(len(Y_train)):
    if Y_train[i]==0:
        c0+=1
        mu0+=X_train[i]
    else:
        c1+=1
        mu1+=X_train[i]

mu0/=c0
mu1/=c1
print("mu0: "+str(mu0))
print("mu1: "+str(mu1))

#calculation of sigma
m = len(Y_train)
sigma = np.zeros((2,2))
for i in range(m):
    xi=X_train[i]
    yi=Y_train[i]
    if yi==1:
        tmp=xi-mu1
        tmp=tmp.reshape(-1,1)
        tmp1=np.transpose(tmp)
        sqr=(tmp)*(tmp1)
        sigma = sigma + sqr
sigma=sigma/m
print("sigma: "+str(sigma))

def calculate_px_y(x,mu):
    n = len(mu)
    det=np.linalg.det(sigma)
    pi = 3.14
    inv=np.linalg.inv(sigma)
    den=pow(2*pi,n/2)*np.sqrt(det)
    tmp=x-mu
    tmp=tmp.reshape(-1,1)
    tmp1=np.transpose(tmp)
    p=np.dot(np.dot(tmp1,inv),tmp)
    p=(-0.5)*p
    num=np.exp(p)
    return num/den

#calculation of P(Y)
def calculate_py(y):
    if y==1:
        return phi
    else:
        return (1-phi)

#Predictor function
def predictor(x):
    p_0 = calculate_px_y(x,mu0)*calculate_py(0)
    p_1 = calculate_px_y(x,mu1)*calculate_py(1) 
    if p_0>p_1:
        return 0
    else:
        return 1
        
correct=0;
predictions=[]
l=len(Y_test)
for i in range(l):
    x=X_test[i]
    y=Y_test[i]
    y_predicted=predictor(x)
    predictions.append(y_predicted)
    if predictions[i]==y:
        correct+=1
print("Correct Predictions: " + str(correct) + " ,Out of: "+ str(l))
print("Accuracy: ",100*correct/l)
    