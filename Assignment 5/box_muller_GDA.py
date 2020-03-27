# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from pylab import show, hist, subplot, figure

# importing the dataset
data = pd.read_csv('microchip_data.csv')
data=data.sample(frac=1)
data=data.reset_index(drop=True)
Y=data.iloc[:,2].values
# performing box_muller transformation in the standard normal distribution as values dont exceed the range of -1 to 1
# transformation function
pi = 3.14

def gaussian(u1, u2):
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * pi * u2)+2
    return z1, z2

# uniformly distributed values between 0 and 1
y=np.array(Y)
y.reshape(-1,1)
u1 = np.random.rand(1000)
u2 = np.random.rand(1000)

# run the transformation
z1, z2 = gaussian(u1, u2)

# plotting the values before and after the transformation
figure()
subplot(221) # the first row of graphs
hist(u1)     # contains the histograms of u1 and u2 
subplot(222)
hist(u2)
subplot(223) # the second contains
hist(z1)     # the histograms of z1 and z2
subplot(224)
hist(z2)
show()

# Concatenating z1 and z2 to create new matrix of features and concatenating y at the end to create the new box muller dataset
x1=[]
x2=[]
n=Y.shape[0]
for i in range(n):
    if Y[i]==1:
        x1.append(random.choice(z2))
        x2.append(random.choice(z2))
    else:
        x1.append(random.choice(z1))
        x2.append(random.choice(z1))
X=np.c_[x1,x2]
X_new = np.c_[z1, z2]
data_new = np.c_[X, y]    # New dataset, corresponding to the box-muller transform
#X = data.iloc[:, :-1].values  # Features Matrix
subplot(211)
hist(X_new)
plt.title('Guassian Distribution of two classes')
Y = Y.reshape(-1, 1)    # classes vector changed to matrix form

# plotting the two classes of data for visualization
subplot(212)
plt.title('Scattering of Data')
pos_data = data_new[data_new[:,-1]==1]
neg_data = data_new[data_new[:,-1]==0]
plt.scatter(pos_data[:,0], pos_data[:,1], color="red")
plt.scatter(neg_data[:,0], neg_data[:,1], color="blue")

# Separate into training and testing data
train_size = int((n * 7) / 10) # 70% train data
X_train_indices = random.sample(range(0, n), train_size)
#print(X_train_indices)
X_train, X_test, Y_train, Y_test = [], [], [], []
for i in range(n) :
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

#calculation of P(X/Y)
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
    