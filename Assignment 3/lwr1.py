

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Housing.csv')
#data=data.sample(frac=1)
data=data.reset_index(drop=True)
y=data.price
#y=y[0:50]
x=data.iloc[:,[2,3,4,5,11]].values
#x=x[0:50,:]
x1=np.array(x)
y_1 = np.array(y)
#for i in range(len(x)):
#    for j in range(11):
#        if x[i][j] == 'yes':
#            x[i][j]=1
#        if x[i][j] =='no':
#            x[i][j]=0
xx= (x1-np.min(x1,axis=0))/(np.max(x1,axis=0)-np.min(x1,axis=0))
x = x.astype(float)

x[:,0]=xx[:,0]
y = (y_1-np.min(y_1,axis=0))/(np.max(y_1,axis=0)-np.min(y_1,axis=0))            
x = x.astype(float)
pmin=np.min(x1,axis=0)
pmax=np.max(x1,axis=0)
m = x.shape[0]
n = x.shape[1]+1
t1 = np.ones((m,1))
x = np.hstack((t1,x))
theta = np.random.rand(1,n)
y = np.array(y)
y = np.reshape(y,(y.shape[0],1))

tau=100
alpha=0.1

def Createtheta():
    Matrix=np.random.randn(n,1)
    return Matrix

def calculateWeight(t):
    weights=[]
    for n in x:
        temp=0
        for i in range(len(t)):
            sqr=(t[i]-n[i])*(t[i]-n[i])
            temp = temp + np.exp(-sqr/2*(tau**2))
        weights.append(temp/len(t))
    return np.reshape(np.array(weights),newshape=(-1,1))
    #return weights

#weights=calculateWeight(x[0])
def fwd(x,w):
    return np.dot(x,w)

def difference(y,yy):
    out=(yy- y)
    return out

def meanSquaredError(y,yy):
    return np.mean(np.multiply(y-yy , y-yy))

def denorm(y):
    return (y*(pmax-pmin))+pmin

def GD(dw,x,weights):
    return (dw*x*weights).mean()

errorArray=[]
errors=[]
lossArray=[]
def train(index):
    temp=x[index]
    theta=Createtheta()
    weights=calculateWeight(temp)
    weights[index]=0
    for epoch in range(8000):
        ypred=fwd(x,theta)
        dw=difference(y,ypred)
        errorArray.append(epoch)
        P=denorm(ypred)
        O=denorm(y)
        diff=abs(abs(P)-abs(O))
        diff=diff/abs(O)
        diff=diff*100
        errors.append(diff)
        lossArray.append(meanSquaredError(y,ypred))
        for i in range(len(theta)):
            grad=GD(dw,x[:,i:i+1],weights)
            theta[i]=theta[i] - alpha*grad
   
    return fwd(temp.reshape(-1,6),theta),y[index]

#test on index
index = 45
predictedValue,originalValue=train(index)
print("Predicted Value: ",(predictedValue))
print("Original Value: ",(originalValue))
diff=abs((originalValue)-(predictedValue))
diff=(diff/(originalValue))/100
print("error is: ",diff)
#plot loss curve
#plt.plot(errorArray,lossArray)
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.show()



index = 75
predictedValue,originalValue=train(index)
print("Predicted Value: ",(predictedValue))
print("Original Value: ",(originalValue))
diff=abs((originalValue)-(predictedValue))
diff=(diff/(originalValue))/100
print("error is: ",diff)



index = 115
predictedValue,originalValue=train(index)
print("Predicted Value: ",(predictedValue))
print("Original Value: ",(originalValue))
diff=abs((originalValue)-(predictedValue))
diff=(diff/(originalValue))/100
print("error is: ",diff)


index = 215
predictedValue,originalValue=train(index)
print("Predicted Value: ",(predictedValue))
print("Original Value: ",(originalValue))
diff=abs((originalValue)-(predictedValue))
diff=(diff/(originalValue))/100
print("error is: ",diff)

index = 315
predictedValue,originalValue=train(index)
print("Predicted Value: ",(predictedValue))
print("Original Value: ",(originalValue))
diff=abs((originalValue)-(predictedValue))
diff=(diff/(originalValue))/100
print("error is: ",diff)


index = 415
predictedValue,originalValue=train(index)
print("Predicted Value: ",(predictedValue))
print("Original Value: ",(originalValue))
diff=abs((originalValue)-(predictedValue))
diff=(diff/(originalValue))/100
print("error is: ",diff)






















#index = 75
#predictedValue,originalValue=train(index)
#print("Predicted Value: ",denorm(predictedValue))
#print("Original Value: ",denorm(originalValue))
#diff=abs(denorm(originalValue)-denorm(predictedValue))
#diff=(diff/denorm(originalValue))*100
#print("error is: ",diff)
#
#index = 125
#predictedValue,originalValue=train(index)
#print("Predicted Value: ",denorm(predictedValue))
#print("Original Value: ",denorm(originalValue))
#diff=abs(denorm(originalValue)-denorm(predictedValue))
#diff=(diff/denorm(originalValue))*100
#print("error is: ",diff)
#
#index = 325
#predictedValue,originalValue=train(index)
#print("Predicted Value: ",denorm(predictedValue))
#print("Original Value: ",denorm(originalValue))
#diff=abs(denorm(originalValue)-denorm(predictedValue))
#diff=(diff/denorm(originalValue))*100
#print("error is: ",diff)
