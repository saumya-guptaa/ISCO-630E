import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
input_size=2

epochs = 10000
learning_rate = 0.1

def predict(x,W):
    z = W.T.dot(x)
    if z>=0:
        return 1
    else:
        return 0

def train(d,er):
    W = np.random.uniform(size = input_size+1) #add on for bias
    print("Initial Values: ")
    print("Threshold = "+str(-W[0]) + " , W1 = " + str(W[1]) + " , W2 = " + str(W[2]))
    for i in range(epochs):
        error = 0 
        for j in range(d.shape[0]):
            x = np.insert(X[j], 0, 1)
            y = predict(x,W)
            e = d[j] - y
            error+=e
            W = W + learning_rate * e * x
        er.append(np.square(error/4))
    print("\n After training : ")       
    print("Threshold = "+str(-W[0]) + " , W1 = " + str(W[1]) + " , W2 = " + str(W[2]))
    return W
    
d_and = np.array([0, 0, 0, 1])
d_or = np.array([0, 1, 1, 1])
d_nand = np.array([1, 1, 1, 0])
d_nor = np.array([1, 0, 0, 0])

e_and = []
e_or = []
e_nand = []
e_nor = []

print('\n------And Gate-----')
W_and = train(d_and,e_and)

print('\n------Or Gate-----')
W_or = train(d_or,e_or)

print('\n------Nand Gate-----')
W_nand = train(d_nand,e_nand)

print('\n------Nor Gate-----')
W_nor = train(d_nor,e_nor)

fig,ax = plt.subplots(4,2,figsize=(12,15))
fig.suptitle('Decision Boundary of Gates, Yellow is 1 and Purple is 0')
xmin, xmax = -0.1, 1.1
x = np.arange(xmin, xmax, 0.1)

# AND
ax[0][0].scatter(X[:,0],X[:,1],c=d_and)
ax[0][0].set_title("AND gate decision boundary")
ax[0][0].set_xlim([xmin, xmax])
ax[0][0].set_ylim([-0.1, 1.1])
W=W_and
m = -W[1] / W[2]
c = -W[0] / W[2]
ax[0][0].plot(x, m*x + c)

#OR
ax[0][1].scatter(X[:,0],X[:,1],c=d_or)
ax[0][1].set_title("OR gate decision boundary")
ax[0][1].set_xlim([xmin, xmax])
ax[0][1].set_ylim([-0.1, 1.1])
W=W_or
m = -W[1] / W[2]
c = -W[0] / W[2]
ax[0][1].plot(x, m*x + c)

# NAND
ax[1][0].scatter(X[:,0],X[:,1],c=d_nand)
ax[1][0].set_title("NAND gate decision boundary")
ax[1][0].set_xlim([xmin, xmax])
ax[1][0].set_ylim([-0.1, 1.1])
W=W_nand
m = -W[1] / W[2]
c = -W[0] / W[2]
ax[1][0].plot(x, m*x + c)

#NOR
ax[1][1].scatter(X[:,0],X[:,1],c=d_nor)
ax[1][1].set_title("NOR gate decision boundary")
ax[1][1].set_xlim([xmin, xmax])
ax[1][1].set_ylim([-0.1, 1.1])
W=W_nor
m = -W[1] / W[2]
c = -W[0] / W[2]
ax[1][1].plot(x, m*x + c)

# AND
ax[2][0].set_title("Sqaure Sum Error v/s epoch AND Gate")
ax[2][0].set_xlim([-1, 30])
ax[2][0].plot(e_and)

# OR
ax[2][1].set_title("Sqaure Sum Error v/s epoch OR Gate")
ax[2][1].set_xlim([-1, 30])
ax[2][1].plot(e_or)

# NAND
ax[3][0].set_title("Sqaure Sum Error v/s epoch NAND Gate")
ax[3][0].set_xlim([-1, 30])
ax[3][0].plot(e_nand)

# NOR
ax[3][1].set_title("Sqaure Sum Error v/s epoch NOR Gate")
ax[3][1].set_xlim([-1, 30])
ax[3][1].plot(e_nor)