import numpy as np
import matplotlib.pyplot as plt

##Generating random data for training
X = np.random.rand(1500,2)
X = 2*X-1

##Initializing weights to random value
weights = np.random.rand(10,10,2)
weights = 2*weights -1

#Initialising parameters
alpha0 = 0.1 ##Learning parameter
lamda = 1e2
sigma0 = 10
rows = weights.shape[0]
columns = weights.shape[1]
n = weights.shape[2]
it = X.shape[0]

#learning parameter decreses with time
def alpha(p):
    return alpha0*np.exp(-p/lamda) 

#Calculating neighbouring penalty
def Neig_penalty(dist_to_bmu,p):
    curr_sigma = sigma(p)
    return np.exp(-(dist_to_bmu**2)/(2*curr_sigma**2))

#Sigma decreases with time
def sigma(p):
    return sigma0*np.exp(-p/lamda) 

#Training Algorithm
for p in range(it):
    
    #Breaking loop when no more significant spread occurs
    if sigma(p) < 0.1:
        break

    if p%5==0:
        a = weights[:,:,0].flatten()
        b = weights[:,:,1].flatten()
        plt.scatter(a,b)
        plt.show()

    #Taking random input data 
    index =  np.random.choice(range(len(X)))
    input_data = X[index]
    
    #Calculating Best Matching Unit (BMU)
    list_bmu = []
    for i in range(rows):
        for j in range(columns):
            dist = np.linalg.norm((input_data-weights[i,j]))
            list_bmu.append(((i,j),dist))
    list_bmu.sort(key=lambda x: x[1])
    min_bmu = list_bmu[0][0]
    min_bmu_x = list_bmu[0][0][0]
    min_bmu_y = list_bmu[0][0][1]
    
    #Updating the weights according to decay rate and distance
    for i in range(rows):
        for j in range(columns):
            dist_to_bmu = np.linalg.norm((np.array(min_bmu)-np.array((i,j))))
            weights[i][j]+=Neig_penalty(dist_to_bmu,p)*alpha(p)*(input_data-weights[i][j])
#or           weights[i][j]+=N(dist_to_bmu,p)*L(p)*(input_data-weights[i][j])
            

print('Number of iterations: ')
print(p)
print('\n')

#Testing Algorithm
testing_input = [[0.1, 0.8], [0.5, -0.2], [-0.8, -0.9], [-0.06, 0.9]]
print('Testing Inputs : ')
print(testing_input)
print('\n')
for z in range(4):
    input_data = testing_input[z]
    
    #Calculating Best Matching Unit (BMU)
    list_bmu = []
    for i in range(rows):
        for j in range(columns):
            dist = np.linalg.norm((input_data-weights[i,j]))
            list_bmu.append(((i,j),dist))
    list_bmu.sort(key=lambda x: x[1])
    min_bmu = list_bmu[0][0]
    print('For input '+str(z)+' Minimum Euclidian distance is : '+str(list_bmu[0][1]))
    print('The cluster it belongs to is :'+str(min_bmu)+'\n')
    
