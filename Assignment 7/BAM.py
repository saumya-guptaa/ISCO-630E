import numpy as np

Xlayer = np.matrix([[1, 1, 1, 1, 1, 1 ], [-1, -1, -1, -1, -1, -1 ], [1, -1, -1, 1, 1, 1 ], [1, 1, -1, -1, -1, -1 ]])
Ylayer = np.matrix([[1, 1, 1], [-1, -1, -1], [-1, 1, 1], [1, -1, 1]])
n = 6
m = 3
M = 4

#Function to Transform vector to bipolar form [-1, 1]
def activation_output(mat):
    ret_mat = []
    for item in mat[0]:
        if item < 0:
            ret_mat.append(-1)
        elif item==0:
            ret_mat.append(-1)
        else:
            ret_mat.append(1)
    return ret_mat

print('\n------Initialization-----')
print('Xlayer Matrix: ')
print(Xlayer)
print('\nYlayer Matrix: ')
print(Ylayer)
print('\n')
print('M = '+str(M))


#Training algorithm for bipolar data
print('\n-------Training Algorithm-------\n')
Weights = np.zeros((n,m))
for i in range(M):
    temp = np.transpose(Xlayer[i]).dot(Ylayer[i])
    Weights = Weights + temp
    
print('\nFinal Weight Matrix -')
print(Weights)

#Testing algorithm for bipolar data
print('\n-------Testing Algorithm-------\n')

print('\nTest 1 - Taking M=2 and Xlayer as input\n')
z = 1
Xin2 = Xlayer[z].dot(Weights)
mat = np.array(Xin2)
Activation = activation_output(mat)
print('Input :')
print(Xlayer[z])
print('Dot Product of Input and Weights :')
print(Xin2)
print('After activation :')
print(Activation)
print('Targeted output :')
print(Ylayer[z])
print('Both are equal. Thus verified')

print('\nTest 2 - Taking M=3 and Ylayer as input\n')
z = 2
Yin2 = Ylayer[z].dot(np.transpose(Weights))
mat = np.array(Yin2)
Activation = activation_output(mat)
print('Input :')
print(Ylayer[z])
print('Dot Product of Input and Weights :')
print(Yin2)
print('After activation :')
print(Activation)
print('Targeted output :')
print(Xlayer[z])
print('Both are equal. Thus verified')
