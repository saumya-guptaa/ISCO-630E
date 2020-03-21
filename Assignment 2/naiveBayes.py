# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread



# reading the channels of the images
img_i = imread('iband.gif')
img_b = imread('bband.gif')
img_g = imread('gband.gif')
img_r = imread('rband.gif')



# verifying the shapes
print("img_i : " + str(len(img_i)) + "," + str(len(img_i[0])))
print("img_b : " + str(len(img_b)) + "," + str(len(img_b[0])))
print("img_g : " + str(len(img_g)) + "," + str(len(img_g[0])))
print("img_r : " + str(len(img_r)) + "," + str(len(img_r[0])))



# Calculating no. of points lying on the river
img_y = plt.imread('op1.jpeg')
img_y = (img_y > (0.5*255)) * 1
river_points_population = np.sum(img_y==1)
print("No. of points on the river is: " + str(river_points_population))



# Creating the river training class
img_y = img_y.flatten()
#Covering the river portion
cover_river = np.floor(np.random.rand(50) * river_points_population)    #Selecting in random distribution
# print(cover_river)
class_riv = []
j = 0
for i in range(512*512):
    if(img_y[i]==1):
        j+=1
        if j in cover_river:
            class_riv.append(i)
print(class_riv)



# Creating the non-river training class
non_river_points_population = np.sum(img_y==0)
print("No.  of points on non river portion is: " + str(non_river_points_population))
cover_non_river = np.floor(np.random.rand(100) * non_river_points_population)    #Selecting in rnon_river_points_poandom distribution
# print(cover_non_river)
class_non_riv = []
j=0
for i in range(512*512):
    if(img_y[i] == 0):
        j+=1
        if j in cover_non_river:
            class_non_riv.append(i)
print(class_non_riv)         



# Finding the means of the channels rgb values
test_data = np.dstack((img_r, img_g, img_b, img_i))
test_cpy = test_data
print(test_data.shape)
plt.imshow(test_data)
test_data = test_data.reshape(-1, 4)
river_mean = np.mean(test_data[class_riv], axis=0)
non_river_mean = np.mean(test_data[class_non_riv], axis=0)
print(river_mean, non_river_mean)



# Finding the covariance matrices
test_data = test_data.reshape(-1,4)
covariance_riv = np.cov(test_data[class_riv].T)
covariance_non_riv = np.cov(test_data[class_non_riv].T)
print(covariance_riv)
print(covariance_non_riv)
print(covariance_riv.shape, covariance_non_riv.shape)



# Generating the river predicted class
riv_1 = np.zeros(512*512)
for i in range(512*512):
    riv_1[i] = (test_data[i] - river_mean).T@np.linalg.inv(covariance_riv)@(test_data[i] - river_mean)
len(riv_1)



# Generating the non-river predicted class
non_riv_1 = np.zeros(512*512)
for i in range(512*512):
    non_riv_1[i] = (test_data[i] - non_river_mean).T@np.linalg.inv(covariance_non_riv)@(test_data[i] - non_river_mean)
len(non_riv_1)    



# Probability multivariate definition of river class
p1 = ((1/np.sqrt(2*3.14159))**(50))*(1/np.sqrt(np.linalg.det(covariance_riv)))*np.exp((-1/2)*riv_1)
print(p1)
p1.shape



# Probability multivariate definition of non-river class
p2 = ((1/np.sqrt(2*3.14159))**100)*(1/np.sqrt(np.linalg.det(covariance_non_riv)))*np.exp((-1/2)*non_riv_1)
print(p2)
p2.shape



# GENERATING THE RESULTS
# P1 = 0.3, P2 = 0.7
op = []
P1 = 0.3
P2 = 0.7
op = (P1*p1>=P2*p2)*255
op = op.reshape(512,512)
plt.imshow(op)



# P1 = P2 = 0.5
op = []
P1 = 0.5
P2 = 0.5
op = (P1*p1>=P2*p2)*255
op = op.reshape(512,512)
plt.imshow(op)



# P1 = 0.7, P2 = 0.3
op = []
P1 = 0.7
P2 = 0.3
op = (P1*p1>=P2*p2)*255
op = op.reshape(512,512)
plt.imshow(op)
