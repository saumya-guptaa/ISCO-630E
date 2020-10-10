#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(42)


# In[2]:


# mahalanobis distance
# Distribution
distribution = {
    "mean": np.random.rand(1, 3),
    "covariance": [[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]],
}

# Point
point = np.random.rand(1, 3)


def mahalanobis_distance(distribution: "dict", point: "np.array()") -> int:
    """ Estimate Mahalanobis Distance 
    
    Args:
        distribution: a sample gaussian distribution
        point: a deterministic point
    
    Returns:
        Mahalanobis distance
    """
    mean = distribution["mean"]
    cov = distribution["covariance"]
    return np.sqrt((point - mean) @ np.linalg.inv(cov) @ (point - mean).T)[0][0]


# Our implementation
distance = mahalanobis_distance(distribution, point)
print(f"Ours : {distance}")

# scipy inbuilt
from scipy.spatial.distance import mahalanobis

distance = mahalanobis(
    point, distribution["mean"], np.linalg.inv(distribution["covariance"])
)
print(f"Scipy: {distance}")


# In[3]:


# Distribution 1
distribution1 = {
    "mean": np.array([[1,3,1]]),
    "covariance": np.array([[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]]),
}


# Distribution 2
distribution2 = {
    "mean": np.array([[1,3,1]]),
    "covariance": np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]),
}

# Distribution 3
d1 = np.random.rand(1,1000)
p1 = np.histogram(d1,100)[0]
p1 = p1 / np.sum(p1)

# Distribution 4
d2 = np.random.rand(1,1000)
p2 = np.histogram(d2,100)[0]
p2 = p1 / np.sum(p2)

def bhattacharyya_gaussian_distance(distribution1: "dict", distribution2: "dict",) -> int:
    """ Estimate Bhattacharyya Distance (between Gaussian Distributions)
    
    Args:
        distribution1: a sample gaussian distribution 1
        distribution2: a sample gaussian distribution 2
    
    Returns:
        Bhattacharyya distance
    """
    mean1 = distribution1["mean"]
    cov1 = distribution1["covariance"]

    mean2 = distribution2["mean"]
    cov2 = distribution2["covariance"]

    cov = (1 / 2) * (cov1 + cov2)

    T1 = (1 / 8) * (
        np.sqrt((mean1 - mean2) @ np.linalg.inv(cov) @ (mean1 - mean2).T)[0][0]
    )
    T2 = (1 / 2) * np.log(
        np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )

    return T1 + T2

def bhattacharyya_distance(distribution1: "dict", distribution2: "dict",) -> int:
    """ Estimate Bhattacharyya Distance (between General Distributions)
    
    Args:
        distribution1: a sample distribution 1
        distribution2: a sample distribution 2
    
    Returns:
        Bhattacharyya distance
    """
    sq = 0
    for i in range(len(distribution1)):
        sq  += np.sqrt(distribution1[i]*distribution2[i])
    
    return -np.log(sq)
    
    
# Our implementation (Gaussian)
distance = bhattacharyya_gaussian_distance(distribution1, distribution2)
print(f"Ours (Gaussian) : {distance}")

# Our implementation (General)
distance = bhattacharyya_distance(p1, p2)
print(f"Ours (General)  : {distance}")


# In[5]:


import numpy
import math

# m = int(input("Enter mean of gaussian distribution: "))
# v = int(input("Enter variance of gaussian distribution: "))
m=1
v=5
x = numpy.random.normal(m, math.sqrt(v), 1000)
p = 10
temp = 0
for i in range (1,p):
    temp=0
    for j in range(len(x)):
        temp+=pow(x[j],i)
    print (pow(temp,1/i))


# In[ ]:




