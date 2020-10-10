#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SVM (hard margin and soft margin)
import random
import numpy as np

class data:
    def gaussian(mean, variance, shape):
        return np.random.normal(loc=mean, scale=variance, size=shape)

def generate_data(dim, size):
    return np.array([data.gaussian(1,5,dim) for i in range(size)])


class SVM:
    def _init_(self, input, output):
        #initilizing weight
        w = numpy.zeros(len(input[0]))
        #initialize learning rate
        l_rate = 1
        #epoch
        epoch = 100000
        #output list
        out = []
        #training svm
        for e in range(epoch):
            for i, val in enumerate(input):
                val1 = numpy.dot(input[i], w)
                if (output[i] * val1 < 1):
                    w = w + l_rate * ((output[i]*input[i]) - (2(1/epoch) * w))
                else:
                    w = w + l_rate * (-2 * (1/epoch) * w)

        for i, val in enumerate(input):
            out.append(numpy.dot(input[i], w))

        self.w = w
        self.out = out


    def test(data):
        result = []
        for i, value in enumerate(data):
            result.append(numpy.dot(data[i], w))
        return result
    
# generate dataset
dim = 5 #dimension of data points
size = 50 #number of data points in the dataset
dataset = generate_data(dim, size)
# create randomized labels for the data
output = [random.choice([-1, 1]) for i in range(size)]
print("Dataset")
print(dataset)
print("Labels")
print(output)

# Divide the dataset and output into train and test
permutation = np.random.permutation(size)
ratio = int(0.8 * size)
data_train = [dataset[x] for x in permutation[:ratio]]
out_train = [output[x] for x in permutation[:ratio]]
data_test = [dataset[x] for x in permutation[ratio:]]
out_test = [output[x] for x in permutation[ratio:]]

# Pipe into SVM, generate it, plot it
svm = SVM(data_train, out_train)
print("**************************")
print("Weights: ", str(svm.w))


# In[ ]:




