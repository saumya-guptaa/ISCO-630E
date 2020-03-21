#Run Cell
#%%
#basic imports
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from sklearn.linear_model import LinearRegression
import random



# extract data
def extract_data() :
    X = []
    Y = []
    
    data_file = open("housing.csv")
    data_reader = csv.reader(data_file)
    
    row_count = 0
    for row in data_reader :
        row_count += 1
        if(row_count != 1) :
            Y.append(float(row[1])) # price
            data_row = [float(1)]
            for i in range(2, len(row)) :
                if(row[i] == "yes" or row[i] == "no") :
                    if(row[i] == "yes") :
                        data_row.append(float(1))
                    else :
                        data_row.append(float(0))
                else :
                    data_row.append(float(row[i]))
            X.append(data_row)
   
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y



# linear regression functions
# cost function linear regression normal
def cost_function_LR(X, Y, theta) :
    sample_count = float(X.shape[0])
    return (float(1) / (float(2) * sample_count)) * float(np.dot(np.transpose(np.dot(X, theta) - Y) , np.dot(X, theta) - Y))

# gradient descent linear regression normal
def gradient_descent_LR(X, Y, theta, alpha, threshold) :
    costs = [cost_function_LR(X, Y, theta)]
    iterations = [1]
    sample_count = float(X.shape[0])
    iteration_count = 2
    
    while(True):
        theta = theta - (alpha / sample_count) * np.dot(np.transpose(X), np.dot(X, theta) - Y)
        
        current_cost = cost_function_LR(X, Y, theta)
#         if(iteration_count % 5000 == 0) :
#             print(iteration_count, " => ", current_cost, " prev cost : ", costs[iteration_count - 2], " error diff :" , costs[iteration_count - 2] - current_cost)
        prev_cost = costs[iteration_count - 2]
        costs.append(current_cost)
        iterations.append(iteration_count)
        
        if(prev_cost - current_cost <= threshold) :
            break
            
        iteration_count = iteration_count + 1
    
    print("Total iterations: ", iteration_count)
    display_graph(costs, iterations)    # display graph 
    return theta

# initialize theta
def init_theta_LR(X) :
    return np.zeros(X.shape[1])
# display cost function v/s iterations
def display_graph(costs, iterations) :
#     print("Number of iterations: ", iterations[len(iterations) - 1])
#     print("Final cost : ", costs[len(costs) - 1])
    plt.plot(iterations, costs)
    
# normal equation 
def normal_equation(X, Y) :
    return np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), Y))



# initialize global variables
X, Y = extract_data()
alpha = 0.001
threshold = 0.001
theta = init_theta_LR(X)



# apply standardization
std_vals = np.std(X, axis=0)
mean_vals = np.mean(X, axis=0)

for i in range(X.shape[0]) :
    for j in range(X.shape[1]) :
        if(std_vals[j] != 0) :
            X[i][j] = X[i][j] - mean_vals[j]
            X[i][j] = X[i][j] / std_vals[j]
            
      
        
# get params for LR
LR_theta = gradient_descent_LR(X, Y, theta, alpha, threshold)
print("Final theta LR : ", LR_theta)
print("Final cost LR : ", cost_function_LR(X, Y, LR_theta))




# for normal equation LR
normal_equation_LR_theta = normal_equation(X, Y)
print("Final cost normal : ", cost_function_LR(X, Y, normal_equation_LR_theta))
print("Final theta values normal : ", normal_equation_LR_theta)



# using scikit learn library to verify answer
lr = LinearRegression()
lr.fit(X, Y)
# print(lr.score(X, Y))
print("Final coefficients LR: " , lr.coef_)
print("Final intercept LR: " , lr.intercept_)



# apply regularization , init global variables 
reg_lambda = 0.1



# linear regression regularized functions
# cost function linear regression regularized
def cost_function_LR_reg(X, Y, theta, reg_lambda) :
    sample_count = float(X.shape[0])
    reg_sum = (reg_lambda / (float(2) * sample_count)) * np.sum(np.square(theta[1:]))
    return (float(1) / (float(2) * sample_count)) * float(np.dot(np.transpose(np.dot(X, theta) - Y) , np.dot(X, theta) - Y)) + reg_sum

# gradient descent linear regression regularized
def gradient_descent_LR_reg(X, Y, theta, alpha, threshold, reg_lambda) :
    costs = [cost_function_LR_reg(X, Y, theta, reg_lambda)]
    iterations = [1]
    sample_count = float(X.shape[0])
    iteration_count = 2
    
    while(True):
        reg_term = (reg_lambda) * theta
        reg_term[0] = 0
        theta = theta - (alpha / sample_count) * (np.dot(np.transpose(X), np.dot(X, theta) - Y) + reg_term)
        
        current_cost = cost_function_LR_reg(X, Y, theta, reg_lambda)
        prev_cost = costs[iteration_count - 2]
        costs.append(current_cost)
        iterations.append(iteration_count)
        
        if(prev_cost - current_cost <= threshold) :
            break
            
        iteration_count = iteration_count + 1
    
    print("Total iterations: ", iteration_count)
    display_graph(costs, iterations)    # display graph 
    return theta

# initialize theta
def init_theta_LR_reg(X) :
    return np.zeros(X.shape[1])
    
# normal equation 
def normal_equation_reg(X, Y, reg_lambda) :
    reg_term = np.zeros((X.shape[1], X.shape[1]))
    for i in range(1, X.shape[1]) :
        reg_term[i][i] = reg_lambda
    
    return np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + reg_term), np.dot(np.transpose(X), Y))



# Regularized linear regression using gradient descent
theta = init_theta_LR_reg(X)
theta_LR_reg = gradient_descent_LR_reg(X, Y, theta, alpha, threshold, reg_lambda)

print("Final cost LR reg : ", cost_function_LR_reg(X, Y, theta_LR_reg, reg_lambda))
print("Final theta values LR reg : ", theta_LR_reg)



# regularized normal equation
theta = init_theta_LR_reg(X)
normal_equation_LR_reg_theta = normal_equation_reg(X, Y, reg_lambda)

print("Final cost LR reg : ", cost_function_LR_reg(X, Y, normal_equation_LR_reg_theta, reg_lambda))
print("Final theta values LR reg : ", normal_equation_LR_reg_theta)



# split into train and test
train_size = int((X.shape[0] * 7) / 10)
X_train_indices = random.sample(range(0, X.shape[0]), train_size)

X_train = []
X_test = []
Y_train = []
Y_test = []

for i in range(X.shape[0]) :
    if i in X_train_indices :
        X_train.append(X[i])
        Y_train.append(Y[i])
    else :
        X_test.append(X[i])
        Y_test.append(Y[i])

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)



# Regularized linear regression using gradient descent
theta = init_theta_LR_reg(X_train)
theta_LR_reg = gradient_descent_LR_reg(X_train, Y_train, theta, alpha, threshold, reg_lambda)

print("Final train cost LR reg : ", cost_function_LR_reg(X_train, Y_train, theta_LR_reg, reg_lambda))
print("Final train theta values LR reg : ", theta_LR_reg)



#regularised normal equation
theta = init_theta_LR_reg(X_train)
normal_equation_LR_reg_theta = normal_equation_reg(X_train, Y_train, reg_lambda)

print("Final train cost LR reg : ", cost_function_LR_reg(X_train, Y_train, normal_equation_LR_reg_theta, reg_lambda))
print("Final train theta values LR reg : ", normal_equation_LR_reg_theta)


# finding error for test data
print("Error obtained for test data : ", cost_function_LR_reg(X_test, Y_test, normal_equation_LR_reg_theta, reg_lambda))