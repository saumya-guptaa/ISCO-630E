import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

# Input Datasets
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
expected_output = np.array([[1], [0], [0], [1]])

epochs = 10000
learning_rate = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

# Random Weights and bias initialization
hidden_weights = np.random.uniform(size = (inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size = (1, hiddenLayerNeurons))
output_weights = np.random.uniform(size = (hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size = (1, outputLayerNeurons))

print("\nInitial hidden weights: ")
print(hidden_weights)
print("\nInitial hidden biases: ")
print(hidden_bias)
print("\nInitial output weights: ")
print(*output_weights)
print("\nInitial output biases: ")
print(output_bias)

e=[]
e_square = []


# Training the Algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Updating Weights and Bias
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis = 0, keepdims = True) * learning_rate
    hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis = 0, keepdims = True) * learning_rate

    er = ((error[0][0]) + (error[1][0]) + (error[2][0]) + (error[3][0])) /4; 
    e_square.append(np.square(er))
    e.append(abs(er))

    
print("\nFinal hidden weights: ")
print(hidden_weights)
print("\nFinal hidden bias: ")
print(hidden_bias)
print("\nFinal output weights: ")
print(output_weights)
print("\nFinal output bias: ")
print(output_bias)

print("\n Expected Outputs: ")
print(expected_output)
print("\nOutput from neural network after 10,000 epochs: ")
print(predicted_output) 

# Plotting the learning curve
ep_l = []
for i in range(epochs):
    ep_l.append(i)
    
plt.plot(ep_l, e)    
plt.title("Absolute Error vs Epoch")
plt.show()
plt.plot(ep_l, e_square)    
plt.title("Square sum Error vs Epoch")
plt.show()   

def predict(x):
    hidden_out=sigmoid(np.dot(x,hidden_weights)+hidden_bias)
    return sigmoid(np.dot(hidden_out,output_weights)+output_bias)
     
    
#decision boundary
resolution=0.02
X=inputs
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(expected_output))])
# plot the decision contour
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))

zz=np.array([xx1.ravel(), xx2.ravel()]).T
Z=[]
for i in zz:
    Z.append((predict(i)[0][0]))
Z=np.array(Z)    
Z = Z.reshape(xx1.shape)

fig,axes = plt.subplots(1,1,figsize=(5,5)) 
fig.suptitle('Red Dots are Class 1 and Blue Dots are Class 0')
axes.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
axes.set_xlabel('Input x1')
axes.set_ylabel('Input x2')
axes.set_title('Decision Contour(Heat Map)')
axes.scatter(0,0,color="r")
axes.scatter(1,0,color="b")
axes.scatter(0,1,color="b")
axes.scatter(1,1,color="r")