# Neural network
import numpy as np
from nnlib import *
import scipy.io
import Predictor
import ProcessImages

# Data Preprocessing
train_percent = 0.7

file = scipy.io.loadmat('ex4data1.mat')
raw_X = file['X'].T
raw_Y = file['y']
# Convert 10 in data to 0
raw_Y[(raw_Y == 10)] = 0

# Randomize Dataset
combined = np.concatenate((raw_X,raw_Y.T)).T
np.random.shuffle(combined)
combined = combined.T

new_X = combined[:-1,:]
new_Y = combined[-1,:]

# Separate out Y for Multi-component Classification
Y_process = np.zeros((new_Y.shape[0],10))

for i in range(10):
    idx = np.where(new_Y==i)[0]
    Y_process[idx,[i]] = 1

train_len = int(new_X.shape[1]*train_percent)
X_train = new_X[:,:train_len]
Y_train = Y_process[:train_len,:].T

X_test = new_X[:,train_len:]
Y_test = new_Y[train_len:].T

# Initial variables for Neural Network
hid_layers = [25]
num_iter = 10000
learn_rate = 1.2

# Initialize Neural Network
#parameters = nn_model(X_train, Y_train, hid_layers, num_iter, learn_rate, True)

"""
Arguments:
X -- dataset of shape (2, number of examples)
Y -- labels of shape (1, number of examples)
n_h -- size of the hidden layer
num_iterations -- Number of iterations in gradient descent loop
print_cost -- if True, print the cost every 1000 iterations

Returns:
parameters -- parameters learnt by the model. They can then be used to predict.
"""
X = X_train
Y = Y_train
n_h = hid_layers
num_iterations = num_iter
learning_rate = learn_rate
print_cost = True

n_x = X.shape[0]
n_y = Y.shape[0]

# Initialize parameters
parameters = initialize_parameters(n_x, n_h, n_y)    

# Loop (gradient descent)
for i in range(0, num_iterations):
     
    # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
    A_fin, cache = forward_propagation(X, parameters)

    # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
    cost = compute_cost(A_fin, Y, parameters)
 
    # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
    grads = backward_propagation(parameters, cache, X, Y)
 
    # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
    parameters = update_parameters(parameters, grads, learning_rate)
    
    ### END CODE HERE ###
    
    # Print the cost every 1000 iterations
    if print_cost and i % 1000 == 0:
        print ("Cost after iteration %i: %f" %(i, cost))


# Using trained NN, find predicted y data
predictions = predict(parameters, X_test)

accuracy = np.sum((Y_test == predictions))/Y_test.shape[0]

print('\n Accuracy is : ')
print(accuracy)
#print ('Accuracy: %d' %np.sum((Y_test == predictions))/Y_test.shape[0] + '%')

