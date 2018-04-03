import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Initial variables for Neural Network
hid_layers = [25, 25]
num_iter = 10000
learn_rate = 1.1

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
num_pixels = X_train.shape[1] * X_train.shape[2]
X = X_train.reshape(X_train.shape[0], num_pixels).astype('float32').T/255
#X = X[:, :100]
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32').T/255
Y = np_utils.to_categorical(y_train, 10).T
#Y = Y[: , :100]
Y_test = np_utils.to_categorical(y_test, 10).T

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
A_fin, cache = forward_propagation(X_test, parameters)
cost = compute_cost(A_fin, Y_test, parameters)
accuracy = np.sum((y_test == predictions))/y_test.shape[0]

print('\n Accuracy is : {}'.format(accuracy))
print('\n Cost of test set is : {}'.format(cost))
#print ('Accuracy: %d' %np.sum((Y_test == predictions))/Y_test.shape[0] + '%')