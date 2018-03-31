# Package imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1) # set a seed so that the results are consistent
  
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    if not isinstance(n_h, list):
        n_h = [n_h]
    
    n = [n_x] + n_h + [n_y]
    parameters = {}
    for i in range(len(n)-1):
    
      parameters["W" + str(i+1)] = np.random.randn(n[i+1],n[i])/100
      parameters["b" + str(i+1)] = np.zeros((n[i+1],1))
    return parameters

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A_fin -- The sigmoid output of the final activation
    cache -- a dictionary containing Z and A values for each layer
    """
    # Determine number of layers in NN
    num_layers= len(parameters)//2 + 1
    tempA = [X]
    cache = {}
    
    # For each layer excluding the final layer, forward propagate the data
    for i in range(num_layers - 2):
        W = parameters["W" + str(i+1)]
        b = parameters["b" + str(i+1)]
        Z = np.matmul(W,tempA[i]) + b
        A = np.tanh(Z)
        cache["Z" + str(i+1)] = Z
        tempA.append(A)
        cache["A" + str(i+1)] = A
        
    # Forward propagate to the final layer using sigmoid function
    W = parameters["W" + str(num_layers-1)]
    b = parameters["b" + str(num_layers-1)]
    Z = np.matmul(W,tempA[num_layers-2]) + b
    A_fin = sigmoid(Z)
    cache["Z" + str(num_layers-1)] = Z
    tempA.append(A_fin)
    cache["A" + str(num_layers-1)] = A_fin

    return A_fin, cache
  



def compute_cost(A_fin, Y, parameters):

    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:

    A_fin -- The sigmoid output of the second activation, of shape (1, number of examples)

    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example

 
    
    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    logprobs = np.multiply(np.log(A_fin),Y) + np.multiply(np.log(1 - A_fin),1 - Y)
    cost = -np.sum(logprobs)/m
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 

                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    grads={}
    lp= len(parameters)//2 + 1
    # Backward propagation: calculate dW1, db1, dW2, db2. 
   
    for i in reversed(range(int(lp))):
        if i == lp-1:
            A_previous = cache["A" + str(i-1)]
            A_final = cache["A" + str(i)]
            dZ_final = A_final - Y
            grads["dZ" + str(i)] = dZ_final 
            grads["dW" + str(i)] = np.matmul(dZ_final, A_previous.transpose())/m
            grads["db" + str(i)] = np.sum(dZ_final, axis=1, keepdims=True)/m
        elif i > 1:
            W_next = parameters["W" + str(i+1)]
            dZ_next = grads["dZ" + str(i+1)]
            A_next = cache["A" + str(i+1)]
            A = cache["A" + str(i)]
            A_previous = cache["A" + str(i-1)]
            dZ = np.multiply(np.matmul(W_next.transpose(), dZ_next), (1 - np.power(A, 2)))
            grads["dZ" + str(i)] = dZ
            grads["dW" + str(i)] = np.matmul(dZ, A_previous.transpose())/m
            grads["db" + str(i)] = np.sum(dZ, axis=1, keepdims=True)/m
        elif i == 1:
            W_next = parameters["W" + str(i+1)]
            dZ_next = grads["dZ" + str(i+1)]
            A_next = cache["A" + str(i+1)]
            A = cache["A" + str(i)]
            dZ = np.multiply(np.matmul(W_next.transpose(), dZ_next), (1 - np.power(A, 2)))
            grads["dZ" + str(i)] = dZ
            grads["dW" + str(i)] = np.matmul(dZ, X.transpose())/m
            grads["db" + str(i)] = np.sum(dZ, axis=1, keepdims=True)/m
        else:
            break
            
    
    return grads

  
def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Calculate number of layers in NN
    num_layers= len(parameters)//2 + 1
    
    # Update W and b for each layer using gradients calculated
    for i in range(num_layers - 1):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] \
        - learning_rate*grads["dW"+str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] \
        - learning_rate*grads["db"+str(i+1)]
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
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
        parameters = update_parameters(parameters, grads)
        
        ### END CODE HERE ###
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X, threshold = 0.5):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using the threshold given.
    A_fin, cache = forward_propagation(X, parameters)
    predictions = (A_fin > threshold)
    
    return predictions

