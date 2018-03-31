# Package imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1) # set a seed so that the results are consistent



def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer
    ### END CODE HERE ###
    return (n_x, n_h, n_y)
  
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
    
    n = [n_x] + n_h + [n_y]
    parameters = {}
    for i in range(len(n)-1):
    
      parameters["W" + str(i+1)] = np.random.randn(n[i+1],n[i])/100
      parameters["b" + str(i+1)] = np.zeros((n[i+1],1))
    return parameters
  
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
    cost = - np.sum(logprobs)  
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)/m     # makes sure cost is the dimension we expect. 
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
