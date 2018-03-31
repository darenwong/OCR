# Neural network
import numpy as np
from nnlib import *
import scipy.io

# Data Preprocessing
train_percent = 0.7

file = scipy.io.loadmat('ex4data1.mat')
raw_X = file['X'].T
raw_Y = file['y']

train_len = int(raw_X.shape[1]*train_percent)
train_X = raw_X[:,:train_len]
train_Y = raw_Y[:train_len]

test_X = raw_X[:,train_len:]
test_Y = raw_Y[train_len:]

#X_train = np.array([[1,2,3,4],
#              [2,3,4,5],
#              [3,4,5,6],
#              [4,5,6,7],
#              [5,6,7,8]])
#Y_train = np.array([1, 1, 0, 0, 0]).reshape((5,1))


# Initial variables for Neural Network
hid_layers = [10, 5, 3]
num_iter = 10000
learn_rate = 1.2

# Initialize Neural Network
parameters = nn_model(X_train, Y_train, hid_layers, num_iter)

# Using trained NN, find predicted y data
predictions = predict(parameters, X, 0.5)

print ('Accuracy: %d' % float((np.dot(Y_train,predictions.T) + \
                               np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')

