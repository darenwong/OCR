import numpy as np
from nnlib import *
import scipy.io
from Predictor import *
from ProcessImages import *

#set filepath
filepath = 'test.png'

#Set Parameters
parameters = np.load('para_dict1.npy').item()

#Predict digit written in Image file
predictor(filepath, parameters)