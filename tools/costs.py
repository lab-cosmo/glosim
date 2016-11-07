"""List of cost functions """
import numpy as np 

def mae(x):
    return np.mean(np.absolute(x))
def mse(x):
    return np.mean(np.power(x,2))
def rmse(x):
    return np.sqrt(mse(x))
def sup_e(x):
    return np.amax(np.absolute(x))