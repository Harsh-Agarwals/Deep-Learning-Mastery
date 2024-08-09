import numpy as np

def mean_squared_error(y, a):
    return -np.mean(np.sum(np.power(y-a, 2)))

def mean_absolute_error(y, a):
    return np.mean(np.sum(np.abs(y - a)))

def binary_cross_entropy_error(y, a, epsilon=1e-15):
    a = np.clip(a, epsilon, 1-epsilon)
    return -np.mean(np.sum(np.multiply(y,np.log(a)) + np.multiply((1-y), np.log(1-a))))

def categorical_cross_entropy_loss(y, a, epsilon=1e-15):
    a = np.clip(a, epsilon, 1-epsilon)
    return -np.mean(np.sum(y*np.log(a), axis=1))