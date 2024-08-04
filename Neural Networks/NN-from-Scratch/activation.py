import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

def relu(z):
    return max(0, z)

def leaky_relu(z):
    return max(0.01*z, z)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))