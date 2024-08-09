import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z):
    return np.where(z>0, z, 0.01*z)

def softmax(z):
    e_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return e_z / np.sum(e_z)

# def softmax(z):
    # return np.exp(z)/np.sum(np.exp(z))