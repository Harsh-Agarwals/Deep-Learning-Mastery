import numpy as np
from activation import sigmoid, tanh, softmax, relu, leaky_relu

def sigmoid_derivative(z):
    return sigmoid(z)*(1 - sigmoid(z))

def relu_derivative(z):
    return np.where(z>0, 1, 0)

def leaky_relu_derivative(z):
    return np.where(z>0, 1, 0.01)

def tanh_derivative(z):
    return 1 - np.power(tanh(z), 2)

def softmax_derivative(z):
    s = softmax(z)
    return np.diag(s) - np.outer(s, s)