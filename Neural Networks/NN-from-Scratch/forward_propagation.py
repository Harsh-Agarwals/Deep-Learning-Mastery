import numpy as np
from activation import sigmoid, softmax, relu, tanh, leaky_relu
from loss import mean_squared_error, mean_absolute_error, binary_cross_entropy_error, categorical_cross_entropy_loss

def forward_prop(X, y, layers, layer_neurals, parameters, activation, loss):
    activations = {}
    activations['a_0'] = X.T
    
    for i in range(1, layers+1):
        activations['z_'+str(i)] = np.dot(parameters['w'+str(i)], activations['a_'+str(i-1)]) + parameters['b'+str(i)]
  
        if i==layers:
            if loss == "mse":
                activations['a_'+str(i)] = activations['z_'+str(i)]
                cost = mean_squared_error(y, activations['a_'+str(i)])
            elif loss == "mae":
                activations['a_'+str(i)] = activations['z_'+str(i)]
                cost = mean_absolute_error(y, activations['a_'+str(i)])
            elif loss == "bce":
                activations['a_'+str(i)] = sigmoid(activations['z_'+str(i)])
                cost = binary_cross_entropy_error(y, activations['a_'+str(i)])
            elif loss == "categorical_ce":
                activations['a_'+str(i)] = softmax(activations['z_'+str(i)])
                cost = categorical_cross_entropy_loss(y, activations['a_'+str(i)])
            else:
                activations['a_'+str(i)] = sigmoid(activations['z_'+str(i)])
                cost = binary_cross_entropy_error(y, activations['a_'+str(i)])
        else:
            if activation=="softmax":
                activations['a_'+str(i)] = softmax(activations['z_'+str(i)])
            elif activation=="relu":
                activations['a_'+str(i)] = relu(activations['z_'+str(i)])
            elif activation=="leaky_relu":
                activations['a_'+str(i)] = leaky_relu(activations['z_'+str(i)])
            elif activation=="tanh":
                activations['a_'+str(i)] = tanh(activations['z_'+str(i)])
            else:
                activations['a_'+str(i)] = sigmoid(activations['z_'+str(i)])
    return activations, cost