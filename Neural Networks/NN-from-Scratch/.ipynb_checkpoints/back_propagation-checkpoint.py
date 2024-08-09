import numpy as np
from derivatives import sigmoid_derivative, relu_derivative, leaky_relu_derivative, tanh_derivative

def back_propagation(y, layers, layer_neurons, parameters, activations, layer_activations, loss):
    gradients = {}
    for i in range(layers, 0, -1):
        if i==layers:
            if loss=="bce":
                gradients['da_'+str(i)] = -np.sum((y/activations['a_'+str(i)])-(1-y)/(1-activations['a_'+str(i)]), axis=1).reshape(1, -1)
                gradients['dz_'+str(i)] = np.multiply(gradients['da_'+str(i)], sigmoid_derivative(activations['z_'+str(i)]))
            elif loss=="categorical_ce":
                gradients['da_'+str(i)] = -np.sum(y/activations['a_'+str(i)])
                gradients['dz_'+str(i)] = np.multiply(gradients['da_'+str(i)], softmax_derivative(activations['z_'+str(i)]))
            elif loss=="mse":
                gradients['da_'+str(i)] = np.mean(np.sum(y-activations['a_'+str(i)]))
                gradients['dz_'+str(i)] = gradients['da_'+str(i)]
            elif loss=="mae":
                gradients['da_'+str(i)] = np.where(y>activations['a_'+str(i)], -1, 1)
                gradients['dz_'+str(i)] = gradients['da_'+str(i)]
            else:
                raise TypeError("Wrong loss function, please choose among('mse', 'mae', 'bce', 'categorical_ce')")
        else:
            activation = layer_activations[i-1]
            if activation=="sigmoid":
                gradients['da_'+str(i)] = np.dot(parameters['w'+str(i+1)].T, gradients['dz_'+str(i+1)])
                gradients['dz_'+str(i)] = np.multiply(gradients['da_'+str(i)], sigmoid_derivative(activations['z_'+str(i)]))
            elif activation=="relu":
                gradients['da_'+str(i)] = np.dot(parameters['w'+str(i+1)].T, gradients['dz_'+str(i+1)])
                gradients['dz_'+str(i)] = np.multiply(gradients['da_'+str(i)], relu_derivative(activations['z_'+str(i)]))
            elif activation=="leaky_relu":
                gradients['da_'+str(i)] = np.dot(parameters['w'+str(i+1)].T, gradients['dz_'+str(i+1)])
                gradients['dz_'+str(i)] = np.multiply(gradients['da_'+str(i)], leaky_relu_derivative(activations['z_'+str(i)]))
            elif activation=="tanh":
                gradients['da_'+str(i)] = np.dot(parameters['w'+str(i+1)].T, gradients['dz_'+str(i+1)])
                gradients['dz_'+str(i)] = np.multiply(gradients['da_'+str(i)], tanh_derivative(activations['z_'+str(i)]))
            else:
                raise TypeError("Wrong middle activation function, please choose among('relu', 'leaky_relu', 'tanh', 'sigmoid')")
        gradients['dw_'+str(i)] = (1/y.shape[0])*np.dot(gradients['dz_'+str(i)], activations['a_'+str(i-1)].T)
        gradients['db_'+str(i)] = (1/y.shape[0])*np.sum(gradients['dz_'+str(i)], axis=1, keepdims=True)
    return gradients