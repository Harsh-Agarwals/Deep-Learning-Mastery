import numpy as np

from weights_init import randomInitialization
from forward_propagation import forward_prop
from back_propagation import back_propagation
from parameters_update import parameters_update

def batch_GD(X, y, layers, layer_neurons, layer_activations, loss, learning_rate, num_iterations):
    costs = []
    parameters = randomInitialization(neuron_layers=layer_neurons)
    
    for i in range(0, num_iterations):
        if i%500 == 0:
            print(f"------------{i}-------------")
        activations, cost = forward_prop(X, y, layers, layer_neurons, parameters, layer_activations, loss)
        costs.append(cost)
        gradients = back_propagation(y, layers, layer_neurons, parameters, activations, layer_activations, loss)
        parameters = parameters_update(layers, parameters, gradients, learning_rate)
    return parameters, costs

def mini_batch_GD():
    pass

def SGC():
    pass