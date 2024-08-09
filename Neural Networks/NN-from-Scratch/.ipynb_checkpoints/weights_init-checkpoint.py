import numpy as np

def zerosInitialization(neuron_layers):
    parameters = {}
    L=len(neuron_layers)-1
    for i in range(1, L+1):
        parameters['w'+str(i)] = np.zeros((neuron_layers[i], neuron_layers[i-1]))
        parameters['b'+str(i)] = np.zeros((neuron_layers[i], 1))
    return parameters

def randomInitialization(neuron_layers):
    parameters = {}
    L=len(neuron_layers)-1
    for i in range(1, L+1):
        parameters['w'+str(i)] = np.random.randn(neuron_layers[i], neuron_layers[i-1])*0.03
        parameters['b'+str(i)] = np.zeros((neuron_layers[i], 1))
    return parameters

def HeInitialization(neuron_layers):
    parameters = {}
    L=len(neuron_layers)-1
    for i in range(1, L+1):
        parameters['w'+str(i)] = np.random.randn(neuron_layers[i], neuron_layers[i-1])*np.sqrt(2/neuron_layers[i-1])
        parameters['b'+str(i)] = np.zeros((neuron_layers[i], 1))
    return parameters

def GlorotInitialization(neuron_layers):
    parameters = {}
    L=len(neuron_layers)-1
    for i in range(1, L+1):
        parameters['w'+str(i)] = np.random.randn(neuron_layers[i], neuron_layers[i-1])*np.sqrt(2/(neuron_layers[i-1]+neuron_layers[i]))
        parameters['b'+str(i)] = np.zeros((neuron_layers[i], 1))
    return parameters

def specialInitialization(neuron_layers):
    parameters = {}
    L=len(neuron_layers)-1
    for i in range(1, L+1):
        parameters['w'+str(i)] = np.random.randn(neuron_layers[i], neuron_layers[i-1])*np.sqrt(1/(neuron_layers[i]+neuron_layers[i-1]))
        parameters['b'+str(i)] = np.zeros((neuron_layers[i], 1))
    return parameters