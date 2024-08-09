import numpy as np

def parameters_update(layers, parameters, gradients, learning_rate):
    for i in range(1, layers+1):
        parameters['w'+str(i)] -= learning_rate*gradients['dw_'+str(i)]
        parameters['b'+str(i)] -= learning_rate*gradients['db_'+str(i)]
    return parameters