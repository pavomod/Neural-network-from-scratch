from .initializer import Initializer
from .activationFunctions import ActivationFunction

class Layer:
    def __init__(self, num_neurons, activation_function, initialization, seed):
        self.num_neurons = num_neurons
        self.activation_function = ActivationFunction(activation_function)
        self.initialization = Initializer(initialization, seed).function
