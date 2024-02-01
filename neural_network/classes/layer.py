# Importing necessary modules
from .initializer import Initializer
from .activationFunctions import ActivationFunction

class Layer:
    def __init__(self, num_neurons, activation_function, initialization, seed):
        self.num_neurons = num_neurons  # Number of neurons in the layer
        # Instantiates an ActivationFunction object with the specified activation function
        self.activation_function = ActivationFunction(activation_function)
        # Instantiates an Initializer object with the specified initialization method
        self.initialization = Initializer(initialization, seed).function

