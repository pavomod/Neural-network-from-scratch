import numpy as np
from typing import Callable, Tuple

class ActivationFunction:
    def __init__(self, name: str):
        self.name = name
        # Retrieves the appropriate function and its derivative based on the name provided
        self.function, self.derivative = self.__get(name)
    
    def __get(self, name: str) -> Tuple[Callable[[float], float], Callable[[float], float]]:
        # Associates the activation function names with their respective functions and derivatives
        if name == 'linear':
            return self.__linear, self.__d_linear
        elif name == 'sigmoid':
            return self.__sigmoid, self.__d_sigmoid
        elif name == 'tanh':
            return self.__tanh, self.__d_tanh
        elif name == 'relu':
            return self.__relu, self.__d_relu
        elif name == 'prelu':
            return self.__prelu, self.__d_prelu
        else:
            raise ValueError('Invalid activation function')


    def __linear(self, x: np.ndarray) -> np.ndarray:
        # Returns the input as is (no transformation)
        return x
    
    def __d_linear(self, x: np.ndarray) -> np.ndarray:
        # Derivative of linear function is a constant 1
        return np.ones(x.shape)
    

    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Sigmoid function, mapping input to a value between 0 and 1
        return 1 / (1 + np.exp(-x))
    
    def __d_sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Derivative of sigmoid function
        s = self.__sigmoid(x)
        return s * (1 - s)
    

    def __tanh(self, x: np.ndarray) -> np.ndarray:
        # Tanh function, mapping input to a value between -1 and 1
        return np.tanh(x)
    
    def __d_tanh(self, x: np.ndarray) -> np.ndarray:
        # Derivative of tanh function
        return 1 - np.power(self.__tanh(x), 2)
    

    def __relu(self, x: np.ndarray) -> np.ndarray:
        # ReLU function, returns max of 0 and input
        return np.maximum(0, x)
    
    def __d_relu(self, x: np.ndarray) -> np.ndarray:
        # Derivative of ReLU function
        return np.where(x > 0, 1, 0)
    

    def __prelu(self, x: np.ndarray, alpha: float=0.01) -> np.ndarray:
        # PReLU function, allows a small gradient when the unit is not active
        return np.where(x > 0, x, alpha * x)
    
    def __d_prelu(self, x: np.ndarray, alpha: float=0.01) -> np.ndarray:
        # Derivative of PReLU function
        return np.where(x > 0, 1, alpha)
