import numpy as np
import math
from typing import Callable

class ActivationFunction:
    def __init__(self, name: str):
        self.name = name
        self.function = self.__getFunction(name)
        self.derivative = self.__getDerivative(name)
    

    def __getFunction(self, name: str) -> Callable[[float], float]:
        if name == 'linear':
            return self.__linear
        elif name == 'sigmoid':
            return self.__sigmoid
        elif name == 'tanh':
            return self.__tanh
        elif name == 'relu':
            return self.__relu
        elif name == 'prelu':
            return self.__prelu
        else:
            raise ValueError('Invalid activation function')
        
    def __getDerivative(self, name: str) -> Callable[[float], float]:
        if name == 'linear':
            return self.__d_linear
        elif name == 'sigmoid':
            return self.__d_sigmoid
        elif name == 'tanh':
            return self.__d_tanh
        elif name == 'relu':
            return self.__d_relu
        elif name == 'prelu':
            return self.__d_prelu
        else:
            raise ValueError('Invalid activation function')


    # linear --------------------
    def __linear(x: float) -> float:
        return x
    
    def __d_linear(x: float) -> float:
        return 1
    
    # sigmoid --------------------
    def __sigmoid(x: float) -> float:
        return 1/(1+np.exp(-x))
    
    def __d_sigmoid(x: float) -> float:
        s = sigmoid(x)
        return s*(1-s)
    
    # tanh --------------------
    def __tanh(x: float) -> float:
        return np.tanh(x)
    
    def __d_tanh(x: float) -> float:
        return 1 - math.pow(np.tanh(x), 2)
    
    # relu --------------------
    def __relu(x: float) -> float:
        return np.maximum(0,x)
    
    def __d_relu(x: float) -> float:
        if x > 0:
            return 1
        return 0
    
    # leaky relu --------------------
    def __prelu(x: float, alpha: float=0.01) -> float:
        return np.maximum(x > 0, x, alpha*x)
    
    def __d_prelu(x: float, alpha: float=0.01) -> float:
        if x > 0:
            return 1
        return alpha