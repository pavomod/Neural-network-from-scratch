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
    def __linear(self,x: np.ndarray) -> np.ndarray:
        return x
    
    def __d_linear(self,x: np.ndarray) -> np.ndarray:
        return 1
    
    # sigmoid --------------------
    def __sigmoid(self,x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-x))
    
    def __d_sigmoid(self,x: np.ndarray) -> np.ndarray:
        s = self.__sigmoid(x)
        return s*(1-s)
    
    # tanh --------------------
    def __tanh(self,x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def __d_tanh(self,x: np.ndarray) -> np.ndarray:
        return 1 - math.pow(np.tanh(x), 2)
    
    # relu --------------------
    def __relu(self,x: np.ndarray) -> np.ndarray:
        return np.maximum(0,x)
    
    def __d_relu(self,x: np.ndarray) -> np.ndarray:
        if x > 0:
            return 1
        return 0
    
    # leaky relu --------------------
    def __prelu(self,x: np.ndarray, alpha: float=0.01) -> np.ndarray:
        return np.maximum(x > 0, x, alpha*x)
    
    def __d_prelu(self,x: np.ndarray, alpha: float=0.01) -> np.ndarray:
        if x > 0:
            return 1
        return alpha