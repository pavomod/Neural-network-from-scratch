import numpy as np
from typing import Callable, Tuple

class ActivationFunction:
    def __init__(self, name: str):
        self.name = name
        self.function, self.derivative = self.__get(name)
    
    def __get(self, name: str) -> Tuple[Callable[[float], float], Callable[[float], float]]:
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


    # linear --------------------
    def __linear(self,x: np.ndarray) -> np.ndarray:
        return x
    
    def __d_linear(self,x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape)
    
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
        return 1 - np.power(self.__tanh(x), 2)
    
    # relu --------------------
    def __relu(self,x: np.ndarray) -> np.ndarray:
        return np.maximum(0,x)
    
    def __d_relu(self,x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
    # leaky relu --------------------
    def __prelu(self,x: np.ndarray, alpha: float=0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha*x)
    
    def __d_prelu(self,x: np.ndarray, alpha: float=0.01) -> np.ndarray:
        return np.where(x > 0, 1, alpha)