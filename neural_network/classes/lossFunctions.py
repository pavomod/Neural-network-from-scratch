import math
import numpy as np
from typing import Callable



class LossFunction:
    def __init__(self, name: str):
        self.name = name
        self.function = self.__getFunction(name)
        self.derivative = self.__getDerivative(name)
    
    def __getFunction(self, name: str)  -> Callable[[float], float]:
        if name == 'mean_squared_error':
            return self.mean_squared_error
        elif name == 'mean_euclidean_distance':
            return self.mean_euclidean_distance
        else:
            raise ValueError(f"Loss function {name} not implemented")
        
    def __getDerivative(self, name: str)  -> Callable[[float], float]:
        if name == 'mean_squared_error':
            return self.mean_squared_error_derivative
        elif name == 'mean_euclidean_distance':
            return self.mean_euclidean_distance_derivative
        else:
            raise ValueError(f"Loss function {name} not implemented")
        

    # Mean Squared Error -------------------------------------------------------
    def mean_squared_error(self,y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.sum(np.power(y_true - y_pred, 2),axis=1))
    
    def mean_squared_error_derivative(self,y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return (2 / y_true.shape[0]) * (y_true - y_pred)
    

    # Mean Absolute Error ------------------------------------------------------
    def mean_euclidean_distance(self,y_true: float, y_pred: np.ndarray) -> float:
        return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2),axis=1)))
    
    # TODO - rivedere la derivata (ordine (y_pred - y_true) e segno)
    def mean_euclidean_distance_derivative(self,y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return - (1 / y_true.shape[0]) * (y_pred - y_true) / np.linalg.norm(y_pred - y_true, axis=1).reshape(-1, 1)