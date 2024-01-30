import numpy as np
from typing import Callable, Tuple



class LossFunction:
    def __init__(self, name: str):
        self.name = name
        self.function, self.derivative = self.__get(name)
    
    def __get(self, name: str) -> Tuple[Callable[[float], float], Callable[[float], float]]:
        if name == 'mean_squared_error':
            return self.mean_squared_error, self.mean_squared_error_derivative
        elif name == 'mean_euclidean_distance':
            return self.mean_euclidean_distance, self.mean_euclidean_distance_derivative
        else:
            raise ValueError(f"Loss function {name} not implemented")
        

    # Mean Squared Error -------------------------------------------------------
    def mean_squared_error(self,y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Calcola la differenza tra i valori veri e quelli predetti
        error = y_true - y_pred
        # Eleva al quadrato ogni elemento della differenza
        squared_error = np.square(error)
        # Calcola la media degli errori quadratici
        mse = np.mean(squared_error)
        return mse
    
    def mean_squared_error_derivative(self,y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Calcola la differenza tra i valori veri e quelli predetti
        error = y_true - y_pred
        # Calcola la derivata della MSE
        derivative = -2 * error / len(y_true)
        return derivative
    

    # Mean Absolute Error ------------------------------------------------------
    @staticmethod
    def mean_euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2),axis=1)))