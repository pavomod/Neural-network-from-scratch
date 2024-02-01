import numpy as np
from typing import Callable, Tuple

class LossFunction:
    def __init__(self, name: str):
        self.name = name    # Name of the loss function
        # Retrieves the appropriate loss function and its derivative based on the name provided
        self.function, self.derivative = self.__get(name)
    
    def __get(self, name: str) -> Tuple[Callable[[float], float], Callable[[float], float]]:
        # Associates loss function names with their respective functions and derivatives
        if name == 'mean_squared_error':
            return self.mean_squared_error, self.mean_squared_error_derivative
        elif name == 'mean_euclidean_distance':
            return self.mean_euclidean_distance, self.mean_euclidean_distance_derivative
        else:
            raise ValueError(f"Loss function {name} not implemented")

    # Mean Squared Error and its derivative
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Calculates the difference between true and predicted values
        error = y_true - y_pred
        # Squares each element of the difference
        squared_error = np.square(error)
        # Calculates the mean of the squared errors
        mse = np.mean(squared_error)
        return mse

    def mean_squared_error_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Calculates the difference between true and predicted values
        error = y_true - y_pred
        # Calculates the derivative of the mean squared error
        derivative = -2 * error / len(y_true)
        return derivative

    # Mean Euclidean Distance
    @staticmethod
    def mean_euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Calculates the Euclidean distance between the true and predicted values
        # and then computes the mean of these distances
        return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=1)))
