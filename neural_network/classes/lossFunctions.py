import math
import numpy as np



class LossFunction:
    def __init__(self):
        pass

    # Mean Squared Error -------------------------------------------------------
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.sum(np.power(y_true - y_pred, 2),axis=1))
    
    def mean_squared_error_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -(2 / y_true.shape[0])*(y_pred - y_true)
    

    # Mean Absolute Error ------------------------------------------------------
    def mean_euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2),axis=1)))
    
    # TODO - rivedere la derivata
    def mean_euclidean_distance_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return - (1 / y_true.shape[0]) * (y_pred - y_true) / np.linalg.norm(y_pred - y_true, axis=1).reshape(-1, 1)