import numpy as np

class Initializer:
    def __init__(self, name: str, seed: int):
        self.name = name        # Name of the initializer
        self.seed = seed        # Seed for random number generation to ensure reproducibility
        # Retrieves the appropriate initialization function based on the name provided
        self.function = self.__getFunction(name)

    def __getFunction(self, name: str):
        # Associates initializer names with their respective functions
        if name == 'uniform_initializer':
            return self.uniformInitializer
        elif name == 'he_initializer':
            return self.heInitializer
        elif name == 'xavier_initializer':
            return self.xavierInitializer
        else:
            raise ValueError(f"Initializer {name} not implemented")

    def uniformInitializer(self, shape: tuple[int, int], min: int = -0.2, max: int = 0.2):
        # Uniform initializer: initializes weights uniformly in the given range
        np.random.seed(self.seed)                   # Setting the seed for reproducibility
        return np.random.uniform(min, max, shape)   # Generating weights uniformly in the specified range
    
    def heInitializer(self, shape: tuple[int, int]):
        # He initializer
        np.random.seed(self.seed)       # Setting the seed for reproducibility
        std = np.sqrt(2 / shape[0])     # Calculating the standard deviation for He initialization
        return np.random.normal(0, std, shape)  # Generating weights with normal distribution

    def xavierInitializer(self, shape: tuple[int, int]):
        # Xavier/Glorot initializer
        np.random.seed(self.seed)           # Setting the seed for reproducibility
        std = np.sqrt(1 / shape[0])         # Calculating the standard deviation for Xavier initialization
        return np.random.normal(0, std, shape)  # Generating weights with normal distribution
