import numpy as np

class Preprocessing:
    def __init__(self, name: str):
        self.name = name    # Name of the preprocessing technique
        # Retrieves the appropriate preprocessing function based on the name provided
        self.encoder = self.__get(name)
        
    def __get(self, name: str):
        # Associates preprocessing technique names with their respective functions
        if name == 'standardization':
            return self.__standardization
        elif name == 'normalization':
            return self.__normalization
        if name == 'one_hot_encode':
            return self.__one_hot_encode
        elif name == 'none':
            return self.__none
        else:
            raise ValueError('Invalid preprocessing function')

    # No preprocessing: returns the matrix as is
    def __none(self, matrix):
        return matrix

    # One-hot encoding preprocessing
    def __one_hot_encode(self, matrix):
        # Define the number of unique values for each column (hardcoded for simplicity)
        unique_values = [3, 3, 2, 3, 4, 2]

        # Calculate the total number of one-hot columns
        total_one_hot_columns = sum(unique_values)

        # Initialize the one-hot encoded matrix
        encoded_matrix = np.zeros((matrix.shape[0], total_one_hot_columns))

        # Start index for the first one-hot column
        start_index = 0

        # Iterate over each column to create one-hot encoded vectors
        for col_index, n_values in enumerate(unique_values):
            for row_index in range(matrix.shape[0]):
                # Calculate the index within the one-hot column
                one_hot_index = start_index + matrix[row_index, col_index] - 1
                encoded_matrix[row_index, one_hot_index] = 1

            # Update the start index for the next one-hot column
            start_index += n_values

        return encoded_matrix

    # Standardization preprocessing
    def __standardization(self, matrix):
        # Calculate the mean and standard deviation for each column
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        # Standardize the matrix
        standardized_matrix = (matrix - mean) / std
        return standardized_matrix

    # Normalization preprocessing
    def __normalization(self, matrix):
        # Find the minimum and maximum values for each column
        min_val = np.min(matrix, axis=0)
        max_val = np.max(matrix, axis=0)
        # Normalize the matrix
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix
