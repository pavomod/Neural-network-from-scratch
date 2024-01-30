import numpy as np

class Preprocessing:

    def __init__(self, name: str):
        self.name = name
        self.encoder = self.__get(name)
        
    def __get(self, name: str):
        # Return the appropriate preprocessing function based on the name
        if name == 'standardization':
            return self.__standardization
        elif name == 'normalization':
            return self.__normalization
        if name == 'one_hot_encode':
            return self.__one_hot_encode
        elif name == 'none':
            return self.__none
        else:
            # If an invalid name is provided, raise an error
            raise ValueError('Invalid preprocessing function')
        
    # Returns the matrix as is (no preprocessing)
    def __none(self, matrix):
        return matrix
    
    # Define the number of unique values for each column (hardcoded for simplicity)
    def __one_hot_encode(self, matrix):
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
    
    # Calculate the mean and standard deviation for each column
    def __standardization(self, matrix):
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        # Standardize the matrix
        standardized_matrix = (matrix - mean) / std
        return standardized_matrix
    
    # Find the minimum and maximum values for each column
    def __normalization(self, matrix):
        min_val = np.min(matrix, axis=0)
        max_val = np.max(matrix, axis=0)
        # Normalize the matrix
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix