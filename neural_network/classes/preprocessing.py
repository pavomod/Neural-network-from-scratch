import numpy as np

class Preprocessing:

    def __init__(self, name: str):
        self.name = name
        self.encoder=self.__get(name)
        
        
    def __get(self, name: str):
        # if name == 'standardization':
        #     return self.__standardization
        # elif name == 'min_max':
        #     return self.__min_max
        if name == 'one_hot_encode':
            return self.__one_hot_encode
        elif name == 'none':
            return self.__none
        else:
            raise ValueError('Invalid preprocessing function')
        
        
    
    def __one_hot_encode(self,matrix):
        # Definizione del numero di valori unici per ogni colonna
        unique_values = [3, 3, 2, 3, 4, 2]

        # Calcolo del numero totale di colonne one-hot
        total_one_hot_columns = sum(unique_values)

        # Inizializzazione della matrice one-hot encoded
        encoded_matrix = np.zeros((matrix.shape[0], total_one_hot_columns))

        # Posizione di inizio per la prima colonna one-hot
        start_index = 0

        for col_index, n_values in enumerate(unique_values):
            for row_index in range(matrix.shape[0]):
                # Calcolo dell'indice all'interno della colonna one-hot
                one_hot_index = start_index + matrix[row_index, col_index] - 1
                encoded_matrix[row_index, one_hot_index] = 1

            # Aggiornamento dell'indice di partenza per la prossima colonna one-hot
            start_index += n_values

        return encoded_matrix
    
    def __none(self,matrix):
        return matrix