import json
import pandas as pd
from neural_network import NeuralNetwork


FILE_PATH = "neural_network\\neural_network_config.json"

def read_neural_network_config():
    """
    Legge un file di configurazione JSON per una rete neurale e salva i dati in un oggetto Python.

    Returns:
    dict: Un dizionario contenente i dati della configurazione della rete neurale.
    """
    with open(FILE_PATH, 'r') as file:
        config = json.load(file)
    return config





def execute(nn, path_train, path_validation, path_test, test=False):
    #-----------------TRAIN-----------------
    df = pd.read_csv(path_train, sep=" ", header=None)
    df.drop(columns=[df.columns[-1]], inplace=True)
    x = df.iloc[:, 2:8].values # tutte le colonne tranne la prima
    y = df.iloc[:, 1].values   # la prima colonna
    y=y.reshape(-1,1)
    print("Train")
    nn.train(x, y)

    #-----------------TEST-----------------
    if test:
        dt = pd.read_csv(path_test, sep=" ", header=None)
        dt.drop(columns=[dt.columns[-1]], inplace=True)
        xTest = dt.iloc[:, 2:8].values # tutte le colonne tranne la prima
        yTest = dt.iloc[:, 1].values   # la prima colonna
        yTest=yTest.reshape(-1,1)
        print("Test")
        nn.test(xTest, yTest)





config = read_neural_network_config()
nn = NeuralNetwork(config)
execute(nn, path_train='neural_network\\dataset\\monks-3.train', path_validation='neural_network\\dataset\\monks-3.train', path_test='neural_network\\dataset\\monks-3.test', test=True)






