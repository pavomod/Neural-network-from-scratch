import json
import pandas as pd
from classes import Preprocessing
from neural_network import NeuralNetwork


PATH_CONFIG = "neural_network\\configuration\\neural_network_config.json"

PATH_TRAIN = "neural_network\\dataset\\data_train_val\\training_set.csv"
PATH_VALIDATION = "neural_network\\dataset\data_train_val\\validation_set.csv"
PATH_RETRAIN = "neural_network\\dataset\data_train_val\\retrain_set.csv"
PATH_TEST = "neural_network\\dataset\data_train_val\\test_set.csv"


def read_neural_network_config():
    """
    Legge un file di configurazione JSON per una rete neurale e salva i dati in un oggetto Python.

    Returns:
    dict: Un dizionario contenente i dati della configurazione della rete neurale.
    """
    with open(PATH_CONFIG, 'r') as file:
        config = json.load(file)
    return config


def read_dataset(path,encoder_name):
    df = pd.read_csv(path, sep=" ", header=None)
    df.drop(columns=[df.columns[-1]], inplace=True)
    
    x = df.iloc[:, 2:8].values # tutte le colonne tranne la prima
    
    y = df.iloc[:, 1].values   # la prima colonna
    y=y.reshape(-1,1)
    encoder=Preprocessing(encoder_name)
    x=encoder.encoder(x)
    
    return x, y


def execute(nn, config, test=False):
    encoder_name=config['preprocessing']['name']
    #-----------------DATASET-----------------
    x_train,y_train = read_dataset(PATH_TRAIN,encoder_name)
    x_validaiton,y_validation = read_dataset(PATH_VALIDATION,encoder_name)
    x_retrain,y_retrain = read_dataset(PATH_RETRAIN,encoder_name)
    x_test,y_test = read_dataset(PATH_TEST,encoder_name)
    
    #-----------------TRAIN E VALIDATION-----------------    
    nn.train(x_train, y_train,x_validaiton,y_validation, retrain=False)
    
    #-----------------RETRAIN-----------------
    #nn.train(x_retrain,y_retrain,x_validaiton,y_validation, retrain=True)
    
    #-----------------TEST-----------------
    if test:
        nn.test(x_test,y_test)

    


config = read_neural_network_config()
nn = NeuralNetwork(config)
execute(nn, config, test=True)

