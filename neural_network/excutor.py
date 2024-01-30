import json
import pandas as pd
from classes import Preprocessing
from neural_network import NeuralNetwork
from dataset import simple_splitter

PATH_CONFIG = "neural_network\\configuration\\neural_network_config.json"
PATH_TRAIN = "neural_network\\dataset\\data_train_val\\training_set.csv"
PATH_VALIDATION = "neural_network\\dataset\data_train_val\\validation_set.csv"
PATH_RETRAIN = "neural_network\\dataset\data_train_val\\retrain_set.csv"
PATH_TEST = "neural_network\\dataset\data_train_val\\test_set.csv"




# read neural network configuration
def read_neural_network_config():
    """
    Legge un file di configurazione JSON per una rete neurale e salva i dati in un oggetto Python.

    Returns:
    dict: Un dizionario contenente i dati della configurazione della rete neurale.
    """
    with open(PATH_CONFIG, 'r') as file:
        config = json.load(file)
    return config


# read dataset about monks
def read_dataset(path, encoder_name):
    df = pd.read_csv(path, sep=" ", header=None)
    df.drop(columns=[df.columns[-1]], inplace=True)
    
    # get all columns except the first (input)
    x = df.iloc[:, 2:8].values
    
    # get the first column (output)
    y = df.iloc[:, 1].values
    y=y.reshape(-1,1)

    # preprocessing of input
    encoder=Preprocessing(encoder_name)
    x=encoder.encoder(x)
    return x, y


# read dataset about CUP
def read_dataset_cup(path, encoder_name):
    df = pd.read_csv(path, sep=",", header=None)
    
    # get input
    x = df.iloc[:, 1:11].values
    
    # get output
    y = df.iloc[:, 11:].values
    
    # preprocessing of input
    encoder=Preprocessing(encoder_name)
    x=encoder.encoder(x)
    return x, y

# create and execute neural network
def create_neural_network(retrain, test, training_set_size, isCup, name_monks):
    # read file of configuration
    config  = read_neural_network_config()
    encoder_name = config['preprocessing']['name']

    # create neural network
    nn = NeuralNetwork(config)

    # create from dataset the training set, validation set, test set and retrain set
    simple_splitter(training_set_size, isCup, name_monks)

    # set correct function to read dataset
    if isCup:
        read = read_dataset_cup
    else:
        read = read_dataset

    # get dataset
    x_train, y_train = read(PATH_TRAIN, encoder_name)
    x_validation, y_validation = read(PATH_VALIDATION, encoder_name)
    x_retrain, y_retrain = read(PATH_RETRAIN, encoder_name)
    x_test, y_test = read(PATH_TEST, encoder_name)

    #-----------------TRAIN E VALIDATION-----------------
    nn.train(x_train, y_train, x_validation, y_validation, retrain=False)

    #-----------------RETRAIN-----------------
    if retrain:
        print("Retraining...")
        nn.train(x_retrain, y_retrain, x_validation, y_validation, retrain=True)
    
    #-----------------TEST-----------------
    if test:
        print("Testing...")
        nn.test(x_test,y_test)



# define parameters
retrain=False
test=False
training_set_size=0.8
isCup=True
name_monks="monks-1"

# create and execute neural network
create_neural_network(retrain, test, training_set_size, isCup, name_monks)