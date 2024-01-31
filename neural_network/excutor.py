import json
import pandas as pd
from classes import Preprocessing, plot_loss_curve, plot_accuracy_curve
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


def mean_execute(retrain, test, training_set_size, isCup, name_monks,train_number=10):
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
    #-----------------DATASET-----------------
    x_train,y_train = read(PATH_TRAIN,encoder_name)
    x_validaiton,y_validation = read(PATH_VALIDATION,encoder_name)
    #x_retrain,y_retrain = read(PATH_RETRAIN,encoder_name)
    x_test,y_test = read(PATH_TEST,encoder_name)
    
    #-----------------TRAIN E VALIDATION-----------------   
    avg_train_loss=[]
    avg_train_accuracy_loss=[]
    avg_validation_loss=[]
    avg_validation_accuracy_loss=[]
    for i in range(train_number):
        nn = NeuralNetwork(config)
        tr_loss,vl_loss,tr_acc,vl_acc = nn.train(x_train, y_train,x_validaiton,y_validation, retrain=False)
        if i==0:
            avg_train_loss=tr_loss
            avg_train_accuracy_loss=tr_acc
            avg_validation_loss=vl_loss
            avg_validation_accuracy_loss=vl_acc
        else:
            avg_train_loss = [x+y for x,y in zip(avg_train_loss, tr_loss)]
            avg_train_accuracy_loss = [x+y for x,y in zip(avg_train_accuracy_loss, tr_acc)]
            avg_validation_loss = [x+y for x,y in zip(avg_validation_loss, vl_loss)]
            avg_validation_accuracy_loss = [x+y for x,y in zip(avg_validation_accuracy_loss, vl_acc)]
    #fai la media
    
    avg_train_loss = [x / train_number for x in avg_train_loss]
    avg_validation_loss = [x / train_number for x in avg_validation_loss]
    avg_train_accuracy_loss = [x / train_number for x in avg_train_accuracy_loss]
    avg_validation_accuracy_loss = [x / train_number for x in avg_validation_accuracy_loss]
    
    #plot
    print("SUMMARY TRAINING")
    print("="*10)
    print("avg_train_loss")
    print(avg_train_loss[-1])
    print("avg_validation_loss")
    print(avg_validation_loss[-1])
    print("avg_train_accuracy_loss")
    print(avg_train_accuracy_loss[-1],"%")
    print("avg_validation_accuracy_loss")
    print(avg_validation_accuracy_loss[-1],"%")
    plot_loss_curve(avg_train_loss,avg_validation_loss)
    plot_accuracy_curve(avg_train_accuracy_loss,avg_validation_accuracy_loss)
    if test:
        nn.test(x_test,y_test)
    

# define parameters
retrain=False
test=True
training_set_size=0.8
isCup=False
name_monks="monks-3"
train_number=10
# create and execute neural network
#create_neural_network(retrain, test, training_set_size, isCup, name_monks)
mean_execute(retrain, test, training_set_size, isCup, name_monks,train_number)
