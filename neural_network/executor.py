import json
import pandas as pd
from classes import Preprocessing, plot_loss_curve, plot_accuracy_curve
from neural_network import NeuralNetwork
from dataset import simple_splitter, splitter_tr_vl_ts



# File paths for configuration and datasets
PATH_CONFIG = "neural_network\\configuration\\neural_network_config.json"
PATH_TRAIN = "neural_network\\dataset\\data_train_val\\training_set.csv"
PATH_VALIDATION = "neural_network\\dataset\\data_train_val\\validation_set.csv"
PATH_RETRAIN = "neural_network\\dataset\\data_train_val\\retrain_set.csv"
PATH_TEST = "neural_network\\dataset\\data_train_val\\test_set.csv"



def read_neural_network_config():
    # Opening and reading the configuration file
    with open(PATH_CONFIG, 'r') as file:
        config = json.load(file)
    return config


def read_dataset(path, encoder_name="standardization"):
    # Reading the dataset from the file
    df = pd.read_csv(path, sep=" ", header=None)
    # Dropping the last column
    df.drop(columns=[df.columns[-1]], inplace=True)

    # Extracting input features (excluding the first column)
    x = df.iloc[:, 2:8].values

    # Extracting output targets (first column)
    y = df.iloc[:, 1].values
    y = y.reshape(-1, 1)

    # Preprocessing the input data
    encoder = Preprocessing(encoder_name)
    x = encoder.encoder(x)
    return x, y


def read_dataset_cup(path, encoder_name="standardization"):
    # Reading the CUP dataset from the file
    df = pd.read_csv(path, sep=",", header=None)

    # Extracting input features
    x = df.iloc[:, 1:11].values

    # Extracting output targets
    y = df.iloc[:, 11:].values

    # Preprocessing the input data
    encoder = Preprocessing(encoder_name)
    x = encoder.encoder(x)
    return x, y


def read_blind(path, encoder_name="normalization"):
    # Reading the blind dataset from the file
    df = pd.read_csv(path, sep=",", header=None, skiprows=7)

    # Extracting input features
    x = df.iloc[:, 1:11].values

    # Preprocessing the input data
    encoder = Preprocessing(encoder_name)
    x = encoder.encoder(x)
    return x


def create_neural_network(retrain, test, training_set_size, isCup, name_monks, blind_test):
    """
    Creates and executes a neural network based on the provided parameters and configuration.

    :param retrain: Boolean flag to indicate retraining.
    :param test: Boolean flag to indicate if testing should be performed.
    :param training_set_size: Size of the training set as a fraction.
    :param isCup: Boolean flag indicating whether the dataset is CUP or not.
    :param name_monks: Name of the MONK's dataset.
    :param blind_test: Boolean flag to indicate if blind testing should be performed.
    """

    # Read neural network configuration
    config = read_neural_network_config()
    encoder_name = config['preprocessing']['name']

    # Create a neural network instance
    nn = NeuralNetwork(config)

    # Create training, validation, test, and retrain sets from the dataset
    splitter_tr_vl_ts()

    # Set the correct function to read the dataset based on its type
    if isCup:
        read = read_dataset_cup
    else:
        read = read_dataset

    # Read datasets for training, validation, retraining, testing, and blind testing
    x_train, y_train = read(PATH_TRAIN, encoder_name)
    x_validation, y_validation = read(PATH_VALIDATION, encoder_name)
    x_retrain, y_retrain = read(PATH_RETRAIN, encoder_name)
    x_test, y_test = read(PATH_TEST, encoder_name)
    x_blind_test = read_blind("neural_network\\dataset\\ML-CUP23-TS.csv", encoder_name)

    # Training and validation
    tr_loss, vl_loss, tr_accuracy, vl_accuracy = nn.train(x_train, y_train, x_validation, y_validation, retrain=False)
    
    # Output summary of training and validation
    print("SUMMARY TRAINING")
    print("=" * 10)
    print("tr_accuracy:", tr_accuracy[-1], "%")
    print("tr_loss:", tr_loss[-1])
    print("SUMMARY VALIDATION")
    print("=" * 10)
    print("vl_accuracy:", vl_accuracy[-1], "%")
    print("vl_loss:", vl_loss[-1])

    # Retraining, if required
    if retrain:
        print("Retraining...")
        nn.train(x_retrain, y_retrain, x_validation, y_validation, retrain=True)

    # Testing, if required
    if test:
        print("Testing...")
        nn.test(x_test, y_test)

    # Blind testing, if required
    if blind_test:
        print("Blind Testing...")
        output = nn.predict(x_blind_test)
        # Save output to a file
        with open("neural_network\\dataset\\blind_test\\Neurons_Not_Found_ML-CUP23-TS.csv", 'w') as file:
            for i in range(len(output)):
                file.write(f"{i+1},{output[i][0]},{output[i][1]},{output[i][2]}\n")


def mean_execute(retrain, test, training_set_size, isCup, name_monks, train_number=10):
    # Read neural network configuration
    config = read_neural_network_config()
    encoder_name = config['preprocessing']['name']

    # Create a neural network instance
    nn = NeuralNetwork(config)

    # Create training, validation, test, and retrain sets from the dataset
    simple_splitter(training_set_size, isCup, name_monks)

    # Set the correct function to read the dataset based on its type
    if isCup:
        read = read_dataset_cup
    else:
        read = read_dataset

    # Read datasets for training, validation, and testing
    x_train, y_train = read(PATH_TRAIN, encoder_name)
    x_validation, y_validation = read(PATH_VALIDATION, encoder_name)
    x_test, y_test = read(PATH_TEST, encoder_name)
    
    # Initialize variables for averaging the results
    avg_train_loss = []
    avg_train_accuracy_loss = []
    avg_validation_loss = []
    avg_validation_accuracy_loss = []

    # Training and validation process
    for i in range(train_number):
        # Create a new instance of the neural network for each training iteration
        nn = NeuralNetwork(config)
        # Train the neural network and get the loss and accuracy metrics
        tr_loss, vl_loss, tr_acc, vl_acc = nn.train(x_train, y_train, x_validation, y_validation, retrain=False)

        # Summing up the metrics for averaging later
        if i == 0:
            avg_train_loss = tr_loss
            avg_train_accuracy_loss = tr_acc
            avg_validation_loss = vl_loss
            avg_validation_accuracy_loss = vl_acc
        else:
            avg_train_loss = [x + y for x, y in zip(avg_train_loss, tr_loss)]
            avg_train_accuracy_loss = [x + y for x, y in zip(avg_train_accuracy_loss, tr_acc)]
            avg_validation_loss = [x + y for x, y in zip(avg_validation_loss, vl_loss)]
            avg_validation_accuracy_loss = [x + y for x, y in zip(avg_validation_accuracy_loss, vl_acc)]

    # Calculating the average metrics
    avg_train_loss = [x / train_number for x in avg_train_loss]
    avg_validation_loss = [x / train_number for x in avg_validation_loss]
    avg_train_accuracy_loss = [x / train_number for x in avg_train_accuracy_loss]
    avg_validation_accuracy_loss = [x / train_number for x in avg_validation_accuracy_loss]

    # Output and plot the summary of training
    print("SUMMARY TRAINING")
    print("=" * 10)
    print("avg_train_loss:", avg_train_loss[-1])
    print("avg_validation_loss:", avg_validation_loss[-1])
    print("avg_train_accuracy_loss:", avg_train_accuracy_loss[-1], "%")
    print("avg_validation_accuracy_loss:", avg_validation_accuracy_loss[-1], "%")
    plot_loss_curve(avg_train_loss, avg_validation_loss)
    plot_accuracy_curve(avg_train_accuracy_loss, avg_validation_accuracy_loss)

    # Perform testing if required
    if test:
        nn.test(x_test, y_test)

    


# define parameters
retrain=True
test=True
training_set_size=0.8
isCup=True
name_monks="monks-3"
train_number=10
blind_test=True

# create and execute neural network
#create_neural_network(retrain, test, training_set_size, isCup, name_monks,blind_test)
mean_execute(retrain, test, training_set_size, isCup, name_monks,train_number)

