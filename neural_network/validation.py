import itertools
import json
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from classes import Preprocessing
from dataset import simple_splitter, k_fold_splitter


PATH_CONFIG = "neural_network\\configuration\\grid_search_config.json"
PATH_TRAIN = "neural_network\\dataset\\data_train_val\\training_set.csv"
PATH_VALIDATION = "neural_network\\dataset\data_train_val\\validation_set.csv"
PATH_RETRAIN = "neural_network\\dataset\data_train_val\\retrain_set.csv"
PATH_HOLD_OUT = "neural_network\\dataset\data_train_val\\k_fold\\hold_out.csv"
PATH_TEST = "neural_network\\dataset\data_train_val\\test_set.csv"
IS_CUP = True
DIM_TRAINING_SET = 0.8
NAME_MONK = "monks-1"



class NeuralNetworkGridSearch:
    def __init__(self, settings):
        self.settings = settings

        # Retrieve k-folds setting from configuration
        self.k_folds = settings['validation'][0]['k_folds']
        # Variables to store the best model and its parameters
        self.best_model = None
        self.best_accuracy = 0
        self.best_params = None
        # List to keep track of the top five models
        self.top_five = []
        
        # Set the best score initial value based on the dataset type
        if IS_CUP:
            self.best_score = +np.inf
        else:
            self.best_score = -np.inf

    def generate_tr_vl_sets(self):
        if self.settings['validation'][0]['enable_k_fold']:
            # Generate K-folds if enabled in settings
            k_fold_splitter(k_folds=self.k_folds, isCup=IS_CUP, name_monks=NAME_MONK)
        else:
            # Otherwise, do a simple split
            simple_splitter(DIM_TRAINING_SET, IS_CUP, NAME_MONK)

    def generate_parameter_combinations(self):
        keys, values = zip(*self.settings.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations
    
    def load_simple_data(self, _):
        if IS_CUP:
            input_training, target_training = read_dataset_cup(PATH_TRAIN)
            input_validation, target_validation = read_dataset_cup(PATH_VALIDATION)
        else:
            input_training, target_training = read_dataset(PATH_TRAIN, "one_hot_encode")
            input_validation, target_validation = read_dataset(PATH_VALIDATION, "one_hot_encode")
    
        return input_training, target_training, input_validation, target_validation

    def load_k_fold_data(self, fold_number):
        train_set = f"neural_network\\dataset\\data_train_val\\k_fold\\training_set_fold{fold_number}.csv"
        val_set = f"neural_network\\dataset\\data_train_val\\k_fold\\validation_set_fold{fold_number}.csv"

        if IS_CUP:
            input_training, target_training = read_dataset_cup(train_set)
            input_validation, target_validation = read_dataset_cup(val_set)
        else:
            input_training, target_training = read_dataset(train_set, "one_hot_encode")
            input_validation, target_validation = read_dataset(val_set, "one_hot_encode")
    
        return input_training, target_training, input_validation, target_validation

    def create_network_settings(self, combination):
        # create json configuration for neural network
        layers = []
        for _ in range(combination['num_hidden_layers']):
            layers.append({
                "num_neurons": combination['num_neurons_hidden_layers'],
                "activation_function": combination['activation_function_hidden'],
                "initialization": combination['initialization_hidden']
            })

        # setting layer di output
        layers.append({
            "num_neurons": combination['output_size'],
            "activation_function": combination['activation_function_output'],
            "initialization": combination['initialization_output']
        })
        
        default_config = {
            "model": {
                "input_size": combination['input_size'],
                "output_size": combination['output_size'],
                "seed": combination['seed'],
                "layers": layers,
            },
            "training": {
                "loss_function": "mean_squared_error",
                "epochs": combination['epochs'],
                "batch_size": combination['batch_size'],
                "print_every": 100,
                "print_loss": False,
                "learning_rate": combination['learning_rate'],
                "regularization_lambda": combination['regularization_lambda'],
                "momentum": combination['momentum'],
                "learning_rate_schedule": {
                    "approach": combination['approach'],
                    "params": {
                        "min_lr": combination['min_lr'],
                        "decay_rate": combination['decay_rate'],
                        "step_size": combination['step_size'],
                        "warmup_epochs": combination['warmup_epochs']
                    }
                },
                "early_stopping": {
                    "enabled": combination['enabled'],
                    "patience": combination['patience'],
                    "min_delta": combination['min_delta']
                }
            },
            "preprocessing": {
                "name": "standardization"
            }
        }

        return default_config

    def search(self):
        # Generate training and validation sets
        self.generate_tr_vl_sets()

        # Iterate over all generated parameter combinations
        for index, setting in enumerate(self.generate_parameter_combinations()):
            total_accuracy = 0

            # Decide the data loading function based on k-fold validation setting
            if settings['validation'][0]['enable_k_fold']:
                load_fun = self.load_k_fold_data
                iteration = self.k_folds
            else:
                load_fun = self.load_simple_data
                iteration = 1

            # Create parameters for the neural network
            params = self.create_network_settings(setting)
            # Create the neural network
            model = NeuralNetwork(params)
            accuracies = []

            # Train the neural network, iterating over all folds if k-fold is enabled
            for fold in range(iteration):
                # Load training and validation data
                input_training, target_training, input_validation, target_validation = load_fun(fold + 1)
                # Train the neural network
                model.train(input_training, target_training, input_validation, target_validation)
                # Calculate the model's performance
                loss, accuracy = model.performance(model.predict(input_validation), target_validation)
                total_accuracy += accuracy
                accuracies.append(accuracy)

            # Calculate the model's average accuracy
            avg_accuracy = total_accuracy / iteration
            # Calculate the variance
            variance = sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)
            # Calculate a score considering both accuracy and variance
            score = avg_accuracy - (variance ** 0.5)

            # Determine if this model is the best one based on the score
            if IS_CUP:
                check = score < self.best_score
            else:
                check = score > self.best_score

            if check:
                # Update the best model, its accuracy, parameters, and score
                self.best_accuracy = avg_accuracy
                self.best_model = model
                self.best_params = model.get_params()
                self.best_score = score
                self.update_top_five_models(model, score, avg_accuracy, params)

            # Print progress after every 5 iterations
            if index % 5 == 0:
                print(f"Iterations completed: {index}/{len(self.generate_parameter_combinations())}")

        # Save the top five models to a file
        with open("neural_network\\configuration\\top_five.json", 'w') as file:
            json.dump(self.top_five, file, indent=4)

        return self.best_model, self.best_accuracy, self.best_params

    def update_top_five_models(self, model, score, accuracy, params):
        model_info = {
            'score': score,
            'accuracy': accuracy,
            'params': params
        }

        # Add the model's information to the top five list
        self.top_five.append(model_info)
        # Sort the list by score
        reverse = True if IS_CUP else False
        self.top_five.sort(key=lambda x: x['score'], reverse=reverse)
        # Keep only the top five models
        self.top_five = self.top_five[:5]
        


def read_grid_search_config():
    with open(PATH_CONFIG, 'r') as file:
        config = json.load(file)
    return config

def read_dataset(path, encoder_name):
    df = pd.read_csv(path, sep=" ", header=None)
    df.drop(columns=[df.columns[-1]], inplace=True)
    
    x = df.iloc[:, 2:8].values # All columns except the first
    y = df.iloc[:, 1].values   # The first column
    y = y.reshape(-1, 1)

    encoder = Preprocessing(encoder_name)
    x = encoder.encoder(x)
    
    return x, y

def read_dataset_cup(path, encoder_name='standardization'):
    df = pd.read_csv(path, sep=",", header=None)
    
    x = df.iloc[:, 1:11].values # First 10 columns except the first
    y = df.iloc[:, 11:].values # The last 3 values

    encoder = Preprocessing(encoder_name)
    x = encoder.encoder(x)
    
    return x, y




# Read grid search configuration
settings = read_grid_search_config()

# Model selection using grid search
grid_search = NeuralNetworkGridSearch(settings)
model, accuracy, params = grid_search.search()

# Write the best parameters to a file
with open("neural_network\\configuration\\best_params.json", 'w') as file:
    json.dump(params, file, indent=4)

# Output accuracy of the final model
print("=" * 10)
print("accuracy -> " + str(accuracy))

# Learning curve plotting and model testing
if IS_CUP:
    x_train, y_train = read_dataset_cup(PATH_RETRAIN)
    x_val, y_val = read_dataset_cup(PATH_VALIDATION)
    x_test, y_test = read_dataset_cup(PATH_HOLD_OUT)
    model.test(x_test, y_test)
    model.plot_loss_curve(0, 0)
else:
    x_train, y_train = read_dataset(PATH_TRAIN, "none")
    x_val, y_val = read_dataset(PATH_VALIDATION, "none")
    model.train(x_train, y_train, x_val, y_val)
    x_test, y_test = read_dataset(PATH_HOLD_OUT, "none")
    model.test(x_test, y_test)
    model.plot_loss_curve(0, 0)


