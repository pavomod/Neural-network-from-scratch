import itertools
import json
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from classes import Preprocessing

PATH_CONFIG = "neural_network\\configuration\\grid_search_config.json"
PATH_TRAIN = "neural_network\\dataset\\data_train_val\\training_set.csv"
PATH_VALIDATION = "neural_network\\dataset\data_train_val\\validation_set.csv"
PATH_TEST = "neural_network\\dataset\data_train_val\\test_set.csv"
class NeuralNetworkGridSearch:
    def __init__(self, settings, input_training, target_training, input_validation, target_validation):
        self.settings = settings
        self.input_training = input_training
        self.target_training = target_training
        self.input_validation = input_validation
        self.target_validation = target_validation

        self.best_model = None
        self.best_accuracy = 0
        self.best_params = None

    def generate_parameter_combinations(self):
        keys, values = zip(*self.settings.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations


    def create_network_settings(self, combination):

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
                "name": "one_hot_encode"
            }
        }

        return default_config


    def search(self):
        conta=0
        for setting in self.generate_parameter_combinations():
            conta+=1
            params = self.create_network_settings(setting)

            model = NeuralNetwork(params)
            model.train(self.input_training, self.target_training, self.input_validation, self.target_validation)
            loss, accuracy = model.performance(model.predict(self.input_validation), self.target_validation)
    
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_params = model.get_params()

            if conta % 100 == 0:
                print(f"Iterazioni effettuate: {conta}/{len(self.generate_parameter_combinations())}")
        return self.best_model, self.best_accuracy, self.best_params






def read_grid_search_config():
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


settings = read_grid_search_config()
x_train,y_train = read_dataset(PATH_TRAIN,"one_hot_encode")
x_validaiton,y_validation = read_dataset(PATH_VALIDATION,"one_hot_encode")
x_test,y_test = read_dataset(PATH_TEST,"one_hot_encode")

grid_search = NeuralNetworkGridSearch(settings, x_train,y_train, x_validaiton,y_validation)
model, accuracy, params =grid_search.search()
print("="*10)
print("accuracy -> "+str(accuracy))

model.test(x_test,y_test)

model.plot_loss_curve(0,accuracy)
# p = grid_search.generate_parameter_combinations()
# print(p)
# print(grid_search.create_network_settings(p[0]))


# best_model, best_accuracy, best_params = grid_search.search()
# print(f"Best Model Accuracy: {best_accuracy}")
# print(f"Best Model Parameters: {best_params}")
