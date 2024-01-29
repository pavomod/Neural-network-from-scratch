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
PATH_TEST = "neural_network\\dataset\data_train_val\\test_set.csv"
IS_CUP = True
DIM_TRAINING_SET = 0.8
NAME_MONK = "monks-1"



class NeuralNetworkGridSearch:
    def __init__(self, settings):
        self.settings = settings

        self.k_folds = settings['validation'][0]['k_folds']
        self.best_model = None
        self.best_accuracy = 0
        self.best_params = None
        self.best_score = 0

    def generate_tr_vl_sets(self):
        if settings['validation'][0]['enable_k_fold']:
            k_fold_splitter(self.k_folds, IS_CUP, NAME_MONK)

        else:
            simple_splitter(DIM_TRAINING_SET, IS_CUP, NAME_MONK)



    def generate_parameter_combinations(self):
        keys, values = zip(*self.settings.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations
    
    def load_simple_data(self,_):

        # Qui dovrai dividere il dataset in input e target, ad esempio:
        input_training, target_training = read_dataset_cup(PATH_TRAIN)
        input_validation, target_validation = read_dataset_cup(PATH_VALIDATION)
    
        return input_training, target_training, input_validation, target_validation

    def load_k_fold_data(self, fold_number):
        train_set = f"neural_network\\dataset\\data_train_val\\k_fold\\training_set_fold{fold_number}.csv"
        val_set = f"neural_network\\dataset\\data_train_val\\k_fold\\validation_set_fold{fold_number}.csv"

        # Qui dovrai dividere il dataset in input e target, ad esempio:
        input_training, target_training = read_dataset_cup(train_set,"none")
        input_validation, target_validation = read_dataset_cup(val_set,"none")
    
        return input_training, target_training, input_validation, target_validation

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
                "name": "none"
            }
        }

        return default_config


    def search(self):
        #geneariamo i set di training e validation
        self.generate_tr_vl_sets()
        
        #iteriamo su tutte le combinazioni di parametri generati
        for index,setting in enumerate(self.generate_parameter_combinations()):
            total_accuracy = 0
            #se è abilitato il k-fold, allora carichiamo i dati in modo diverso
            if settings['validation'][0]['enable_k_fold']:
                load_fun=self.load_k_fold_data
                iteration=self.k_folds
            else:
                load_fun=self.load_simple_data
                iteration=1
                
            #creiamo i parametri da passare alla rete neurale
            params = self.create_network_settings(setting)
            # creiamo la rete neurale
            model = NeuralNetwork(params)
            accuracies = []

            # addestriamo la rete neurale, iterando su tutti i fold (se abilitato)
            for fold in range(0, iteration):
                # carichiamo i dati di training e validation
                input_training, target_training, input_validation, target_validation = load_fun(fold+1)
                # addestriamo la rete neurale
                model.train(input_training, target_training, input_validation, target_validation)
                # calcoliamo le performance del modello
                loss, accuracy = model.performance(model.predict(input_validation), target_validation)
                total_accuracy += accuracy
                accuracies.append(accuracy)
                
                
            # calcoliamo l'accuracy media del modello su tutti i training eseguiti
            avg_accuracy = total_accuracy / iteration    
            # calcoliamo la varianza 
            variance = sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)
            #selezioniamo il modello migliore
            score = avg_accuracy - (variance ** 0.5)
            
            if score > self.best_score:
                self.best_accuracy = avg_accuracy
                self.best_model = model
                self.best_params = model.get_params()
                self.best_score = score

            if index % 100 == 0:
                    print(f"Iterazioni effettuate: {index}/{len(self.generate_parameter_combinations())}")
                    
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

def read_dataset_cup(path,encoder_name='none'):
    df = pd.read_csv(path, sep=",", header=None)
    #df.drop(columns=[df.columns[-1]], inplace=True)
    
    x = df.iloc[:, 1:11].values # le prime 10 colonne tranne la prima
    
    y = df.iloc[:, 11:].values # gli ultimi 3 valori
    #y=y.reshape(-1,1)
    encoder=Preprocessing(encoder_name)
    x=encoder.encoder(x)
    
    return x, y


settings = read_grid_search_config()


#MODEL SELECTION
grid_search = NeuralNetworkGridSearch(settings)
model, accuracy, params = grid_search.search()

#write params on file
with open("neural_network\\configuration\\best_params.json", 'w') as file:
    json.dump(params, file, indent=4)

#ACCURACY VALIDATION MODELLO FINALE
print("="*10)
print("accuracy -> "+str(accuracy))

#LEARNING CURVE
model.plot_loss_curve(0,accuracy)


# TEST


# x_test,y_test = read_dataset(PATH_TEST,"none")
# model.test(x_test,y_test)



