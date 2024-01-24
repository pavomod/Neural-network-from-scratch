import numpy as np
from classes import Layer, LossFunction, LearningRateScheduler, EarlyStopping , plot_loss_curve
import pandas as pd

class NeuralNetwork:
    def __init__(self, settings):
        self.n_input=settings['model']['input_size']
        self.layers = []
        seed = np.random.randint(0, 2**31 - 1)
        for i in range (len(settings['model']['layers'])):
            if settings['model']['seed'] != -1:
                seed = settings['model']['seed'] 
            self.layers.append(Layer(settings['model']['layers'][i]['num_neurons'], settings['model']['layers'][i]['activation_function'], settings['model']['layers'][i]['initialization'], seed))
        print("Seed: ", seed)
        self.loss_function=LossFunction(settings['training']['loss_function'])
        self.epochs = settings['training']['epochs']
        self.batch_size = settings['training']['batch_size']
        self.print_every=settings['training']['print_every']
        self.print_loss=settings['training']['print_loss']
        self.learning_rate = LearningRateScheduler(
                                            settings['training']['learning_rate'],
                                            self.epochs,
                                            settings['training']['learning_rate_schedule']['approach'],
                                            settings['training']['learning_rate_schedule']['params']
                                        )
        self.regularization_lambda = settings['training']['regularization_lambda']
        self.momentum = settings['training']['momentum']
        
        self.early_stopping = EarlyStopping(settings['training']['early_stopping']['enabled'], settings['training']['early_stopping']['patience'], settings['training']['early_stopping']['min_delta'])

        self.weights=[]                         #! weights per ogni layer
        self.bias=[]                            #! bias per ogni layer
        self.grad_weights=[]                    #! gradienti dei pesi per ogni layer
        self.grad_bias=[]                       #! gradienti dei bias per ogni layer
        self.deltas=[]                          #! delta per ogni layer
        self.not_activated_output=[]            #! output prima di avere applicato la funzione di attivazione
        self.activated_output=[]                #! output dopo avere applicato la funzione di attivazione
        self.loss_history=[]                    #! array che contiene i valori di loss del training
        self.velocity_weights = []              #! velocità dei pesi per ogni layer
        self.velocity_bias = []                 #! velocità dei bias per ogni layer
            
        self.val_loss_history = []  # Aggiunta per tenere traccia della loss di validazione
        # inizializzazione della rete
        self.__networkInitialization()
    


    # inizializzazione dei pesi e dei bias della rete neurale
    def __networkInitialization(self):
        #! primo layer nascosto
        first_layer = self.layers[0]
        self.weights.append(first_layer.initialization((self.n_input,first_layer.num_neurons))) 
        
        #! restanti layer nascosti
        for i in range(1, len(self.layers)-1):
            self.weights.append(self.layers[i].initialization((self.layers[i-1].num_neurons,self.layers[i].num_neurons))) 
        
        #! output layer
        self.weights.append(self.layers[-1].initialization((self.layers[-2].num_neurons,self.layers[-1].num_neurons)))
                
        #! primo layer nascosto
        #! restanti layer nascosti
        for i in range(len(self.layers)):
            self.bias.append(np.zeros((1,self.layers[i].num_neurons)))
        
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_bias = [np.zeros_like(b) for b in self.bias]
            
    # propagazione in avanti dei dati di input attraverso la rete neurale --> calcolo dell'output
    def forward(self, input_data):
        self.not_activated_output=[]      
        self.activated_output=[]
        
        self.activated_output.append(input_data)

        # Propagazione attraverso gli strati nascosti
        for i in range(len(self.layers)-1):
            layer_input = self.activated_output[-1]
            z = np.dot(layer_input, self.weights[i]) + self.bias[i]
            a = self.layers[i].activation_function.function(z)
            
            self.not_activated_output.append(z)
            self.activated_output.append(a)

        # Propagazione attraverso lo strato di output
        layer_input = self.activated_output[-1]
        z = np.dot(layer_input, self.weights[-1]) + self.bias[-1]
        a = self.layers[-1].activation_function.function(z)
        
        self.not_activated_output.append(z)
        self.activated_output.append(a)

        return self.activated_output[-1]



    def backpropagation(self, y_true):
        self.deltas = [None] * (len(self.layers))
        self.grad_weights = [None] * (len(self.layers))
        self.grad_bias = [None] * (len(self.layers))
        
        # Calcolo del delta per lo strato di output
        self.deltas[-1] = self.loss_function.derivative(y_true,self.activated_output[-1]) * self.layers[-1].activation_function.derivative(self.not_activated_output[-1])
        self.grad_weights[-1] = np.dot(self.activated_output[-2].T, self.deltas[-1])
        self.grad_bias[-1] = np.sum(self.deltas[-1], axis=0, keepdims=True)
        
        # Calcolo del delta per gli strati nascosti
        for i in reversed(range(len(self.layers)-1)):
            self.deltas[i] = np.dot(self.deltas[i + 1], self.weights[i + 1].T) * self.layers[i].activation_function.derivative(self.not_activated_output[i])
            self.grad_weights[i] = np.dot(self.activated_output[i].T, self.deltas[i])
            self.grad_bias[i] = np.sum(self.deltas[i], axis=0, keepdims=True)


    def update(self, epoch):
        for i in range(len(self.weights)):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] + self.learning_rate.update_func(epoch) * (self.grad_weights[i] + self.regularization_lambda * self.weights[i])
            self.velocity_bias[i] = self.momentum * self.velocity_bias[i] + self.learning_rate.update_func(epoch) * self.grad_bias[i]
            
            self.weights[i] -= self.velocity_weights[i]
            self.bias[i] -= self.velocity_bias[i]

    def train(self, input_data, target, val_input ,val_target):
        self.loss_history = []
        self.val_loss_history = []
        self.early_stopping.reset()
        # Calcolo del numero di mini-batch
        n_samples = input_data.shape[0] 
        n_batches = int(np.ceil(n_samples / self.batch_size))

        for epoch in range(self.epochs):
            
            # Mescolare i dati 
            permutation = np.random.permutation(n_samples)
            input_data_shuffled = input_data[permutation]
            target_shuffled = target[permutation]

            for batch in range(n_batches): 
                
                start = batch * self.batch_size 
                end = min(start + self.batch_size, n_samples)   
                batch_input = input_data_shuffled[start:end]
                batch_target = target_shuffled[start:end]

                # Forward propagation, backpropagation e aggiornamento per il mini-batch
                self.forward(batch_input)
                self.backpropagation(batch_target)
                self.update(epoch)
            
            # Calcolo della loss per il training
            output_predict=self.predict(input_data)
            loss, training_accuracy = self.performance(output_predict,target)
            self.loss_history.append(loss)
            #calcola la loss per il validation
            val_output=self.predict(val_input)
            performance_loss, validation_accuracy =self.performance(val_output,val_target)
            self.val_loss_history.append(performance_loss)
            
            if self.print_loss and epoch % self.print_every == 0:
                print(f"( Epoch {epoch} ) training loss: {loss}\t validation loss: {performance_loss}")
            
            if self.early_stopping(performance_loss):
                print(f"Early stopping at epoch {epoch}")
                break
            
            
        if self.print_loss:
            self.plot_loss_curve(training_accuracy,validation_accuracy)
            
    # calcolo della predizione
    def predict(self, input_data):
        return self.forward(input_data)

    # calcolo dell'errore tra la predizione e il target
    def performance(self, output_predict, output_target):
        loss = self.loss_function.function(output_target, output_predict)
        accuracy = self.accuracy(output_target, output_predict)
        return loss,accuracy



    def accuracy(self, y_test, y_pred):
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return round(np.sum(y_test == y_pred) / len(y_test), 3) * 100
    
    def test(self, input_data, target):
        output = self.predict(input_data)
        loss_test, accuracy_test = self.performance(output, target)
        print("\n\n-------------------- TEST --------------------")
        print(f"Loss:\t{loss_test}")
        print(f"Accuracy:\t{accuracy_test}%")
    

    def plot_loss_curve(self,training_accuracy,validation_accuracy):
        plot_loss_curve(self.loss_history,self.val_loss_history,training_accuracy,validation_accuracy)

    # stampa dei pesi e dei bias della rete neurale
    def printNetwork(self):
        print("Weights:")
        for i in range(len(self.weights)):
            print(f"Layer {i+1}:")
            print(self.weights[i])
        print("Bias:")
        for i in range(len(self.bias)):
            print(f"Layer {i+1}:")
            print(self.bias[i])