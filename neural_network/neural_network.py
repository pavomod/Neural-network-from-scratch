import numpy as np
from classes import Initializer, ActivationFunction, LossFunction, plot_loss_curve
import pandas as pd

class NeuralNetwork:
    def __init__(self,n_hidden_unit=3,n_hidden_layer=1,n_input=10,n_output_unit=3,activation_function='relu',output_activation_function='sigmoid',loss_function='mean_squared_error'):
        self.n_input=n_input                    #! numero di unità di input
        self.n_hidden_unit=n_hidden_unit        #! numero di unità per ogni layer nascosto
        self.n_output_unit=n_output_unit        #! numero di unità di output
        self.n_hidden_layer=n_hidden_layer      #! numero di layer nascosti
        self.weights=[]                         #! weights per ogni layer
        self.bias=[]                            #! bias per ogni layer
        self.grad_weights=[]                    #! gradienti dei pesi per ogni layer
        self.grad_bias=[]                       #! gradienti dei bias per ogni layer
        self.deltas=[]                          #! delta per ogni layer
        self.not_activated_output=[]            #! output prima di avere applicato la funzione di attivazione
        self.activated_output=[]                #! output dopo avere applicato la funzione di attivazione
        self.loss_history=[]                    #! array che contiene i valori di loss del training

        self.initializer=Initializer()
        self.__networkInitialization()
        self.activation_function=ActivationFunction(activation_function)
        self.output_activation_function=ActivationFunction(output_activation_function)
        self.loss_function=LossFunction(loss_function)

        
    # inizializzazione dei pesi e dei bias della rete neurale
    def __networkInitialization(self):
        #! primo layer nascosto
        self.weights.append(self.initializer.uniformInitializer(-0.5,0.5,(self.n_input,self.n_hidden_unit))) 
        
        #! restanti layer nascosti
        for _ in range(self.n_hidden_layer -1):
            self.weights.append(self.initializer.uniformInitializer(-0.5,0.5,(self.n_hidden_unit,self.n_hidden_unit))) 
        #! output layer
        self.weights.append(self.initializer.uniformInitializer(-0.5,0.5,(self.n_hidden_unit,self.n_output_unit)))
                
        #! primo layer nascosto
        self.bias.append(np.zeros((1,self.n_hidden_unit)))
        #! restanti layer nascosti
        for _ in range(self.n_hidden_layer -1):
            self.bias.append(np.zeros((1,self.n_hidden_unit)))
        #! output layer   
        self.bias.append(np.zeros((1,self.n_output_unit)))
            
    # propagazione in avanti dei dati di input attraverso la rete neurale --> calcolo dell'output
    def forward(self, input_data):
        self.not_activated_output=[]      
        self.activated_output=[]
        
        self.activated_output.append(input_data)

        # Propagazione attraverso gli strati nascosti
        for i in range(self.n_hidden_layer):
            layer_input = self.activated_output[-1]
            z = np.dot(layer_input, self.weights[i]) + self.bias[i]
            a = self.activation_function.function(z)
            
            self.not_activated_output.append(z)
            self.activated_output.append(a)

        # Propagazione attraverso lo strato di output
        layer_input = self.activated_output[-1]
        z = np.dot(layer_input, self.weights[-1]) + self.bias[-1]
        a = self.output_activation_function.function(z)
        
        self.not_activated_output.append(z)
        self.activated_output.append(a)

        return self.activated_output[-1]



    def backpropagation(self, y_true):
        self.deltas = [None] * (self.n_hidden_layer + 1)
        self.grad_weights = [None] * (self.n_hidden_layer + 1)
        self.grad_bias = [None] * (self.n_hidden_layer + 1)
        
        # Calcolo del delta per lo strato di output
        self.deltas[-1] = self.loss_function.derivative(y_true, self.activated_output[-1]) * self.output_activation_function.derivative(self.not_activated_output[-1])
        self.grad_weights[-1] = np.dot(self.activated_output[-2].T, self.deltas[-1])
        self.grad_bias[-1] = np.sum(self.deltas[-1], axis=0, keepdims=True)
        
        # Calcolo del delta per gli strati nascosti
        for i in reversed(range(self.n_hidden_layer)):
            self.deltas[i] = np.dot(self.deltas[i + 1], self.weights[i + 1].T) * self.activation_function.derivative(self.not_activated_output[i])
            self.grad_weights[i] = np.dot(self.activated_output[i].T, self.deltas[i])
            self.grad_bias[i] = np.sum(self.deltas[i], axis=0, keepdims=True)


    def update(self, learning_rate=0.01, lambda_reg=0.0):
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * (self.grad_weights[i] + lambda_reg * self.weights[i])
            self.bias[i] += learning_rate * (self.grad_bias[i])


    def train(self, input_data, target, learning_rate=0.01, epochs=124):
        self.loss_history = []

        for i in range(epochs):
            self.forward(input_data)
            self.backpropagation(target)
            self.update(learning_rate)
            loss = self.loss_function.function(target, self.activated_output[-1])
            self.loss_history.append(loss)
            
            if i % 1000 == 0:
                print(f"Epoch {i}: {loss}")
        
        self.plot_loss_curve()
        self.test(input_data, target, 'TRAINING')
                
    # calcolo della predizione
    def predict(self, input_data):
        return self.forward(input_data)

    # calcolo dell'errore tra la predizione e il target
    def test(self, test_input_data, test_target, name='TEST'):
        output = self.predict(test_input_data)
        print(f"\n\n-------------------- {name} --------------------")
        print(f"Loss:\t{self.loss_function.function(test_target, output)}")
        print(f"Accuracy:\t{self.accuracy(test_target, output)}%")


    def accuracy(self, y_test, y_pred):
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return round(np.sum(y_test == y_pred) / len(y_test), 3) * 100
    

    def plot_loss_curve(self):
        plot_loss_curve(self.loss_history)

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
        
    


#self,n_hidden_unit=4,n_hidden_layer=1,n_input=10,n_output_unit=3,activation_function='sigmoid',loss_function='mean_squared_error'
nn = NeuralNetwork(n_input=6,n_output_unit=1,n_hidden_layer=2,n_hidden_unit=4,activation_function='relu',output_activation_function='sigmoid',loss_function='mean_squared_error')


#-----------------TRAIN-----------------
df = pd.read_csv('neural_network\\dataset\\monks-1.train', sep=" ", header=None)
df.drop(columns=[df.columns[-1]], inplace=True)
x = df.iloc[:, 2:8].values # tutte le colonne tranne la prima
y = df.iloc[:, 1].values   # la prima colonna

y=y.reshape(-1,1)

nn.train(x, y, learning_rate=0.01, epochs=100000)



#-----------------TEST-----------------
dt = pd.read_csv('neural_network\\dataset\\monks-1.test', sep=" ", header=None)
dt.drop(columns=[dt.columns[-1]], inplace=True)
xTest = dt.iloc[:, 2:8].values # tutte le colonne tranne la prima
yTest = dt.iloc[:, 1].values   # la prima colonna


yTest=yTest.reshape(-1,1)

nn.test(xTest, yTest)


