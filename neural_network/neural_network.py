import numpy as np
from classes import Initializer, ActivationFunction, LossFunction
import pandas as pd

class NeuralNetwork:
    def __init__(self,n_hidden_unit=3,n_hidden_layer=1,n_input=10,n_output_unit=3,activation_function='sigmoid',loss_function='mean_squared_error'):
        self.n_input=n_input
        self.n_hidden_unit=n_hidden_unit
        self.n_output_unit=n_output_unit
        self.n_hidden_layer=n_hidden_layer
        self.weights=[]
        self.bias=[]
        self.output_history=[]
        self.initializer=Initializer()
        self.__networkInitilization()
        self.activation_function=ActivationFunction(activation_function)
        self.loss_function=LossFunction(loss_function)
        
    def __networkInitilization(self):
        self.weights.append(self.initializer.uniformInitializer(-0.5,0.5,(self.n_input,self.n_hidden_unit)))
        
        for _ in range(self.n_hidden_layer-1):
            self.weights.append(self.initializer.uniformInitializer(-0.5,0.5,(self.n_hidden_unit,self.n_hidden_unit)))
            
        self.weights.append(self.initializer.uniformInitializer(-0.5,0.5,(self.n_hidden_unit,self.n_output_unit)))
        
        
        self.bias.append(np.zeros((1,self.n_hidden_unit)))
        
        for _ in range(self.n_hidden_layer-1):
            self.bias.append(np.zeros((1,self.n_hidden_unit)))
            
        self.bias.append(np.zeros((1,self.n_output_unit)))
        
    def printNetwork(self):
        print("Weights:")
        for i in range(len(self.weights)):
            print(f"Layer {i+1}:")
            print(self.weights[i])
        print("Bias:")
        for i in range(len(self.bias)):
            print(f"Layer {i+1}:")
            print(self.bias[i])
            
    def forward(self, input_data):     
        current_output = input_data
        self.output_history=[current_output]
        for i in range(self.n_hidden_layer+1):
            current_output = np.dot(current_output, self.weights[i]) + self.bias[i]
            current_output = self.activation_function.function(current_output)  
            self.output_history.append(current_output)
        return current_output

    def backpropagation(self, target, learning_rate=0.01,lambda_reg=0.001): #lambda_reg -> tichonov regularization
        # Calcolo dell'errore
        
        error = self.loss_function.derivative(target, self.output_history[-1])
        # Backward pass
        for i in reversed(range(self.n_hidden_layer + 1)):
            # Calcolo del gradiente della funzione di attivazione
            d_activation = self.activation_function.derivative(self.output_history[i+1])
            # Calcolo del delta
            delta = error * d_activation
            # Calcolo del gradiente dei pesi e dei bias
            d_weights = np.dot(self.output_history[i].T, delta)  #+ lambda_reg * self.weights[i] #tikonov regularization
            d_bias = np.sum(delta, axis=0, keepdims=True)
            # Aggiornamento dei pesi e dei bias
            self.weights[i] -= learning_rate * d_weights
            self.bias[i] -= learning_rate * d_bias
            # Propagazione dell'errore al livello precedente
            if i != 0:
                error = np.dot(delta, self.weights[i].T)

    
    def train(self, input_data, target, learning_rate=0.01, epochs=124):
        for i in range(epochs):
            self.forward(input_data[i].reshape(1, -1))
            self.backpropagation(np.array([target[i]]), learning_rate)
            print(target[i], self.output_history[-1])
            if i % 10 == 0: #!in realtà stiamo facendo online learning
                print(f'Epoch {i} loss: {self.loss_function.function(np.array(target[i]), self.output_history[-1])}')
                
    
#self,n_hidden_unit=4,n_hidden_layer=1,n_input=10,n_output_unit=3,activation_function='sigmoid',loss_function='mean_squared_error'
            
nn = NeuralNetwork(n_input=6,n_output_unit=1)

df = pd.read_csv('neural_network\\classes\\monks-1.train', sep=" ", header=None)
df.drop(columns=[df.columns[-1]], inplace=True)
X = df.iloc[:, 1:-1].values # tutte le colonne tranne la prima
y = df.iloc[:, 1].values   # la prima colonna
nn.train(X, y, learning_rate=0.01, epochs=124)





