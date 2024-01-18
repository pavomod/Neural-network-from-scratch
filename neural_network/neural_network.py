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
        self.grad_weights=[]
        self.grad_bias=[]
        self.deltas=[]
        self.bias=[]
        self.not_activated_output=[] #! output prima di avere applicato la funzione di attivazione
        self.activated_output=[] #! output dopo avere applicato la funzione di attivazione
        self.initializer=Initializer()
        self.__networkInitilization()
        self.activation_function=ActivationFunction(activation_function)
        self.loss_function=LossFunction(loss_function)
        
    def __networkInitilization(self):
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
        self.not_activated_output=[]      
        self.activated_output=[]
        
        self.not_activated_output.append(input_data)
        self.activated_output.append(input_data)

        for i in range(self.n_hidden_layer): #! attenzionare la lunghezza
            self.not_activated_output.append(np.dot(self.activated_output[i],self.weights[i]) + self.bias[i])
            self.activated_output.append(self.activation_function.function(self.not_activated_output[i+1]))  

        return self.activated_output[-1]

    def backpropagation(self, target, learning_rate=0.01,lambda_reg=0.01): #lambda_reg -> tichonov regularization
        # Calcolo dell'errore
        self.grad_weights=[]
        self.grad_bias=[]
        self.deltas=[]
        
        self.deltas.append(self.activated_output[-1] - target)
        
        self.grad_weights.append(np.dot(self.activated_output[-2].T, self.deltas[-1])) 
        self.grad_bias.append(np.sum(self.deltas[-1], axis=0, keepdims=True))
        
        for i in range(self.n_hidden_layer-1,0,-1):
            self.deltas.append(np.dot(self.deltas[-1], self.weights[i].T) * self.activation_function.derivative(self.not_activated_output[i]))
            self.grad_weights.append(np.dot(self.activated_output[i-1].T, self.deltas[-1]))
            self.grad_bias.append(np.sum(self.deltas[-1], axis=0, keepdims=True))
        
        self.grad_weights.reverse()
        self.grad_bias.reverse()
        self.deltas.reverse()
        
        for i in range(self.n_hidden_layer):
            self.weights[i] -= learning_rate * (self.grad_weights[i]) # QUI SI REGOLARIZZA CON TICHONOV -> self.weights[i] -= learning_rate * (self.grad_weights[i] + lambda_reg * self.weights[i])
            self.bias[i] -= learning_rate * (self.grad_bias[i])
            
    def train(self, input_data, target, learning_rate=0.01, epochs=124):
        for i in range(epochs):
            self.forward(input_data)
            self.backpropagation(target, learning_rate)
            #if i % 100 == 0:
                #print(f"Epoch {i}:")
        print("Training")
        print(self.loss_function.function(target,self.predict(input_data)))
                
    def predict(self, input_data):
        return self.forward(input_data)

    def test(self, test_input_data, test_target):
        print("Test")
        print(self.loss_function.function(test_target, self.predict(test_input_data)))
        
    
#self,n_hidden_unit=4,n_hidden_layer=1,n_input=10,n_output_unit=3,activation_function='sigmoid',loss_function='mean_squared_error'
            
nn = NeuralNetwork(n_input=6,n_output_unit=1,n_hidden_layer=2,n_hidden_unit=3,activation_function='sigmoid', loss_function='cross_entropy')

#-----------------TRAIN-----------------
df = pd.read_csv('neural_network\\dataset\\monks-1.train', sep=" ", header=None)
df.drop(columns=[df.columns[-1]], inplace=True)
x = df.iloc[:, 2:8].values # tutte le colonne tranne la prima
y = df.iloc[:, 1].values   # la prima colonna

y=y.reshape(-1,1)

nn.train(x, y, learning_rate=0.001, epochs=100000)
#-----------------TEST-----------------
dt = pd.read_csv('neural_network\\dataset\\monks-1.test', sep=" ", header=None)
dt.drop(columns=[dt.columns[-1]], inplace=True)
xTest = dt.iloc[:, 2:8].values # tutte le colonne tranne la prima
yTest = dt.iloc[:, 1].values   # la prima colonna


yTest=yTest.reshape(-1,1)

nn.test(xTest, yTest)


