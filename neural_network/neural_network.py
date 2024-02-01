import numpy as np
from classes import Layer, LossFunction, LearningRateScheduler, EarlyStopping, plot_loss_curve, plot_accuracy_curve

class NeuralNetwork:
    def __init__(self, settings):
        self.settings = settings
        self.n_input = settings['model']['input_size']          # Number of input features
        self.layers = []                                        # List to store layers of the network
        seed = np.random.randint(0, 2**31 - 1)                  # Generate a random seed

        # Initialize layers as per the settings
        for i in range(len(settings['model']['layers'])):
            if settings['model']['seed'] != -1:
                seed = settings['model']['seed']                # Use specified seed if provided
            else:
                settings['model']['seed'] = seed                # Save the generated seed

            # Add layers to the network
            self.layers.append(Layer(
                settings['model']['layers'][i]['num_neurons'],
                settings['model']['layers'][i]['activation_function'],
                settings['model']['layers'][i]['initialization'],
                seed))

        # Initialize the loss function
        self.loss_function = LossFunction(settings['training']['loss_function'])
        # Training parameters
        self.epochs = settings['training']['epochs']
        self.batch_size = settings['training']['batch_size']
        self.print_every = settings['training']['print_every']
        self.print_loss = settings['training']['print_loss']

        # Initialize learning rate scheduler
        self.learning_rate = LearningRateScheduler(
            settings['training']['learning_rate'],
            self.epochs,
            settings['training']['learning_rate_schedule']['approach'],
            settings['training']['learning_rate_schedule']['params'])

        # Regularization and momentum parameters
        self.regularization_lambda = settings['training']['regularization_lambda']
        self.momentum = settings['training']['momentum']

        # Initialize early stopping mechanism
        self.early_stopping = EarlyStopping(
            settings['training']['early_stopping']['enabled'],
            settings['training']['early_stopping']['patience'],
            settings['training']['early_stopping']['min_delta'])

        # Internal variables for network operation
        self.weights = []               # Weights for each layer
        self.bias = []                  # Biases for each layer
        self.grad_weights = []          # Weight gradients for each layer
        self.grad_bias = []             # Bias gradients for each layer
        self.deltas = []                # Delta values for each layer
        self.not_activated_output = []  # Outputs before activation function
        self.activated_output = []      # Outputs after activation function
        self.loss_history = []          # History of training loss values

        # Velocity terms for momentum-based optimization
        self.velocity_weights = []       # Velocity of weights for each layer
        self.velocity_bias = []          # Velocity of biases for each layer

        # Additional variables for tracking validation loss and accuracy
        self.val_loss_history = []       # Validation loss history
        self.tr_accuracy_history = []    # Training accuracy history
        self.vl_accuracy_history = []    # Validation accuracy history

        # Print seed if print_loss flag is true
        if self.print_loss:
            print("Seed: ", seed)

        # Network initialization
        self.__networkInitialization()

    


    def __networkInitialization(self):
        # Initializing weights for the first hidden layer
        first_layer = self.layers[0]
        self.weights.append(first_layer.initialization((self.n_input, first_layer.num_neurons)))

        # Initializing weights for the remaining hidden layers
        for i in range(1, len(self.layers) - 1):
            self.weights.append(self.layers[i].initialization((self.layers[i - 1].num_neurons, self.layers[i].num_neurons)))

        # Initializing weights for the output layer
        self.weights.append(self.layers[-1].initialization((self.layers[-2].num_neurons, self.layers[-1].num_neurons)))

        # Initializing biases for all layers
        for i in range(len(self.layers)):
            self.bias.append(np.zeros((1, self.layers[i].num_neurons)))

        # Initializing velocity terms for momentum-based optimization
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_bias = [np.zeros_like(b) for b in self.bias]


    def forward(self, input_data):
        self.not_activated_output = []      # Stores the input to each layer before activation
        self.activated_output = []          # Stores the output from each layer after activation

        # Initial input
        self.activated_output.append(input_data)

        # Forward propagation through hidden layers
        for i in range(len(self.layers) - 1):
            layer_input = self.activated_output[-1]
            z = np.dot(layer_input, self.weights[i]) + self.bias[i]     # Linear combination
            a = self.layers[i].activation_function.function(z)          # Activation

            self.not_activated_output.append(z)
            self.activated_output.append(a)

        # Forward propagation through the output layer
        layer_input = self.activated_output[-1]
        z = np.dot(layer_input, self.weights[-1]) + self.bias[-1]       # Linear combination for output layer
        a = self.layers[-1].activation_function.function(z)             # Activation for output layer

        self.not_activated_output.append(z)
        self.activated_output.append(a)

        return self.activated_output[-1]


    def backpropagation(self, y_true):
        self.deltas = [None] * len(self.layers)
        self.grad_weights = [None] * len(self.layers)
        self.grad_bias = [None] * len(self.layers)

        # Calculate delta for the output layer
        self.deltas[-1] = (self.loss_function.derivative(y_true, self.activated_output[-1]) *
                           self.layers[-1].activation_function.derivative(self.not_activated_output[-1]))
        self.grad_weights[-1] = np.dot(self.activated_output[-2].T, self.deltas[-1])
        self.grad_bias[-1] = np.sum(self.deltas[-1], axis=0, keepdims=True)

        # Calculate delta for the hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            self.deltas[i] = (np.dot(self.deltas[i + 1], self.weights[i + 1].T) *
                              self.layers[i].activation_function.derivative(self.not_activated_output[i]))
            self.grad_weights[i] = np.dot(self.activated_output[i].T, self.deltas[i])
            self.grad_bias[i] = np.sum(self.deltas[i], axis=0, keepdims=True)


    def update(self, epoch):
        for i in range(len(self.weights)):
            # Clip gradients to avoid exploding gradients
            self.grad_weights[i] = np.clip(self.grad_weights[i], -1, 1)
            self.grad_bias[i] = np.clip(self.grad_bias[i], -1, 1)

            # Update the velocity of weights and biases for momentum-based optimization
            self.velocity_weights[i] = (self.momentum * self.velocity_weights[i] +
                                        self.learning_rate.update_func(epoch) *
                                        (self.grad_weights[i] + self.regularization_lambda * self.weights[i]))
            self.velocity_bias[i] = (self.momentum * self.velocity_bias[i] +
                                     self.learning_rate.update_func(epoch) * self.grad_bias[i])

            # Update weights and biases
            self.weights[i] -= self.velocity_weights[i]
            self.bias[i] -= self.velocity_bias[i]


    def train(self, input_data, target, val_input, val_target, retrain=False):
        # Initialize loss and accuracy histories
        self.loss_history = []
        self.val_loss_history = []
        self.tr_accuracy_history = []
        self.vl_accuracy_history = []

        # Reset early stopping mechanism
        self.early_stopping.reset()

        # Calculate the number of mini-batches
        n_samples = input_data.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))

        for epoch in range(self.epochs):
            # Shuffle the data
            permutation = np.random.permutation(n_samples)
            input_data_shuffled = input_data[permutation]
            target_shuffled = target[permutation]

            # Iterate over mini-batches
            for batch in range(n_batches):
                start = batch * self.batch_size
                end = min(start + self.batch_size, n_samples)
                batch_input = input_data_shuffled[start:end]
                batch_target = target_shuffled[start:end]

                # Forward propagation, backpropagation, and update for the mini-batch
                self.forward(batch_input)
                self.backpropagation(batch_target)
                self.update(epoch)

            # Calculate training loss and accuracy
            output_predict = self.predict(input_data)
            loss, training_accuracy = self.performance(output_predict, target)
            self.loss_history.append(loss)

            # Calculate validation loss and accuracy
            if not retrain:
                val_output = self.predict(val_input)
                performance_loss, validation_accuracy = self.performance(val_output, val_target)
            else:
                # Calculate retraining loss and accuracy
                tr_output = self.predict(input_data)
                performance_loss, validation_accuracy = self.performance(tr_output, target)

            self.val_loss_history.append(performance_loss)
            self.tr_accuracy_history.append(training_accuracy)
            self.vl_accuracy_history.append(validation_accuracy)

            # Print training and validation loss every 'print_every' epochs
            if self.print_loss and epoch % self.print_every == 0 and not retrain:
                print(f"( Epoch {epoch} ) training loss: {loss}\t validation loss: {performance_loss}")
            if self.print_loss and epoch == self.epochs - 1 and not retrain:
                print(f"( Epoch {epoch} ) training loss: {loss}\t validation loss: {performance_loss}")

            # Check for early stopping
            if self.early_stopping(performance_loss):
                break

        # Plot loss curve if enabled
        if self.print_loss:
            self.plot_loss_curve(training_accuracy, validation_accuracy, retrain)

        return self.loss_history, self.val_loss_history, self.tr_accuracy_history, self.vl_accuracy_history
            

    def predict(self, input_data):
        return self.forward(input_data)


    def performance(self, output_predict, output_target):
        loss = self.loss_function.function(output_target, output_predict)
        accuracy = self.accuracy(output_target, output_predict)
        return loss, accuracy


    def accuracy(self, y_test, y_pred):
        if self.layers[-1].activation_function.name == 'sigmoid':
            y_pred = np.where(y_pred > 0.5, 1, 0)
            return round(np.sum(y_test == y_pred) / len(y_test), 3) * 100
        else:
            return LossFunction.mean_euclidean_distance(y_test, y_pred)


    def test(self, input_data, target):
        output = self.predict(input_data)
        loss_test, accuracy_test = self.performance(output, target)
        print("\n\n-------------------- TEST --------------------")
        print(f"Loss:\t{loss_test}")
        print(f"Accuracy:\t{accuracy_test}%")


    def plot_loss_curve(self, training_accuracy, validation_accuracy, retrain=False):
        plot_loss_curve(self.loss_history, self.val_loss_history, training_accuracy, validation_accuracy, retrain)
        plot_accuracy_curve(self.tr_accuracy_history, self.vl_accuracy_history)


    def get_params(self):
        weights_list = [weight.tolist() for weight in self.weights]
        bias_list = [bias.tolist() for bias in self.bias]
        params = {
            "settings": self.settings,
            "network_configuration": {
                "weights": weights_list,
                "bias": bias_list
            }
        }
        return params


    def printNetwork(self):
        print("Weights:")
        for i in range(len(self.weights)):
            print(f"Layer {i + 1}:")
            print(self.weights[i])
        print("Bias:")
        for i in range(len(self.bias)):
            print(f"Layer {i + 1}:")
            print(self.bias[i])