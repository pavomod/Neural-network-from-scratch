import matplotlib.pyplot as plt



def plot_loss_curve(training_loss, validation_loss, training_accuracy=0, validation_accuracy=0, retrain=False):
    # Calculate the number of epochs based on the length of the training loss array
    epochs = range(1, len(training_loss) + 1)

    # Setting the size of the plot
    plt.figure(figsize=(10, 6))

    # Plotting the training loss
    plt.plot(epochs, training_loss, label='Training Loss')

    # Plotting the validation loss if it's not a retraining phase
    if not retrain:
        plt.plot(epochs, validation_loss, label='Validation Loss', color='red')

    # Setting the title and labels for the plot
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Displaying the legend
    plt.legend()

    # Enabling the grid for better readability
    plt.grid(True)

    # Displaying the plot
    plt.show()



def plot_accuracy_curve(tr_accuracy_history, vl_accuracy_history):
    # Calculate the number of epochs based on the length of the training accuracy array
    epochs = range(1, len(tr_accuracy_history) + 1)

    # Setting the size of the plot
    plt.figure(figsize=(10, 6))

    # Plotting the training and validation accuracy
    plt.plot(epochs, tr_accuracy_history, label='Accuracy Training')
    plt.plot(epochs, vl_accuracy_history, label='Accuracy Validation', color='red')

    # Setting the title and labels for the plot
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Displaying the legend
    plt.legend()

    # Enabling the grid for better readability
    plt.grid(True)

    # Displaying the plot
    plt.show()
