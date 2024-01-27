import matplotlib.pyplot as plt




def plot_loss_curve(training_loss,validation_loss,training_accuracy,validation_accuracy,retrain=False):
    """
    Plot a learning curve using the provided loss values.

    :param loss_values: Array of loss values, one for each epoch.
    """

    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_loss, label='Training Loss')
    if not retrain:
        plt.plot(epochs,validation_loss, label='Validation Loss', color='red')
    x_coord = max(epochs) * 0.4
    y_coord_train = max(training_loss) * 0.9
    y_coord_val = max(validation_loss) * 0.9

    plt.text(x_coord, y_coord_train, "Training accuracy: " + str(training_accuracy) + "%")
    if not retrain:
        plt.text(x_coord, y_coord_val, "Validation accuracy: " + str(validation_accuracy) + "%")

    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()