import matplotlib.pyplot as plt




def plot_loss_curve(loss_values):
    """
    Plot a learning curve using the provided loss values.

    :param loss_values: Array of loss values, one for each epoch.
    """

    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()