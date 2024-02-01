class EarlyStopping:
    def __init__(self, active=True, patience=5, min_delta=0):
        self.active = active            # Determines if early stopping is to be used
        self.patience = patience        # Number of epochs to wait for improvement before stopping
        self.min_delta = min_delta      # Minimum change in the monitored quantity to qualify as an improvement
        self.counter = 0                # Counts the number of epochs since the last improvement
        self.best_loss = None           # Stores the best loss observed
        self.early_stop = False         # Indicates whether early stopping is triggered

    def __call__(self, val_loss):
        # Method called at the end of each epoch to check if early stopping criteria are met
        if self.active:
            if self.best_loss is None:
                # If this is the first epoch, set the current loss as the best loss
                self.best_loss = val_loss
            elif self.best_loss - val_loss > self.min_delta:
                # If the loss has improved more than min_delta, update best_loss and reset counter
                self.best_loss = val_loss
                self.counter = 0
            else:
                # If the loss hasn't improved, increment the counter
                self.counter += 1
                if self.counter >= self.patience:
                    # If the counter reaches the patience threshold, trigger early stopping
                    self.early_stop = True
                    
            return self.early_stop
        # Return False if early stopping is not active
        return False
    
    def reset(self):
        # Resets the early stopping parameters, useful for reusing the object for a new training process
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
