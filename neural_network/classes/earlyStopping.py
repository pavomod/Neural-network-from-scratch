class EarlyStopping:
    def __init__(self, active=True,  patience=5, min_delta=0):
        """
        Inizializza l'early stopping.

        :param patience: Numero di epoche da attendere dopo l'ultima volta che il validation loss è migliorato.
        :param min_delta: Variazione minima del validation loss da considerare come un miglioramento.
        """
        self.active = active
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        

    
    def __call__(self, val_loss):
        if self.active:
            if self.best_loss is None:
                self.best_loss = val_loss
            elif self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    
            return self.early_stop
        return False
    
    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False