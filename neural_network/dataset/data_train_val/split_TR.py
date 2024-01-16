import pandas as pd

# Carica il dataset da un file CSV
file_path = "neural_network\\dataset\\monks-1.test"
dataset = pd.read_csv(file_path)

# Imposta un seed per la riproducibilità della divisione
# Esegue la divisione del dataset in training set e validation set
train_set = dataset.sample(frac=0.8, random_state=44)
val_set = dataset.drop(train_set.index)

# Visualizza le dimensioni dei set di training e validation
print(f"Dimensioni del training set: {train_set.shape}")
print(f"Dimensioni del validation set: {val_set.shape}")

# Salva i set in due file CSV separati
train_set.to_csv("neural_network\\dataset\\data_train_val\\train_set.csv", index=False)
val_set.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
