import pandas as pd

# Carica il dataset da un file CSV
file_path = "../monks-1.test"
dataset = pd.read_csv(file_path, skiprows=7, index_col=0)

# Dividi il dataset in training set e validation set
# random_state è un seed per rendere la divisione riproducibile
train_set, val_set = pd.train_test_split(dataset, test_size=0.2, random_state=42)

# Salva il training set su un file CSV
train_set.to_csv("train_set.csv", index=False)

# Salva il validation set su un file CSV
val_set.to_csv("val_set.csv", index=False)