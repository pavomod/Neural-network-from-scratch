import pandas as pd

DATASET_NAME = "monks-1"
DIM_TRAINING_SET = 0.8

# PATH
file_path = "neural_network\\dataset\\"+DATASET_NAME+".train"
test_path="neural_network\\dataset\\"+DATASET_NAME+".test"

# Legge il dataset e lo divide in training set e validation set
dataset = pd.read_csv(file_path)
test_set = pd.read_csv(test_path)


# Divide il dataset in training set e validation set
train_set = dataset.sample(frac=DIM_TRAINING_SET)
val_set = dataset.drop(train_set.index)

# Stampa le dimensioni dei set
print(f"Dimensioni del training set: {train_set.shape}")
print(f"Dimensioni del validation set: {val_set.shape}")

# Salva i set in due file CSV separati
train_set.to_csv("neural_network\\dataset\\data_train_val\\training_set.csv", index=False)
val_set.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
test_set.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)
dataset.to_csv("neural_network\\dataset\\data_train_val\\retrain_set.csv", index=False)