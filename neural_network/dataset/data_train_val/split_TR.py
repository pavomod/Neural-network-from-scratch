import pandas as pd



DATASET_NAME = "monks-1"
DIM_TRAINING_SET = 0.8



def simple_splitter():
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
    # print(f"Dimensioni del training set: {train_set.shape}")
    # print(f"Dimensioni del validation set: {val_set.shape}")

    # Salva i set in due file CSV separati
    train_set.to_csv("neural_network\\dataset\\data_train_val\\training_set.csv", index=False)
    val_set.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
    test_set.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)
    dataset.to_csv("neural_network\\dataset\\data_train_val\\retrain_set.csv", index=False)
    
    
def k_fold_splitter(k_folds=4):

    if k_folds < 2:
        raise ValueError("k_folds deve essere maggiore o uguale a 2")
    
    # PATH
    file_path = f"neural_network\\dataset\\{DATASET_NAME}.train"
    test_path = f"neural_network\\dataset\\"+DATASET_NAME+".test"
    # Legge il dataset
    dataset = pd.read_csv(file_path)
    test_set = pd.read_csv(test_path)
    # Mescola il dataset
    shuffled_dataset = dataset.sample(frac=1).reset_index(drop=True)

    # Calcola la dimensione di ciascun fold
    fold_size = len(shuffled_dataset) // k_folds

    # Divide il dataset in K fold
    for fold in range(k_folds):
        start = fold * fold_size
        end = start + fold_size if fold != k_folds - 1 else len(shuffled_dataset)

        # Crea il validation set per l'attuale fold
        val_set = shuffled_dataset.iloc[start:end]

        # Crea il training set escludendo il validation set dell'attuale fold
        train_set = shuffled_dataset.drop(shuffled_dataset.index[start:end])

        # Salva i set in due file CSV separati
        train_set.to_csv(f"neural_network\\dataset\\data_train_val\\k_fold\\training_set_fold{fold+1}.csv", index=False)
        val_set.to_csv(f"neural_network\\dataset\\data_train_val\\k_fold\\validation_set_fold{fold+1}.csv", index=False)
    
    test_set.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)
        # print(f"Fold {fold+1}:")
        # print(f"Dimensioni del training set: {train_set.shape}")
        # print(f"Dimensioni del validation set: {val_set.shape}")
        
        
def simple_splitter_cup():
    # PATH
    file_path = "neural_network\\dataset\\ML-CUP23-TR.csv"
    test_path = "neural_network\\dataset\\ML-CUP23-TS.csv"

    # Legge il dataset e lo divide in training set e validation set
    training_set = pd.read_csv(file_path,skiprows=7)
    test_set = pd.read_csv(test_path, skiprows=7)
    


    # Divide il training_Set in training set e validation set
    train_set = training_set.sample(frac=DIM_TRAINING_SET)
    val_set = training_set.drop(train_set.index)

    # Stampa le dimensioni dei set
    # print(f"Dimensioni del training set: {train_set.shape}")
    # print(f"Dimensioni del validation set: {val_set.shape}")

    # Salva i set in due file CSV separati
    train_set.to_csv("neural_network\\dataset\\data_train_val\\training_set.csv", index=False)
    val_set.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
    test_set.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)
    training_set.to_csv("neural_network\\dataset\\data_train_val\\retrain_set.csv", index=False)
    
    
def k_fold_cup(k_folds=4):
    
    if k_folds < 2:
        raise ValueError("k_folds deve essere maggiore o uguale a 2")
    
    # PATH
    file_path = "neural_network\\dataset\\ML-CUP23-TR.csv"
    test_path = "neural_network\\dataset\\ML-CUP23-TS.csv"

    # Legge il dataset e lo divide in training set e validation set
    training_set = pd.read_csv(file_path,skiprows=7)
    test_set = pd.read_csv(test_path, skiprows=7)
    
    # Mescola il dataset
    shuffled_dataset = training_set.sample(frac=1).reset_index(drop=True)

    # Calcola la dimensione di ciascun fold
    fold_size = len(shuffled_dataset) // k_folds

    # Divide il dataset in K fold
    for fold in range(k_folds):
        start = fold * fold_size
        end = start + fold_size if fold != k_folds - 1 else len(shuffled_dataset)

        # Crea il validation set per l'attuale fold
        val_set = shuffled_dataset.iloc[start:end]

        # Crea il training set escludendo il validation set dell'attuale fold
        train_set = shuffled_dataset.drop(shuffled_dataset.index[start:end])

        # Salva i set in due file CSV separati
        train_set.to_csv(f"neural_network\\dataset\\data_train_val\\k_fold\\training_set_fold{fold+1}.csv", index=False)
        val_set.to_csv(f"neural_network\\dataset\\data_train_val\\k_fold\\validation_set_fold{fold+1}.csv", index=False)
    
    test_set.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)
        # print(f"Fold {fold+1}:")
        # print(f"Dimensioni del training set: {train_set.shape}")
        # print(f"Dimensioni del validation set: {val_set.shape}")
        
        
simple_splitter_cup()        
        
