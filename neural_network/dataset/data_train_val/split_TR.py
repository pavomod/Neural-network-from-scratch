import pandas as pd


# divide il dataset in training set e validation set
def simple_splitter(dim_training_set, isCup, name_monks):
    # PATH
    file_path = "neural_network\\dataset\\"
    test_path="neural_network\\dataset\\"
    if isCup:
        file_path += "ML-CUP23-TR.csv"
        test_path += "ML-CUP23-TS.csv"
    else:
        file_path += name_monks+".train"
        test_path += name_monks+".test"

    # Legge il dataset
    dataset = pd.read_csv(file_path)
    test_set = pd.read_csv(test_path)

    # Divide il dataset in training set e validation set
    train_set = dataset.sample(frac=dim_training_set)
    val_set = dataset.drop(train_set.index)

    # Salva i set in due file CSV separati
    train_set.to_csv("neural_network\\dataset\\data_train_val\\training_set.csv", index=False)
    val_set.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
    test_set.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)
    dataset.to_csv("neural_network\\dataset\\data_train_val\\retrain_set.csv", index=False)


# divide il dataset in k-fold
def k_fold_splitter(k_folds=4, isCup=False, name_monks="monks-1"):
    if k_folds < 2:
        raise ValueError("k_folds deve essere maggiore o uguale a 2")
    
    # PATH
    file_path = "neural_network\\dataset\\"
    test_path="neural_network\\dataset\\"
    skiprows=0
    if isCup:
        file_path += "ML-CUP23-TR.csv"
        test_path += "ML-CUP23-TS.csv"
        skiprows=7
    else:
        file_path += name_monks+".train"
        test_path += name_monks+".test"
        skiprows=0

    # Legge il dataset
    dataset = pd.read_csv(file_path, skiprows=skiprows)
    test_set = pd.read_csv(test_path, skiprows=skiprows)

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