import pandas as pd


# divide il dataset in training set e validation set
def simple_splitter(dim_training_set, isCup, name_monks):
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

    # Divide il dataset in training set e validation set
    train_set = dataset.sample(frac=dim_training_set)
    val_set = dataset.drop(train_set.index)

    # Salva i set in due file CSV separati
    train_set.to_csv("neural_network\\dataset\\data_train_val\\training_set.csv", index=False)
    val_set.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
    test_set.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)
    dataset.to_csv("neural_network\\dataset\\data_train_val\\retrain_set.csv", index=False)


def splitter_tr_vl_ts():
    # Legge il dataset
    file_path = "neural_network\\dataset\\ML-CUP23-TR.csv"
    dataset = pd.read_csv(file_path, skiprows=7)
    
    # Mescola il dataset
    dataset_shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calcola gli indici per il training (70%), validation (20%) e test (10%)
    total_rows = len(dataset_shuffled)
    train_end = int(total_rows * 0.7)
    validation_end = int(total_rows * 0.9)  # 70% + 20% = 90%
    
    # Divide il dataset
    train_dataset = dataset_shuffled.iloc[:train_end]
    validation_dataset = dataset_shuffled.iloc[train_end:validation_end]
    test_dataset = dataset_shuffled.iloc[validation_end:]
    
    train_dataset.to_csv("neural_network\\dataset\\data_train_val\\training_set.csv", index=False)
    validation_dataset.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
    test_dataset.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)

def k_fold_splitter(k_folds=4, hold_out_fraction=0.1, isCup=False, name_monks="monks-1"):
    if k_folds < 2:
        raise ValueError("k_folds deve essere maggiore o uguale a 2")
    
    # PATH
    file_path = "neural_network\\dataset\\"
    skiprows=0
    if isCup:
        file_path += "ML-CUP23-TR.csv"
        skiprows=7
    else:
        file_path += name_monks+".train"

    # Legge il dataset
    dataset = pd.read_csv(file_path, skiprows=skiprows)

    # Divide il dataset in hold out set e il resto
    
    hold_out_set = dataset.sample(frac=hold_out_fraction)
    remaining_set = dataset.drop(hold_out_set.index)

    # Mescola il dataset rimanente
    shuffled_dataset = remaining_set.sample(frac=1).reset_index(drop=True)
    
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
    
    # Salva il hold out set e il test set
    
    hold_out_set.to_csv("neural_network\\dataset\\data_train_val\\k_fold\\hold_out.csv", index=False)
    


# simple_splitter(0.8, True, "monks-1")
