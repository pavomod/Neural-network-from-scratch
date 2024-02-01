import pandas as pd



def simple_splitter(dim_training_set, isCup, name_monks):
    # File paths for the dataset
    file_path = "neural_network\\dataset\\"
    test_path = "neural_network\\dataset\\"
    skiprows = 0

    # Path adjustments based on dataset type
    if isCup:
        file_path += "ML-CUP23-TR.csv"
        test_path += "ML-CUP23-TS.csv"
        skiprows = 7
    else:
        file_path += name_monks + ".train"
        test_path += name_monks + ".test"
        skiprows = 0

    # Reading the dataset
    dataset = pd.read_csv(file_path, skiprows=skiprows)
    test_set = pd.read_csv(test_path, skiprows=skiprows)

    # Splitting the dataset into training and validation sets
    train_set = dataset.sample(frac=dim_training_set)
    val_set = dataset.drop(train_set.index)

    # Saving the sets into separate CSV files
    train_set.to_csv("neural_network\\dataset\\data_train_val\\training_set.csv", index=False)
    val_set.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
    test_set.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)
    dataset.to_csv("neural_network\\dataset\\data_train_val\\retrain_set.csv", index=False)



def splitter_tr_vl_ts():
    # Reading the dataset
    file_path = "neural_network\\dataset\\ML-CUP23-TR.csv"
    dataset = pd.read_csv(file_path, skiprows=7)

    # Shuffling the dataset
    dataset_shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculating indices for splitting into training, validation, and test sets
    total_rows = len(dataset_shuffled)
    train_end = int(total_rows * 0.7)  # 70% for training
    validation_end = int(total_rows * 0.9)  # additional 20% for validation

    # Splitting the dataset
    train_dataset = dataset_shuffled.iloc[:train_end]
    validation_dataset = dataset_shuffled.iloc[train_end:validation_end]
    test_dataset = dataset_shuffled.iloc[validation_end:]

    # Saving the sets into separate CSV files
    dataset.to_csv("neural_network\\dataset\\data_train_val\\retrain_set.csv", index=False)
    train_dataset.to_csv("neural_network\\dataset\\data_train_val\\training_set.csv", index=False)
    validation_dataset.to_csv("neural_network\\dataset\\data_train_val\\validation_set.csv", index=False)
    test_dataset.to_csv("neural_network\\dataset\\data_train_val\\test_set.csv", index=False)



def k_fold_splitter(k_folds=4, hold_out_fraction=0.1, isCup=False, name_monks="monks-1"):
    if k_folds < 2:
        raise ValueError("k_folds deve essere maggiore o uguale a 2")

    # File path for the dataset
    file_path = "neural_network\\dataset\\"
    skiprows = 0

    # Path adjustments based on dataset type
    if isCup:
        file_path += "ML-CUP23-TR.csv"
        skiprows = 7
    else:
        file_path += name_monks + ".train"

    # Reading the dataset
    dataset = pd.read_csv(file_path, skiprows=skiprows)

    # Splitting the dataset into hold-out set and the rest
    hold_out_set = dataset.sample(frac=hold_out_fraction)
    remaining_set = dataset.drop(hold_out_set.index)

    # Shuffling the remaining dataset
    shuffled_dataset = remaining_set.sample(frac=1).reset_index(drop=True)

    # Calculating the size of each fold
    fold_size = len(shuffled_dataset) // k_folds

    # Splitting the dataset into K folds
    for fold in range(k_folds):
        start = fold * fold_size
        end = start + fold_size if fold != k_folds - 1 else len(shuffled_dataset)

        # Creating validation and training sets for the current fold
        val_set = shuffled_dataset.iloc[start:end]
        train_set = shuffled_dataset.drop(shuffled_dataset.index[start:end])

        # Saving the sets into separate CSV files
        train_set.to_csv(f"neural_network\\dataset\\data_train_val\\k_fold\\training_set_fold{fold+1}.csv", index=False)
        val_set.to_csv(f"neural_network\\dataset\\data_train_val\\k_fold\\validation_set_fold{fold+1}.csv", index=False)

    # Saving the hold-out set
    hold_out_set.to_csv("neural_network\\dataset\\data_train_val\\k_fold\\hold_out.csv", index=False)