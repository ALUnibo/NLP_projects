import pandas as pd


def import_features():
    train_dataframe = pd.read_csv('arguments-training.tsv', sep='\t')
    validation_dataframe = pd.read_csv('arguments-validation.tsv', sep='\t')
    test_dataframe = pd.read_csv('arguments-test.tsv', sep='\t')

    return train_dataframe, validation_dataframe, test_dataframe


def import_labels():
    lab_train_dataframe = pd.read_csv('labels-training.tsv', sep='\t')
    lab_validation_dataframe = pd.read_csv('labels-validation.tsv', sep='\t')
    lab_test_dataframe = pd.read_csv('labels-test.tsv', sep='\t')
    return lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe
