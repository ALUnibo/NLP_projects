import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from BertClassifier import BertClassifier
from network_trainer import train
from torch.utils.data import Dataset, DataLoader
from CustomDataset import CustomDataset


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


def modify_stance(train_dataframe, validation_dataframe, test_dataframe):
    train_dataframe['Stance'].replace('against', .0, inplace=True)
    train_dataframe['Stance'].replace('in favor of', 1., inplace=True)

    validation_dataframe['Stance'].replace('against', .0, inplace=True)
    validation_dataframe['Stance'].replace('in favor of', 1., inplace=True)

    test_dataframe['Stance'].replace('against', .0, inplace=True)
    test_dataframe['Stance'].replace('in favor of', 1., inplace=True)


def create_third_level_labels(lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe):
    oc_cols = ['thought', 'action', 'stimulation', 'hedonism']
    st_cols = ['humility', 'caring', 'dependability', 'concern', 'nature', 'tollerance', 'objectivity']
    se_cols = ['hedonism', 'achievement', 'dominance', 'resources', 'face']
    cn_cols = ['humility', 'interpersonal', 'rules', 'tradition', 'societal', 'personal', 'face']

    funct = lambda y, z: list(filter(lambda x: any([i.lower() in x.lower() for i in z]), y.columns))

    third_level_cols = {'OC': oc_cols, 'ST': st_cols, 'SE': se_cols, 'CN': cn_cols}
    third_level_train_dataframe = pd.DataFrame()
    third_level_validation_dataframe = pd.DataFrame()
    third_level_test_dataframe = pd.DataFrame()

    third_level_train_dataframe['Argument ID'] = lab_train_dataframe['Argument ID']
    third_level_validation_dataframe['Argument ID'] = lab_validation_dataframe['Argument ID']
    third_level_test_dataframe['Argument ID'] = lab_test_dataframe['Argument ID']

    for k, v in third_level_cols.items():
        train_dataframe_reduce = lab_train_dataframe[funct(lab_train_dataframe, v)].apply(np.sum, axis=1) > 0
        third_level_train_dataframe[k] = train_dataframe_reduce.astype(np.float64)

        validation_dataframe_reduce = lab_validation_dataframe[funct(lab_validation_dataframe, v)].apply(np.sum,
                                                                                                         axis=1) > 0
        third_level_validation_dataframe[k] = validation_dataframe_reduce.astype(np.float64)

        test_dataframe_reduce = lab_test_dataframe[funct(lab_test_dataframe, v)].apply(np.sum, axis=1) > 0
        third_level_test_dataframe[k] = test_dataframe_reduce.astype(np.float64)

    third_level_train_dataframe.drop(columns=['Argument ID'], inplace=True)
    third_level_validation_dataframe.drop(columns=['Argument ID'], inplace=True)
    third_level_test_dataframe.drop(columns=['Argument ID'], inplace=True)

    return third_level_train_dataframe, third_level_validation_dataframe, third_level_test_dataframe


def generate_datasets(tokenizer, training_set, validation_set, test_set):
    pass


if __name__ == '__main__':
    train_dataframe, validation_dataframe, test_dataframe = import_features()
    lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe = import_labels()
    modify_stance(train_dataframe, validation_dataframe, test_dataframe)

    train_dataframe.drop(columns=['Argument ID'], inplace=True)
    validation_dataframe.drop(columns=['Argument ID'], inplace=True)
    test_dataframe.drop(columns=['Argument ID'], inplace=True)

    third_level_train_dataframe, third_level_validation_dataframe, third_level_test_dataframe = \
        create_third_level_labels(lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # bert-base-uncased
    tokenizer.model_max_length = 512

    # Generate datasets
    training_set = CustomDataset(train_dataframe, third_level_train_dataframe, tokenizer)
    validation_set = CustomDataset(validation_dataframe, third_level_validation_dataframe, tokenizer)

    c_dl_train = DataLoader(training_set, batch_size=8, shuffle=True)
    c_dl_validation = DataLoader(validation_set, batch_size=len(validation_set), shuffle=True)

    model = BertClassifier(4)
    train(model, c_dl_train, c_dl_validation, 100)
