import pandas as pd
import numpy as np


def modify_stance(train_dataframe, validation_dataframe, test_dataframe):
    train_dataframe['Stance'].replace('against', 0, inplace=True)
    train_dataframe['Stance'].replace('in favor of', 1, inplace=True)

    validation_dataframe['Stance'].replace('against', 0, inplace=True)
    validation_dataframe['Stance'].replace('in favor of', 1, inplace=True)

    test_dataframe['Stance'].replace('against', 0, inplace=True)
    test_dataframe['Stance'].replace('in favor of', 1, inplace=True)

    train_dataframe.drop(columns=['Argument ID'], inplace=True)
    validation_dataframe.drop(columns=['Argument ID'], inplace=True)
    test_dataframe.drop(columns=['Argument ID'], inplace=True)


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
