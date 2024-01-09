from file_reader import import_features, import_labels
from dataframe_modifier import modify_stance, create_third_level_labels
from transformers import AutoTokenizer
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from BertClassifier import BertClassifierC, BertClassifierCP, BertClassifierCPS
from network_trainer import train


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
    test_set = CustomDataset(test_dataframe, third_level_test_dataframe, tokenizer)

    c_dl_train = DataLoader(training_set, batch_size=8, shuffle=True)
    c_dl_validation = DataLoader(validation_set, batch_size=len(validation_set), shuffle=True)

    model = BertClassifierC(4)
    # model = BertClassifierCP(4)
    # model = BertClassifierCPS(4)
    train(model, c_dl_train, c_dl_validation, 1)




