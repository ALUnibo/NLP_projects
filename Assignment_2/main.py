from file_reader import import_features, import_labels
from dataframe_modifier import modify_stance, create_third_level_labels
from transformers import AutoTokenizer
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from BertClassifier import BertClassifierC, BertClassifierCP, BertClassifierCPS
from network_trainer import train
from models_generator import random_uniform_classifier, majority_classifier


model_id = 0
models_dict = [{'name': 'bert-base-uncased', 'head_size': 768},
               {'name': 'roberta-base', 'head_size': 768},
               ]


if __name__ == '__main__':
    train_dataframe, validation_dataframe, test_dataframe = import_features()
    lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe = import_labels()
    modify_stance(train_dataframe, validation_dataframe, test_dataframe)

    third_level_train_dataframe, third_level_validation_dataframe, third_level_test_dataframe = \
        create_third_level_labels(lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe)

    tokenizer = AutoTokenizer.from_pretrained(models_dict[model_id]['name'])
    # tokenizer.model_max_length = 512

    # Generate datasets
    training_set = CustomDataset(train_dataframe, third_level_train_dataframe, tokenizer)
    validation_set = CustomDataset(validation_dataframe, third_level_validation_dataframe, tokenizer)
    test_set = CustomDataset(test_dataframe, third_level_test_dataframe, tokenizer)

    training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=16, shuffle=False)

    # model = BertClassifierC(**models_dict[model_id])
    # model = BertClassifierCP(**models_dict[model_id])
    model = BertClassifierCPS(**models_dict[model_id])
    train(model, training_loader, validation_loader, 5)

    # Baselines
    # 1. Random
    # labels_random = random_uniform_classifier(validation_set)

    # 2. Majority




