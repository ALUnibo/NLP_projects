from file_reader import import_features, import_labels
from dataframe_modifier import modify_stance, create_third_level_labels
from transformers import AutoTokenizer
from CustomDataset import CustomDataset
import torch
from torch.utils.data import DataLoader
from Classifier import ClassifierC, ClassifierCP, ClassifierCPS
from network_trainer import train, evaluate, get_predictions, evaluate_model
from baselines import random_uniform_classifier, majority_classifier, one_baseline
from metrics import calculate_f1_score
import random
import numpy as np


model_id = 1
models_dict = [{'name': 'bert-base-uncased', 'head_size': 768},
               {'name': 'roberta-base', 'head_size': 768},
               ]
seed = 43


if __name__ == '__main__':
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    # model = ClassifierC(**models_dict[model_id])
    # model = ClassifierCP(**models_dict[model_id])
    model = ClassifierCPS(**models_dict[model_id])
    # best_val_model, last_model = train(model, training_loader, validation_loader, 10, test_loader)

    # Test results
    print('Best model:')
    # evaluate_model(best_val_model, test_loader, 'cuda')
    print('Last model:')
    # evaluate_model(last_model, test_loader, 'cuda')

    # Baselines
    # 1. Random
    labels_random = random_uniform_classifier(third_level_test_dataframe)

    # 2. Majority
    labels_majority = majority_classifier(third_level_train_dataframe, third_level_test_dataframe)

    # 3. 1-baseline
    labels_one = one_baseline(third_level_test_dataframe)

    # Print scores of baselines
    print('Random baseline:')
    print('F1 score: ', calculate_f1_score(labels_random, torch.Tensor(third_level_test_dataframe.values)))
    print('Majority baseline:')
    print('F1 score: ', calculate_f1_score(labels_majority, torch.Tensor(third_level_test_dataframe.values)))
    print('One baseline:')
    print('F1 score: ', calculate_f1_score(labels_one, torch.Tensor(third_level_test_dataframe.values)))
