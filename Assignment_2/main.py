import copy
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from file_reader import import_features, import_labels
from dataframe_modifier import modify_stance, create_third_level_labels
from metrics import calculate_f1_score
from baselines import random_uniform_classifier, majority_classifier, one_baseline
from CustomDataset import CustomDataset
from Classifier import ClassifierC, ClassifierCP, ClassifierCPS
from network_trainer import train, evaluate_model
from plots import generate_summary, generate_precision_recall_curve, generate_confusion_matrix, \
    generate_f1_scores_table, generate_bar_plot_with_f1_scores, generate_training_history_plots


save_best_models = False
load_best_models = False
models_load_path = 'best_models.tar'
models_save_path = 'best_models.tar'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_id = 1
models_dict = [{'name': 'bert-base-uncased', 'head_size': 768},
               {'name': 'roberta-base', 'head_size': 768},
               {'name': 'roberta-large', 'head_size': 1024}]

initializer_seed = 111
seeds = [569,
         106,
         999
         ]

num_epochs = 10


if __name__ == '__main__':
    random.seed(initializer_seed)
    np.random.seed(initializer_seed)
    torch.manual_seed(initializer_seed)

    train_dataframe, validation_dataframe, test_dataframe = import_features()
    lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe = import_labels()
    modify_stance(train_dataframe, validation_dataframe, test_dataframe)

    third_level_train_dataframe, third_level_validation_dataframe, third_level_test_dataframe = \
        create_third_level_labels(lab_train_dataframe, lab_validation_dataframe, lab_test_dataframe)

    # Baselines
    # 1. Random
    predictions_random = random_uniform_classifier(third_level_test_dataframe)
    val_predictions_random = random_uniform_classifier(third_level_validation_dataframe)

    # 2. Majority
    predictions_majority = majority_classifier(third_level_train_dataframe, third_level_test_dataframe)
    val_predictions_majority = majority_classifier(third_level_train_dataframe, third_level_validation_dataframe)

    # 3. 1-baseline
    predictions_one = one_baseline(third_level_test_dataframe)
    val_predictions_one = one_baseline(third_level_validation_dataframe)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(models_dict[model_id]['name'])

    # Generate datasets and dataloaders
    training_set = CustomDataset(train_dataframe, third_level_train_dataframe, tokenizer)
    validation_set = CustomDataset(validation_dataframe, third_level_validation_dataframe, tokenizer)
    test_set = CustomDataset(test_dataframe, third_level_test_dataframe, tokenizer)

    training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    # Initialize training data structures
    histories = {}
    best_models = {}
    models = {'C': ClassifierC,
              'CP': ClassifierCP,
              'CPS': ClassifierCPS
              }

    # Train models
    if not load_best_models:
        for seed in seeds:
            for model_type, model_class in models.items():
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                print('Set seed to: ', seed)

                model = model_class(**models_dict[model_id])
                model_trained, history = train(model, training_loader, validation_loader, num_epochs)
                if model_type not in histories:
                    histories[model_type] = history
                    best_models[model_type] = model_trained
                else:
                    if history['best_val_macro_f1'] > histories[model_type]['best_val_macro_f1']:
                        histories[model_type] = history
                        best_models[model_type] = model_trained

                # TODO: delete these two lines
                print('Test set results (TO REMOVE):')
                evaluate_model(model_trained, test_loader, device)

        if save_best_models:
            torch.save({
                'modelC_state_dict': best_models['C'].state_dict(),
                'modelCP_state_dict': best_models['CP'].state_dict(),
                'modelCPS_state_dict': best_models['CPS'].state_dict(),
                'historyC': histories['C'],
                'historyCP': histories['CP'],
                'historyCPS': histories['CPS']
            }, models_save_path)
            print('Models saved successfully to: ', models_save_path, '\n')
    else:
        checkpoint = torch.load(models_load_path)
        best_models['C'] = ClassifierC(**models_dict[model_id])
        best_models['C'].load_state_dict(checkpoint['modelC_state_dict'])
        best_models['CP'] = ClassifierCP(**models_dict[model_id])
        best_models['CP'].load_state_dict(checkpoint['modelCP_state_dict'])
        best_models['CPS'] = ClassifierCPS(**models_dict[model_id])
        best_models['CPS'].load_state_dict(checkpoint['modelCPS_state_dict'])
        histories['C'] = checkpoint['historyC']
        histories['CP'] = checkpoint['historyCP']
        histories['CPS'] = checkpoint['historyCPS']
        print('Models loaded successfully from: ', models_load_path, '\n')

    # Printing validation macro F1 scores
    val_macro_f1_scores = []
    # Baselines over validation set
    val_macro_f1_scores.append(
        ['random', calculate_f1_score(val_predictions_random, torch.Tensor(third_level_validation_dataframe.values))[0].item()])
    val_macro_f1_scores.append(
        ['majority', calculate_f1_score(val_predictions_majority, torch.Tensor(third_level_validation_dataframe.values))[0].item()])
    val_macro_f1_scores.append(
        ['one', calculate_f1_score(val_predictions_one, torch.Tensor(third_level_validation_dataframe.values))[0].item()])

    for model_type, history in histories.items():
        val_macro_f1_scores.append([model_type, history['best_val_macro_f1']])

    val_macro_f1_scores = pd.DataFrame(val_macro_f1_scores, columns=['Model', 'Macro F1 score'])
    print(val_macro_f1_scores)

    generate_training_history_plots(histories)

    print('Evaluating the best models on the test set...')
    outputs_dict = {'random': predictions_random, 'majority': predictions_majority, 'one': predictions_one}
    labels = third_level_test_dataframe.values
    crisp_predictions_dict = copy.deepcopy(outputs_dict)
    for model_type, model in best_models.items():
        print('Model type: ', model_type)
        _, _, _, outputs, labels_, crisp_predictions = evaluate_model(model, test_loader, device)
        assert np.array_equal(labels, labels_)
        outputs_dict[model_type] = outputs
        crisp_predictions_dict[model_type] = crisp_predictions

        generate_summary(crisp_predictions, labels)

    # F1 scores table
    generate_f1_scores_table(outputs_dict, labels, crisp_predictions_dict)
    # Precision-Recall curve
    generate_precision_recall_curve(outputs_dict, labels, crisp_predictions_dict)
    # Confusion matrix
    generate_confusion_matrix(outputs_dict, labels, crisp_predictions_dict)
    # Distribution imbalance bar plot with f1 scores
    generate_bar_plot_with_f1_scores(outputs_dict, labels, crisp_predictions_dict)

    # TODO: notebook
