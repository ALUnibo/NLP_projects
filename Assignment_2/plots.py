import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, confusion_matrix, \
    ConfusionMatrixDisplay
import torch

from network_trainer import calculate_f1_score

labels_columns = ['Openness to change', 'Self-transcendence', 'Self-enhancement', 'Conservation']


def generate_training_history_plots(histories):
    for model_type, history in histories.items():
        # Plot losses and macro F1 scores
        fig, ax1 = plt.subplots()
        ax1.set_title('Loss and macro F1 score of model ' + model_type)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_ylim([0.4, 0.7])
        ax1.plot(history['train_loss'], label='Train loss', color='blue')
        ax1.plot(history['val_loss'], label='Validation loss', color='orange')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Macro F1 score')
        ax2.set_ylim([0.55, 0.8])
        ax2.plot(history['val_macro_f1'], label='Validation macro F1 score', color='green')
        ax2.tick_params(axis='y')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig('plots/loss_and_macro_f1_score_' + model_type + '.png')
        plt.show()

        # Plot class F1 scores and macro F1 score
        fig, ax = plt.subplots()
        ax.set_title('Class F1 scores and macro F1 score of model ' + model_type)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 score')
        ax.plot(history['val_class_f1'][:, 0], label='Openness to change', color='blue')
        ax.plot(history['val_class_f1'][:, 1], label='Self-transcendence', color='orange')
        ax.plot(history['val_class_f1'][:, 2], label='Self-enhancement', color='red')
        ax.plot(history['val_class_f1'][:, 3], label='Conservation', color='violet')
        ax.plot(history['val_macro_f1'], label='Macro F1 score', color='green')
        ax.tick_params(axis='y')
        ax.legend(loc='lower left')

        fig.tight_layout()
        plt.savefig('plots/class_f1_scores_and_macro_f1_score_' + model_type + '.png')
        plt.show()


def generate_summary(crisp_predictions, labels, verbose=True):
    precision = []
    recall = []
    f1 = []
    for i in range(labels.shape[1]):
        precision.append(precision_score(labels[:, i], crisp_predictions[:, i]))
        recall.append(recall_score(labels[:, i], crisp_predictions[:, i]))
        f1_i = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        f1.append(f1_i)

        assert np.isclose(f1_i, calculate_f1_score(torch.Tensor(crisp_predictions), torch.Tensor(labels))[1][i].item())

    summary = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1': f1}, index=labels_columns)
    if verbose:
        print(summary)

    return summary


def generate_f1_scores_table(outputs_dict, labels, crisp_predictions_dict, verbose=True):
    f1_scores = []
    for model_type, predictions in crisp_predictions_dict.items():
        macro_f1_score, class_f1_score = calculate_f1_score(torch.Tensor(predictions), torch.Tensor(labels))
        macro_f1_score, class_f1_score = macro_f1_score.item(), class_f1_score.cpu().numpy()

        f1_scores.append([model_type, macro_f1_score, *class_f1_score])

    f1_scores = pd.DataFrame(f1_scores, columns=['Model', 'Macro F1 score', *labels_columns])
    if verbose:
        print(f1_scores)

    return f1_scores


def generate_precision_recall_curve(outputs_dict, labels, crisp_predictions_dict):
    plt.figure()
    plt.title('Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    for model_type in outputs_dict.keys():
        outputs = outputs_dict[model_type]
        precision, recall, _ = precision_recall_curve(labels.flatten(), outputs.flatten())
        plt.plot(recall, precision, label=model_type)

    plt.legend(loc='upper right')
    plt.savefig('plots/precision_recall_curve.png')
    plt.show()


def generate_confusion_matrix(outputs_dict, labels, crisp_predictions_dict):
    n_rows = len(labels_columns)
    n_cols = len(outputs_dict.keys())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    for i, model_type in enumerate(outputs_dict.keys()):
        crisp_predictions = crisp_predictions_dict[model_type]

        for j in range(labels.shape[1]):
            cm = confusion_matrix(labels[:, j], crisp_predictions[:, j])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot(ax=axes[j, i], xticks_rotation='horizontal')
            disp.ax_.set_title(model_type + ' - ' + labels_columns[j])
            disp.im_.colorbar.remove()

    fig.tight_layout(pad=5.0)
    plt.savefig('plots/confusion_matrix.png')
    plt.show()


def generate_bar_plot_with_f1_scores(outputs_dict, labels, crisp_predictions_dict):
    n_classes = labels.shape[1]
    n_labels = np.sum(labels, axis=0)
    markers = ['o', 'v', 's', 'P', 'X', 'D']

    fig, ax1 = plt.subplots(figsize=(n_classes * 3, 9))
    ax1.set_title('Distribution imbalance and F1 score')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Number of instances')
    ax1.bar(labels_columns, n_labels, color='#6aa0f7')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 score')
    for model_type in outputs_dict.keys():
        crisp_predictions = crisp_predictions_dict[model_type]
        _, class_f1_score = calculate_f1_score(torch.Tensor(crisp_predictions), torch.Tensor(labels))
        class_f1_score = class_f1_score.cpu().numpy()
        ax2.plot(labels_columns, class_f1_score, label=model_type, marker=markers.pop(0))
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig('plots/distribution_imbalance_and_f1_score.png')
    plt.show()


def generate_bar_plot(train_dataframe, validation_dataframe, test_dataframe):
    n_classes = train_dataframe.shape[1]
    n_train_labels = np.sum(train_dataframe.values, axis=0)
    n_validation_labels = np.sum(validation_dataframe.values, axis=0)
    n_test_labels = np.sum(test_dataframe.values, axis=0)

    ind = np.arange(n_classes)
    width = 0.3

    fig, ax = plt.subplots(figsize=(n_classes * 3, 9))
    ax.set_title('Distribution imbalance')
    ax.set_xlabel('Label')
    ax.set_ylabel('Number of instances')
    ax.bar(ind - width, n_train_labels, width, label='Train', color='#6aa0f7')
    ax.bar(ind, n_validation_labels, width, label='Validation', color='#f76a6a')
    ax.bar(ind + width, n_test_labels, width, label='Test', color='#6af79f')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels_columns)
    ax.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig('plots/distribution_imbalance.png')
    plt.show()


def generate_correlation_heatmap(train_dataframe, validation_dataframe, test_dataframe):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.suptitle('Correlation heatmap')
    ax[0].set_title('Train')
    ax[1].set_title('Validation')
    ax[2].set_title('Test')
    sns.heatmap(train_dataframe.corr(), annot=True, ax=ax[0])
    sns.heatmap(validation_dataframe.corr(), annot=True, ax=ax[1])
    sns.heatmap(test_dataframe.corr(), annot=True, ax=ax[2])

    fig.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.show()


def show_some_misclassified_examples(test_dataframe, labels, crisp_predictions_dict, verbose=True):
    misclassified_dict = {'C': [], 'CP': [], 'CPS': []}
    pd_columns = ['Predicted_OC', 'Predicted_ST', 'Predicted_SE', 'Predicted_CN']
    for model_type, predictions in crisp_predictions_dict.items():
        if model_type in ['random', 'majority', 'one']:
            continue
        for i in range(labels.shape[1]):
            misclassified_dataframe = test_dataframe[
                np.logical_and(labels[:, i] != predictions[:, i], labels[:, i] == 1)
            ]

            if len(misclassified_dataframe) == 0:
                misclassified_dict[model_type].append([])
                continue
            elif len(misclassified_dataframe) > 10:
                misclassified_dataframe = misclassified_dataframe.sample(n=10)

            misclassified_predictions = predictions[misclassified_dataframe.index]
            misclassified_predictions = pd.DataFrame(misclassified_predictions, columns=pd_columns)
            misclassified_dataframe = pd.concat(
                [misclassified_dataframe.reset_index(drop=True), misclassified_predictions.reset_index(drop=True)],
                axis=1
            )
            misclassified_dict[model_type].append(misclassified_dataframe)

            if verbose:
                print('Model type: ', model_type, ' - Label: ', labels_columns[i])
                print(misclassified_dataframe)
                print()

    return misclassified_dict
