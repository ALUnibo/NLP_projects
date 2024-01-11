import numpy as np
import torch


def random_uniform_classifier(test_labels):
    test_labels = test_labels.values
    n_instances = test_labels.shape[0]
    n_classes = test_labels.shape[1]

    predictions = np.random.randint(2, size=(n_instances, n_classes))
    predictions = torch.Tensor(predictions)
    return predictions


def majority_classifier(training_labels, test_labels):
    training_labels = training_labels.values
    test_labels = test_labels.values
    n_instances = test_labels.shape[0]
    n_classes = test_labels.shape[1]

    predictions = np.zeros((n_instances, n_classes))
    for i in range(n_classes):
        if np.sum(training_labels[:, i]) >= len(training_labels) / 2:
            predictions[:, i] = 1
    predictions = torch.Tensor(predictions)
    return predictions


def one_baseline(test_labels):
    test_labels = test_labels.values
    n_instances = test_labels.shape[0]
    n_classes = test_labels.shape[1]

    predictions = np.ones((n_instances, n_classes))
    predictions = torch.Tensor(predictions)
    return predictions
