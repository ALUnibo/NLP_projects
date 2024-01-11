import torch
from torcheval.metrics.functional import binary_f1_score


def calculate_f1_score(predictions, targets, verbose=False):
    cols = predictions.shape[1]
    single_class_scores = torch.zeros(cols)
    for i in range(cols):
        single_class_scores[i] = binary_f1_score(predictions[:, i], targets[:, i])
        if verbose:
            print('F1 score for column %d: %.3f' % (i, single_class_scores[i]))
    return torch.mean(single_class_scores), single_class_scores
