import torch
from torcheval.metrics.functional import binary_f1_score


def f1_score(predictions, targets):
    cols = predictions.shape[1]
    per_col_scores = torch.zeros(cols)
    for i in range(cols):
        per_col_scores[i] = binary_f1_score(predictions[:, i], targets[:, i])
    return torch.mean(per_col_scores)

