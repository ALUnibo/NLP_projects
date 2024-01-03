from sklearn.metrics import f1_score
import evaluate
import numpy as np


def f1_score(predictions, targets):
    cols = predictions.columns
    per_col_scores = np.zeros(len(cols))
    for i, c in enumerate(cols):
        per_col_scores[i] = f1_score(predictions[c], targets[c])
    return per_col_scores.mean()


def compute_metrics(output_info):
    acc_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    predictions, labels = output_info
    print(predictions)
    print(labels)
    predictions = np.argmax(predictions, axis=-1)

    print(labels)

    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return {**f1, **acc}
