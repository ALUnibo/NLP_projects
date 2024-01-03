import random as rnd
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertModel
from torch import nn

# baseline: random uniform classifier
def random_uniform_classifier(n_instances, cols_name):
    random_dataframe = pd.DataFrame(0, index=list(range(n_instances)), columns=cols_name)
    for i in range(n_instances):
        rnd_col = rnd.choice(cols_name)
        random_dataframe.iloc[i, list(cols_name).index(rnd_col)] = 1
    return random_dataframe


# baseline: random uniform classifier
def majority_classifier(targets):
    n_instances = targets.shape[0]
    max_idx = np.argmax(targets.apply(np.sum, axis=0))
    majority_dataframe = pd.DataFrame(0, index=list(range(n_instances)), columns=targets.columns)
    for i in range(n_instances):
        majority_dataframe.iloc[i, max_idx] = 1
    return majority_dataframe


class BertMultiLabelClassifier(nn.Module):
    def __init__(self, num_out_layer_labels):
        self.transformer = BertModel.from_pretrained("prajjwal1/bert-tiny")
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(128, num_out_layer_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.transformer(x)
        return self.softmax(self.linear(x))

    # def train(self):
    #     pass
    #
    # def evaluate(self):
    #     pass


def c_model(num_labels):
    model = BertMultiLabelClassifier(num_labels)
    return model


def c_model(num_labels):
    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny",
                                                          num_labels=num_labels,
                                                          problem_type="multi_label_classification")
    return model


def cp_model(num_labels):
    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny",
                                                          num_labels=num_labels,
                                                          problem_type="multi_label_classification")
    return model


def cps_model(num_labels):
    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny",
                                                          num_labels=num_labels,
                                                          problem_type="multi_label_classification")
    return model
