import random as rnd
import pandas as pd
import numpy as np
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



