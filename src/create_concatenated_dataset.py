import os

import numpy as np
from sktime.datasets import load_from_tsfile

from src.preprocessing import get_paths, TargetEncoder


def create_contatenated(root_data_path='data/') -> (np.array, np.array):
    global X_final, y_final
    X_final = []
    y_final = []
    for path in get_paths(root_data_path):
        X, y = load_from_tsfile(path)
        y = TargetEncoder(y).get_categorical_column(prefix=path.split(os.sep)[2])

        X_final.append(X)
        y_final.append(y)
    return np.concatenate(X_final), np.concatenate(y_final)


def save_datasets(X_final, y_final, SAVING_DATA_PATH='data/concatenated/'):
    os.makedirs(SAVING_DATA_PATH, exist_ok=True)
    with open(os.path.join(SAVING_DATA_PATH, 'X.npy'), 'wb') as file:
        np.save(file, X_final)
    with open(os.path.join(SAVING_DATA_PATH, 'y.npy'), 'wb') as file:
        np.save(file, y_final)


X_final, y_final = create_contatenated()
save_datasets(X_final, y_final)
