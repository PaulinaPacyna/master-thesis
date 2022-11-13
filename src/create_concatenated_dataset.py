import os

import numpy as np

from src.preprocessing import get_paths, TargetEncoder, read_univariate_ts


def create_concatenated(root_data_path='data/') -> (np.array, np.array):
    X_final = []
    y_final = []
    for path in get_paths(root_data_path):
        X, y = read_univariate_ts(path)
        y = TargetEncoder(y).get_categorical_column(prefix=path.split(os.sep)[1])

        X_final.append(X)
        y_final.append(y)
    return np.concatenate(X_final, dtype = 'object'), np.concatenate(y_final)


def save_datasets(X_final, y_final, SAVING_DATA_PATH='data/concatenated/'):
    os.makedirs(SAVING_DATA_PATH, exist_ok=True)
    with open(os.path.join(SAVING_DATA_PATH, 'X.npy'), 'wb') as file:
        np.save(file, X_final)
    with open(os.path.join(SAVING_DATA_PATH, 'y.npy'), 'wb') as file:
        np.save(file, y_final)


X_final, y_final = create_concatenated()
save_datasets(X_final, y_final)
