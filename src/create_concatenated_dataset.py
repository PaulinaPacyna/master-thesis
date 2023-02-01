import json
import os

import numpy as np

from src.preprocessing import get_paths, TargetEncoder, read_univariate_ts


def create_concatenated(root_data_path="data/") -> (np.array, np.array, np.array):
    X_final = []
    y_final = []
    categories_dict = json.load(open(os.path.join(root_data_path, "categories.json")))
    categories = []
    for path in get_paths(root_data_path):
        X, y = read_univariate_ts(path)
        dataset_name = path.split(os.sep)[1]
        y = TargetEncoder(y).get_categorical_column(prefix=dataset_name)
        for series in X:  # TODO handle nans via interpolation
            assert not np.isnan(series).any()
        X_final.append(X)
        y_final.append(y)
        try:
            categories.append(np.full(y.shape, categories_dict[dataset_name]))
        except KeyError:
            raise KeyError(f"Dataset {dataset_name} missing in the categories.json")
    return (
        np.concatenate(X_final, dtype="object"),
        np.concatenate(y_final),
        np.concatenate(categories),
    )


def save_datasets(X_final, y_final, categories, SAVING_DATA_PATH="data/"):
    os.makedirs(SAVING_DATA_PATH, exist_ok=True)
    with open(os.path.join(SAVING_DATA_PATH, "X.npy"), "wb") as file:
        np.save(file, X_final)
    with open(os.path.join(SAVING_DATA_PATH, "y.npy"), "wb") as file:
        np.save(file, y_final)
    with open(os.path.join(SAVING_DATA_PATH, "categories.npy"), "wb") as file:
        np.save(file, categories)


X_final, y_final, categories = create_concatenated()
save_datasets(X_final, y_final, categories)
