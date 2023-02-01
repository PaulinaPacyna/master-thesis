import json
import logging
import os
from functools import reduce
from typing import Tuple, List

import numpy as np

from preprocessing import (
    get_path_to_dataset,
    TargetEncoder,
    read_univariate_ts,
    get_all_datasets_by_name,
)


class ConcatenatedDataset:
    def __init__(self, data_root_path=os.getenv("DATA_ROOT", "data")):
        self.saving_data_path = data_root_path
        self.data_root_path = data_root_path
        self.categories = self.__load_categories_dict()

    def __load_categories_dict(self):
        return json.load(open(os.path.join(self.data_root_path, "categories.json")))

    def create_concatenated(self) -> (np.array, np.array, np.array, np.array):
        X_train_final = []
        y_train_final = []
        X_test_final = []
        y_test_final = []
        for dataset_name in sorted(get_all_datasets_by_name(self.data_root_path)):
            X_train, y_train = self.load_single_dataset(dataset_name, split="TRAIN")
            X_test, y_test = self.load_single_dataset(dataset_name, split="TEST")
            X_train_final.append(X_train)
            y_train_final.append(y_train)
            X_test_final.append(X_test)
            y_test_final.append(y_test)
        return (
            np.concatenate(X_train_final, dtype="object"),
            np.concatenate(y_train_final),
            np.concatenate(X_test_final, dtype="object"),
            np.concatenate(y_test_final),
        )

    @staticmethod
    def load_single_dataset(dataset_name, split="TRAIN") -> Tuple[np.array, np.array]:
        X, y = read_univariate_ts(get_path_to_dataset(dataset_name, split=split))
        y = TargetEncoder(y).get_categorical_column(prefix=dataset_name)
        for series in X:  # TODO handle nans via interpolation
            assert not np.isnan(series).any()
        return X, y

    def save_datasets(
        self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array
    ):
        os.makedirs(self.saving_data_path, exist_ok=True)
        path_mapping = {
            "X_train.npy": X_train,
            "y_train.npy": y_train,
            "X_test.npy": X_test,
            "y_test.npy": y_test,
        }
        for path in path_mapping:
            with open(os.path.join(self.saving_data_path, path), "wb") as file:
                np.save(file, path_mapping[path])



if __name__ == "__main__":
    concatenated_dataset = ConcatenatedDataset()
    datasets = concatenated_dataset.create_concatenated()
    concatenated_dataset.save_datasets(*datasets)
