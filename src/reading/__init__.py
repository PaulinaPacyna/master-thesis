import json
import logging
import os
from functools import reduce
from typing import Tuple, List

import mlflow
import numpy as np
from mlflow import MlflowException

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

    def read_dataset_train_test_split(
        self, category: str = None, dataset: str = None, split: str = "train", log=True
    ) -> List[np.array]:
        split = split.lower()
        X: np.array = np.load(f"{self.data_root_path}/X_{split}.npy", allow_pickle=True)
        y: np.array = np.load(f"{self.data_root_path}/y_{split}.npy")
        if not category and not dataset:
            self.log_param(f"dataset_{split}", "whole")
            self.log_param(f"shapes_{split}", f"X: {X.shape}, y: {y.shape}")
            return X, y
        elif dataset:
            datasets = [dataset]
            logging.info("Loading only one dataset: %s", dataset)
            self.log_param(f"dataset_{split}", dataset)
        else:
            logging.info("Loading only one category: %s", category)
            datasets = [
                dataset_name
                for dataset_name in self.categories
                if self.categories[dataset_name] == category
            ]
            self.log_param(f"dataset_{split}", category)
        masks = [(np.char.startswith(y, dataset)).reshape(-1) for dataset in datasets]
        mask = reduce(np.logical_or, masks)
        y = y[mask, :]
        X = X[mask]  # TODO log category and y_unique here
        self.log_param(f"shapes_{split}", f"X: {X.shape}, y: {y.shape}")
        return X, y

    def read_dataset(
        self, category: str = None, dataset: str = None
    ) -> Tuple[np.array, np.array]:
        X_train, y_train = self.read_dataset_train_test_split(
            split="train", category=category, dataset=dataset
        )
        X_test, y_test = self.read_dataset_train_test_split(
            split="test", category=category, dataset=dataset
        )
        return np.concatenate([X_train, X_test], axis=0), np.concatenate(
            [y_train, y_test], axis=0
        )

    def return_datasets_for_category(self, category: str) -> List[str]:
        return sorted(
            [
                key
                for key, value in self.categories.items()
                if value.lower() == category.lower()
            ]
        )
    def log_param(self, key, val):
        try: 
            mlflow.log_param(key, val)
        except MlflowException as e:
            logging.warning("Skipping logging to mlflow: %s", e)

if __name__ == "__main__":
    concatenated_dataset = ConcatenatedDataset()
    datasets = concatenated_dataset.create_concatenated()
    concatenated_dataset.save_datasets(*datasets)
