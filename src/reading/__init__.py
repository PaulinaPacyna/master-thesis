import json
import logging
import os
from functools import reduce
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowException
from preprocessing import get_path_to_dataset
from preprocessing import remove_zeros_at_end
from preprocessing import TargetEncoder
from sktime.datasets import load_from_arff_to_dataframe
from sktime.datasets import load_from_tsfile


class Reading:
    def __init__(
        self, data_root_path=os.path.join(Path(__file__).parent.parent, "data")
    ):
        self.saving_data_path = data_root_path
        self.data_root_path = data_root_path
        self.categories = self.__load_categories_dict()

    def __load_categories_dict(self) -> dict:
        return json.load(open(os.path.join(self.data_root_path, "categories.json")))

    def create_concatenated(self) -> (np.array, np.array, np.array, np.array):
        X_train_final = []
        y_train_final = []
        X_test_final = []
        y_test_final = []
        for dataset_name in sorted(self.categories.keys()):
            X_train, y_train = self.load_single_dataset(dataset_name, split="TRAIN")
            X_test, y_test = self.load_single_dataset(dataset_name, split="TEST")
            X_train_final.append(X_train)
            y_train_final.append(y_train)
            X_test_final.append(X_test)
            y_test_final.append(y_test)
        return (
            np.concatenate(  # pylint: disable=unexpected-keyword-arg
                X_train_final, dtype="object"
            ),
            np.concatenate(y_train_final),
            np.concatenate(  # pylint: disable=unexpected-keyword-arg
                X_test_final, dtype="object"
            ),
            np.concatenate(y_test_final),
        )

    @staticmethod
    def load_single_dataset(dataset_name, split="TRAIN") -> Tuple[np.array, np.array]:
        try:
            try:
                path = get_path_to_dataset(dataset_name, split=split, file_format="ts")
                X, y = load_from_tsfile(
                    path,
                    return_data_type="nested_univ",
                    replace_missing_vals_with="0.0",
                )
            except FileNotFoundError as e:
                path = get_path_to_dataset(
                    dataset_name, split=split, file_format="arff"
                )
                logging.warning(e)
                logging.warning("Loading arff file instead.")
                X, y = load_from_arff_to_dataframe(
                    path, replace_missing_vals_with="0.0"
                )

        except OSError as e:
            raise OSError("Error when reading:", path) from e

        assert X.columns == ["dim_0"], f"more than one dimension in {path}"
        X = X["dim_0"]
        X = pd.Series([remove_zeros_at_end(x) for x in X])
        y = TargetEncoder(y).get_categorical_column(prefix=dataset_name)
        for series in X:
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
        self,
        category: str = None,
        dataset: Union[List[str], str] = None,
        split: str = "train",
        log=True,
        exclude_dataset: str = None,
    ) -> List[np.array]:
        split = split.lower()
        X: np.array = np.load(f"{self.data_root_path}/X_{split}.npy", allow_pickle=True)
        y: np.array = np.load(f"{self.data_root_path}/y_{split}.npy")
        if not category and not dataset:
            self.log_param(f"dataset_{split}", "whole")
            self.log_param(f"shapes_{split}", f"X: {X.shape}, y: {y.shape}")
            return X, y
        if dataset:
            if isinstance(dataset, str):
                datasets = [dataset]
            else:
                datasets = dataset
            if log:
                logging.info("Loading datasets: %s", ", ".join(datasets))
            self.log_param(f"dataset_{split}", ", ".join(datasets))
        else:
            if log:
                logging.info("Loading only one category: %s", category)
            datasets = [
                dataset_name
                for dataset_name in self.categories
                if self.categories[dataset_name] == category
                and dataset_name != exclude_dataset
            ]
            self.log_param(f"dataset_{split}", category)
        masks = [(np.char.startswith(y, dataset)).reshape(-1) for dataset in datasets]
        mask = reduce(np.logical_or, masks)
        y = y[mask, :]
        X = X[mask]
        self.log_param(f"shapes_{split}", f"X: {X.shape}, y: {y.shape}")
        return X, y

    def read_dataset(
        self,
        category: str = None,
        dataset: Union[str, List[str]] = None,
        exclude_dataset: str = None,
    ) -> Tuple[np.array, np.array]:
        X_train, y_train = self.read_dataset_train_test_split(
            split="train",
            category=category,
            dataset=dataset,
            exclude_dataset=exclude_dataset,
        )
        X_test, y_test = self.read_dataset_train_test_split(
            split="test",
            category=category,
            dataset=dataset,
            exclude_dataset=exclude_dataset,
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

    @staticmethod
    def log_param(key, val):
        try:
            mlflow.log_param(key, val)
        except MlflowException as e:
            logging.warning("Skipping logging to mlflow: %s", e)


if __name__ == "__main__":
    reading = Reading()
    datasets_concatenated = reading.create_concatenated()
    reading.save_datasets(*datasets_concatenated)
