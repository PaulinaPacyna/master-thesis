import logging
import os
from typing import List

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sktime.datasets import load_from_tsfile
import matplotlib.pyplot as plt


def get_paths(root="data", file_format="ts", split="TRAIN"):
    return [
        get_path_to_dataset(dataset, root=root, file_format=file_format, split=split)
        for dataset in get_all_datasets_by_name(root)
    ]


def get_path_to_dataset(name: str, root: str = "data", file_format="ts", split="TRAIN"):
    return os.path.join(root, file_format, name, f"{name}_{split}.{file_format}")


def get_all_datasets_by_name(root="data"):
    ts_formatted_datasets = os.listdir(os.path.join(root, "ts"))
    weka_formatted_datasets = os.listdir(os.path.join(root, "weka"))
    return list(set(ts_formatted_datasets + weka_formatted_datasets))


def read_univariate_ts(
    path: str, return_data_type="nested_univ"
) -> (np.array, np.array):
    try:
        X, y = load_from_tsfile(
            path,
            return_data_type=return_data_type,
            replace_missing_vals_with="0.0",
        )
        return X["dim_0"], y
    except OSError as e:
        print("filepath", path)
        raise e


def stretch_interpolate(
    df: np.array, target_length: int = 600, type_: str = "linear"
) -> np.array:
    interpolator = interp1d(np.arange(len(df)), df, kind=type_)
    new_index = np.linspace(0, len(df) - 1, target_length)
    return np.array(interpolator(new_index))


def random_sub_interval(df: np.array, target_length: int = 600):
    length_df = len(df)
    if length_df < target_length:
        raise KeyError(f"df length {length_df} shorter than {target_length}")
    if length_df == target_length:
        return df
    start = np.random.randint(length_df - target_length + 1)
    return df[start : start + target_length]


def pad(df: np.array, target_length: int) -> np.array:
    original_length = len(df)
    if target_length < original_length:
        raise ValueError(
            f"Cannot pad. Target length: {target_length}, original length: {original_length}."
        )
    return np.pad(
        df,
        pad_width=(0, target_length - original_length),
        mode="constant",
        constant_values=(0, 0),
    )


def normalize_length(
    df: np.array,
    target_length: int = 600,
    cutting_probability=0.5,
    stretching_probability=0.5,
):
    df_length = len(df)
    if df_length == target_length:
        result = df
    elif df_length > target_length:
        if np.random.random() <= cutting_probability:
            result = random_sub_interval(df, target_length)
        else:
            result = stretch_interpolate(df, target_length)
    else:
        if np.random.random() <= stretching_probability:
            result = stretch_interpolate(df, target_length)
        else:
            result = pad(df, target_length)
    assert len(result) == target_length
    return result


def stretch_interpolate_matrix(arr: np.array, target_width: int) -> np.array:
    return np.vstack(
        [stretch_interpolate(series, target_length=target_width) for series in arr]
    )


def sample_rows(X, y=None, n=200):
    if n >= X.shape[0]:
        return X, y if y is not None else X
    index = np.random.choice(X.shape[0], n, replace=False)
    return X[index], y[index] if y is not None else X[index]


class TargetEncoder:
    def __init__(self, column: np.array):
        self.column = column.reshape(-1, 1)
        try:
            self.categorical: np.array = self.column.astype("int")
        except ValueError:
            self.categorical: np.array = pd.Series(
                self.column.reshape(-1), dtype="category"
            ).cat.codes.values.reshape(-1, 1)
        self.__scaler = MinMaxScaler()

    def get_categorical_column(self, prefix: str = None) -> np.array:
        if prefix:
            return np.char.add(f"{prefix}_", self.categorical.astype("str"))
        return self.categorical

    def get_0_1_scaled(self):
        return self.__scaler.fit_transform(self.categorical)

    def reverse_0_1_scale(self, X: np.array):
        return self.__scaler.inverse_transform(X)


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def plot(X, y=None):
    color = TargetEncoder(y).get_0_1_scaled() if y is not None else None
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(len(X)):
        plt.plot(
            X[i, :],
            c=plt.cm.rainbow(color[i]).ravel() if y is not None else None,
            label=y[i] if y is not None else None,
        )
    legend_without_duplicate_labels(ax)
    return fig


def get_lengths(X: np.array) -> np.array:
    return np.apply_along_axis(len, arr=X, axis=-1)


def read_dataset(
    root_data_path: str = "./data",
    category: str = None,
    dataset: str = None,
    return_cat: bool = False,
    logging_call: callable = None,
) -> List[np.array]:
    X: np.array = np.load(f"{root_data_path}/X.npy", allow_pickle=True)
    y: np.array = np.load(f"{root_data_path}/y.npy")
    categories: np.array = np.load(f"{root_data_path}/categories.npy")
    if category and not dataset:
        logging.info("Loading only one category: %s", category)
        mask = (categories == category).reshape(-1)
        y = y[mask, :]
        X = X[mask]
        if logging_call:
            logging_call("category", category)
        return X, y
    if dataset:
        logging.info("Loading only one dataset: %s", dataset)
        mask = (np.char.startswith(y, dataset)).reshape(-1)
        y = y[mask, :]
        X = X[mask]
        if logging_call:
            logging_call("dataset", dataset)
        return X, y
    if logging_call:
        logging_call("y.unique", ", ".join(np.unique(y))[:500])
    if return_cat:
        return X, y, categories
    return X, y
