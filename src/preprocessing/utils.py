import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sktime.datasets import load_from_tsfile
import matplotlib.pyplot as plt


def get_paths(root="data", file_format="ts", task="TRAIN"):
    return [
        os.path.join(root, file_format, dataset, f"{dataset}_{task}.{file_format}")
        for dataset in os.listdir(os.path.join(root, file_format))
    ]


def read_univariate_ts(
        path: str, return_data_type="nested_univ"
) -> (np.array, np.array):
    X, y = load_from_tsfile(
        path, return_data_type=return_data_type, replace_missing_vals_with="0.0"
    )
    return X["dim_0"], y


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
    return df[start: start + target_length]


def pad(df: np.array, target_length: int) -> np.array:
    original_length = len(df)
    if target_length < original_length:
        raise ValueError(f"Cannot pad. Target length: {target_length}, original length: {original_length}.")
    np.pad(df, pad_width=(0, target_length - original_length), mode='constant', constant_values=(0, 0))


def normalize_length(df: np.array, target_length: int = 600, cutting_probability=0.5, stretching_probability=0.5):
    df_length = len(df)
    if df_length == target_length:
        return df
    if df_length >= target_length:
        if np.random.random() <= cutting_probability:
            return random_sub_interval(df, target_length)
        return stretch_interpolate(df, target_length)
    else:
        if np.random.random() <= stretching_probability:
            return stretch_interpolate(df, target_length)
        return pad(df, target_length)


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


def plot(X, y):
    color = TargetEncoder(y).get_0_1_scaled()
    plt.figure(figsize=(15, 10))
    for i in range(len(X)):
        plt.plot(X[i, :], c=plt.cm.rainbow(color[i]))
    plt.legend(y.ravel())


def get_lengths(X: np.array) -> np.array:
    return np.apply_along_axis(len, arr=X, axis=-1)
