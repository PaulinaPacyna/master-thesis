from typing import Tuple

import scipy.io.arff
import pandas as pd
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler


def read_ts(path: str) -> Tuple[np.array]:
    data = structured_to_unstructured(scipy.io.arff.loadarff(path)[0])
    len_columns = data.shape[1]
    return data[:, :len_columns - 1], data[:, len_columns - 1:len_columns]


def stretch_interpolate(df: np.array, target_length: int, type_: str = 'linear') -> np.array:
    interpolator = interp1d(np.arange(len(df)), df, kind=type_)
    new_index = np.linspace(0, len(df) - 1, target_length)
    return np.array(interpolator(new_index))


def stretch_interpolate_matrix(arr: np.array, target_width: int) -> np.array:
    return np.apply_along_axis(lambda arr: stretch_interpolate(arr, target_length=target_width),
                               axis=1,
                               arr=arr)


class TargetEncoder:
    def __init__(self, column: np.array):
        self.column = column.reshape(-1, 1)
        try:
            self.categorical: np.array = self.column.astype("int")
        except ValueError:
            self.categorical: np.array = self.column.astype('category').cat.codes
        self.__scaler = MinMaxScaler()

    def get_categorical_column(self, prefix: str = None):
        if prefix:
            return prefix + self.categorical
        return self.categorical

    def get_0_1_scaled(self):
        return self.__scaler.fit_transform(self.categorical)

    def reverse_0_1_scale(self, X: np.array):
        return self.__scaler.inverse_transform(X)
