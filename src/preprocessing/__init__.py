import scipy.io.arff
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler


def read_ts(path):
    return scipy.io.arff.loadarff(path)


def stretch_interpolate(df: pd.Series, target_length: int, type_: str = 'linear') -> pd.Series:
    interpolator = interp1d(df.index, df.values, kind=type_)
    new_index = np.linspace(0, len(df) - 1, target_length)
    return pd.Series(interpolator(new_index))


class TargetEncoder:
    def __init__(self, column: pd.Series):
        self.column = column
        try:
            self.categorical: pd.Series = column.astype("int")
        except ValueError:
            self.categorical: pd.Series = column.astype('category').cat.codes
        self.__scaler = MinMaxScaler()

    def get_categorical_column(self, prefix: str = None):
        if prefix:
            return prefix + self.categorical
        return self.categorical

    def get_0_1_scaled(self):
        return self.__scaler.fit_transform(self.categorical)

    def reverse_0_1_scale(self, X: pd.Series):
        return self.__scaler.inverse_transform(X)
