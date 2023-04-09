from collections import Counter

import mlflow
import numpy as np
import logging

from mlflow import MlflowException

from preprocessing.utils import normalize_length

try:
    from keras.utils.all_utils import Sequence
except ModuleNotFoundError:
    from keras.utils import Sequence


class BaseDataGenerator(Sequence):
    def __init__(
        self,
        X: np.array,
        y: np.array,
        shuffle: bool = True,
        batch_size: int = 32,
        dtype: np.dtype = np.float16,
        length: int = 2**8,
        augmentation_probability: float = 0,
        cutting_probability: float = 0,
        padding_probability: float = 1,
    ):
        """Initialization"""
        self.shuffle = shuffle
        self.X = X
        self.y: np.array = y
        self.indices = range(X.shape[0])
        self.batch_size = batch_size
        self.length = length
        self.dtype = dtype
        self._y_inverse_probabilities = self._calculate_y_inverse_probabilities()
        self.augmentation_probability = augmentation_probability
        self.cutting_probability = cutting_probability
        self.padding_probability = padding_probability
        self.log()
        self.epoch = 0

    def __augment(self, X: np.array):
        if np.random.random() > self.augmentation_probability:
            return X
        else:
            if np.random.random() < 0.5:
                X *= -1
            if np.random.random() < 0.5:
                X = np.flip(X, axis=1)
            return X

    def __normalize_rows(self, X) -> np.array:
        return np.array(
            [self.__normalize_row(row) for row in X],
            dtype=np.object_,
        )

    @staticmethod
    def __normalize_row(row) -> np.array:
        return (row - np.mean(row)) / max(np.std(row), 1e-10)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return (self.X.shape[0] // self.batch_size + 1) * 10

    def __iter__(self):
        return self

    def __getitem__(self, i):
        return next(self)

    def _calculate_y_inverse_probabilities(self):
        if len(self.y.shape) == 1:
            y_list = self.y.tolist()
        elif len(self.y.shape) == 2:
            y_list = np.argmax(self.y, axis=1).tolist()
        else:
            raise NotImplementedError("self.y should be at most 2 dimensional")
        count_dict = Counter(y_list)
        counts = np.array([count_dict[item] for item in y_list])
        assert not (counts == 0).any()
        inverse_counts = 1 / counts
        return inverse_counts / sum(inverse_counts)

    def __next__(self):
        """Generate one batch of data"""
        batch_size = self.batch_size
        index = np.random.choice(
            self.indices, batch_size, p=self._y_inverse_probabilities
        )
        X_batch = self.prepare_X(self.X[index])
        y_batch = self.y[index]
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def prepare_X(self, X, series_length=None):
        if not series_length:
            series_length = self.length
        X_batch = [
            normalize_length(
                series,
                target_length=series_length,
                cutting_probability=self.cutting_probability,
                stretching_probability=1 - self.padding_probability,
            )
            for series in X
        ]

        X_batch = self.__normalize_rows(X_batch)
        X_batch = np.vstack(X_batch)
        X_batch = np.array(X_batch, dtype=self.dtype)
        X_batch = self.__augment(X_batch)
        return X_batch

    def on_epoch_end(self):
        self.indices = range(self.X.shape[0])
        self.epoch += 1

    def log(self, ignore=None):
        if ignore is None:
            ignore = ["X", "y", "_ConstantLengthDataGenerator__y_inverse_probabilities"]
        try:
            mlflow.log_params(
                {
                    "gen_" + key: value
                    for key, value in vars(self).items()
                    if key not in ignore
                }
            )
        except MlflowException as e:
            logging.warning(e)


class ConstantLengthDataGenerator(BaseDataGenerator):
    pass
