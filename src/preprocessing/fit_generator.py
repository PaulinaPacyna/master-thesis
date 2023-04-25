import logging
from collections import Counter

import mlflow
import numpy as np
from mlflow import MlflowException
from preprocessing.utils import get_lengths
from preprocessing.utils import normalize_length
from reading import Reading
from tensorflow.keras.utils import Sequence  # pylint: disable


class BaseDataGenerator(Sequence):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        X: np.array,
        y: np.array,
        shuffle: bool = True,
        batch_size: int = 32,
        dtype: np.dtype = np.float16,
        augmentation_probability: float = 0,
        multiply_augmentation_probability: float = 1,
    ):
        """Initialization"""
        self.shuffle = shuffle
        self.X = X
        self.y: np.array = y
        self.indices = range(X.shape[0])
        self.batch_size = batch_size
        self.dtype = dtype
        self._y_inverse_probabilities = self._calculate_y_inverse_probabilities()
        self.augmentation_probability = augmentation_probability
        self.multiply_augmentation_probability = multiply_augmentation_probability
        self.log()
        self.epoch = 0

    def __augment(self, X: np.array):
        if np.random.random() > self.augmentation_probability:
            return X
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

    def __iter__(self):  # pylint: disable=non-iterator-returned
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

    def prepare_X(self, X: np.array, series_length: int):
        X_batch = self.__normalize_rows(X)
        X_batch = [
            normalize_length(
                series,
                target_length=series_length,
            )
            for series in X_batch
        ]
        X_batch = np.vstack(X_batch)
        X_batch = np.array(X_batch, dtype=self.dtype)
        X_batch = self.__augment(X_batch)
        return X_batch

    def on_epoch_end(self):
        self.indices = range(self.X.shape[0])
        self.epoch += 1
        self.augmentation_probability *= self.multiply_augmentation_probability

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
    def __init__(  # pylint: disable=too-many-arguments
        self,
        X: np.array,
        y: np.array,
        shuffle: bool = True,
        batch_size: int = 32,
        dtype: np.dtype = np.float16,
        length: int = 64,
        augmentation_probability: float = 0,
        multiply_augmentation_probability: float = 1,
    ):
        super().__init__(
            X=X,
            y=y,
            shuffle=shuffle,
            batch_size=batch_size,
            dtype=dtype,
            augmentation_probability=augmentation_probability,
            multiply_augmentation_probability=multiply_augmentation_probability,
        )
        self.length = length

    def __next__(self):
        """Generate one batch of data"""
        batch_size = self.batch_size
        index = np.random.choice(
            self.indices, batch_size, p=self._y_inverse_probabilities
        )
        X_batch = self.prepare_X(self.X[index], series_length=self.length)
        y_batch = self.y[index]
        y_batch = np.array(y_batch)
        return X_batch, y_batch


class VariableLengthDataGenerator(BaseDataGenerator):
    def __next__(self):
        """Generate one batch of data"""
        batch_size = self.batch_size
        index = np.random.choice(
            self.indices, batch_size, p=self._y_inverse_probabilities
        )
        X_batch = self.X[index]
        X_batch = self.prepare_X(X_batch, series_length=max(get_lengths(X_batch)))
        y_batch = self.y[index]
        y_batch = np.array(y_batch)
        return X_batch, y_batch
