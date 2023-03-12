from collections import Counter

import mlflow
import numpy as np
import keras
import sklearn
from sklearn.preprocessing import OneHotEncoder
import logging
from preprocessing.utils import normalize_length

try:
    from keras.utils.all_utils import Sequence
except:
    from keras.utils import Sequence


class ConstantLengthDataGenerator(Sequence):
    def __init__(
        self,
        X: np.array,
        y: np.array,
        shuffle: bool = True,
        batch_size: int = 32,
        dtype: np.dtype = np.float16,
        min_length: int = 2**8,
        max_length: int = 2**8,
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
        self.possible_lengths = [
            2**i
            for i in range(int(np.log2(min_length)), int(np.log2(max_length)) + 1)
        ]
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
        y_hashed = np.apply_along_axis(
            lambda x: hash(tuple(x)), axis=1, arr=self.y
        ).tolist()
        count_dict = Counter(y_hashed)
        counts = np.array([count_dict[item] for item in y_hashed])
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
            series_length = np.random.choice(self.possible_lengths)
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
        except Exception as e:
            logging.warning(e)


class SelfLearningDataGenerator(ConstantLengthDataGenerator):
    def __init__(
        self,
        X: np.array,
        y: np.array,
        X_self_learning: np.array,
        self_learning_threshold: float = 0.9,
        shuffle: bool = True,
        batch_size: int = 32,
        dtype: np.dtype = np.float16,
        min_length: int = 2**8,
        max_length: int = 2**8,
        augmentation_probability: float = 0,
        cutting_probability: float = 0,
        padding_probability: float = 1,
        self_learning_cold_start: int = 0,
    ):
        super().__init__(
            X,
            y,
            shuffle=shuffle,
            batch_size=batch_size,
            dtype=dtype,
            min_length=min_length,
            max_length=max_length,
            augmentation_probability=augmentation_probability,
            padding_probability=padding_probability,
            cutting_probability=cutting_probability,
        )
        self.self_learning_threshold = self_learning_threshold
        self.model = None
        self.self_learning_X = X_self_learning
        self.self_learning_cold_start = self_learning_cold_start
        self.number_of_observation_added_sl = dict()
        self.original_X = X
        self.original_y = y

    def add_model(self, model: keras.models.Model) -> None:
        self.model = model

    def on_epoch_end(self):
        if self.epoch >= self.self_learning_cold_start:
            self.__add_self_learning_data()
        self.__add_self_learning_data()
        super().on_epoch_end()

    def __add_self_learning_data(self):
        self_learning_X = self.prepare_X(self.self_learning_X)
        predictions = self.model.predict(self_learning_X)
        score = np.max(predictions, axis=1)
        index = score >= self.self_learning_threshold
        self.X = np.concatenate([self.original_X, self.self_learning_X[index]])
        self.y = np.concatenate([self.original_y, predictions[index]])
        print(self.y.shape)
        no_observations_added = sum(index)
        logging.info(
            "Added %s observations with a threshold of %s",
            no_observations_added,
            self.self_learning_threshold,
        )
        self.number_of_observation_added_sl[self.epoch] = no_observations_added
        mlflow.log_metric(
            "number_of_observation_added_sl", no_observations_added, step=self.epoch
        )
        self._y_inverse_probabilities = self._calculate_y_inverse_probabilities()


if __name__ == "__main__":
    X = np.load("../data/X.npy", allow_pickle=True)
    y = OneHotEncoder(sparse=False).fit_transform(
        np.load("../data/y.npy", allow_pickle=True)
    )
    data_gen = ConstantLengthDataGenerator(X, y)
    sum_x = 0
    variable = next(data_gen)
    for x, y in data_gen:
        sum_x += x.shape[0]
        print(x.shape)
