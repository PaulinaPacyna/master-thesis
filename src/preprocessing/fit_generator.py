from collections import Counter

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


class VariableLengthDataGenerator(Sequence):
    def __init__(self, X: np.array, y: np.array, shuffle=True):
        """Initialization"""
        self.shuffle = shuffle
        self.X = X
        self.lengths = self.__get_lengths()
        self.batches = np.unique(self.lengths).tolist()
        self.y = y
        self.on_epoch_end()

    def __get_lengths(self):
        return np.vectorize(len)(self.X).reshape(-1)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.batches)

    def __getitem__(self, index):
        """Generate one batch of data"""

        length = self.batches[index]
        index = self.lengths == length
        X_batch = np.vstack(self.X[index])
        y_batch = self.y[index]
        return np.array(X_batch, dtype=np.float16), np.array(y_batch, dtype=np.float16)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            np.random.shuffle(self.batches)


class ConstantLengthDataGenerator(Sequence):
    def __init__(
            self, X: np.array,
            y: np.array,
            shuffle=True,
            batch_size=32,
            dtype=np.float16,
            min_length=2 ** 4,
            max_length=2 ** 11,
            augmentation=False
    ):
        """Initialization"""
        self.shuffle = shuffle
        self.X = self.__normalize_rows(X)
        self.y: np.array = y
        self.indices = range(X.shape[0])
        self.batch_size = batch_size
        self.possible_lengths = [2 ** i for i in range(int(np.log2(min_length)), int(np.log2(max_length)) + 1)]
        self.dtype = dtype
        self.__y_inverse_probabilities = self.__calculate_y_inverse_probabilities()
        self.__augmentation = augmentation

    def __augment(self, X: np.array):
        if not self.__augmentation:
            return X
        else:
            if np.random.random() < 0.5:
                X *= -1
            if np.random.random() < 0.5:
                X = np.flip(X, axis=1)
            return X

    @staticmethod
    def __normalize_rows(X) -> np.array:
        return np.array([(row - np.mean(row)) / (np.std(row)) for row in X], dtype=np.object_)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.X.shape[0] // self.batch_size * 10

    def __iter__(self):
        return self

    def __getitem__(self, i):
        return next(self)

    def __calculate_y_inverse_probabilities(self):
        y_hashed = np.apply_along_axis(lambda x: hash(tuple(x)), axis=1, arr=self.y).tolist()
        count_dict = Counter(y_hashed)
        counts = np.array([count_dict[item] for item in y_hashed])
        assert not (counts == 0).any()
        inverse_counts = 1 / counts
        return inverse_counts / sum(inverse_counts)

    def __next__(self):
        """Generate one batch of data"""
        series_length = np.random.choice(self.possible_lengths)
        batch_size = self.batch_size
        index = np.random.choice(
            self.indices, batch_size, p=self.__y_inverse_probabilities
        )
        X_batch = np.vstack(
            [
                normalize_length(series, target_length=series_length)
                for series in self.X[index]
            ]
        )
        y_batch = self.y[index]
        X_batch, y_batch = np.array(X_batch, dtype=self.dtype), np.array(y_batch)
        X_batch = self.__augment(X_batch)
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = range(self.X.shape[0])


if __name__ == "__main__":
    X = np.load("../data/concatenated/X.npy", allow_pickle=True)
    y = OneHotEncoder(sparse=False).fit_transform(
        np.load("../data/concatenated/y.npy", allow_pickle=True)
    )
    data_gen = ConstantLengthDataGenerator(X, y)
    sum_x = 0
    variable = next(data_gen)
    for x, y in data_gen:
        sum_x += x.shape[0]
        print(x.shape)
