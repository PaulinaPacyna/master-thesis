from collections import Counter

import numpy as np
import keras
import sklearn
from sklearn.preprocessing import OneHotEncoder
import logging
from preprocessing.utils import normalize_length
from keras.utils import Sequence

# try:
#     from keras.utils.all_utils import Sequence
# except:
#     from keras.utils import Sequence


class ConstantLengthDataGenerator(Sequence):
    def __init__(
        self,
        X: np.array,
        y: np.array,
        shuffle: bool = True,
        batch_size: int = 32,
        dtype: np.dtype = np.float16,
        min_length: int = 2**4,
        max_length: int = 2**11,
        augmentation_probability: float = 0.01,
        cutting_probability: float = 0.01,
        padding_probability: float = 0.99,
        logging_call: callable = None,
    ):
        """Initialization"""
        self.shuffle = shuffle
        self.X = self.__normalize_rows(X)
        self.y: np.array = y
        self.indices = range(X.shape[0])
        self.batch_size = batch_size
        self.possible_lengths = [
            2**i
            for i in range(int(np.log2(min_length)), int(np.log2(max_length)) + 1)
        ]
        self.dtype = dtype
        self.__y_inverse_probabilities = self.__calculate_y_inverse_probabilities()
        self.augmentation_probability = augmentation_probability
        self.cutting_probability = cutting_probability
        self.padding_probability = padding_probability
        self.logging_call = logging_call

    def __augment(self, X: np.array):
        if np.random.random() > self.augmentation_probability:
            return X
        else:
            if np.random.random() < 0.5:
                X *= -1
            if np.random.random() < 0.5:
                X = np.flip(X, axis=1)
            return X

    @staticmethod
    def __normalize_rows(X) -> np.array:
        return np.array(
            [(row - np.mean(row)) / (np.std(row)) for row in X], dtype=np.object_
        )

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.X.shape[0] // self.batch_size * 10 + 1

    def __iter__(self):
        return self

    def __getitem__(self, i):
        return next(self)

    def __calculate_y_inverse_probabilities(self):
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
        series_length = np.random.choice(self.possible_lengths)
        batch_size = self.batch_size
        index = np.random.choice(
            self.indices, batch_size, p=self.__y_inverse_probabilities
        )
        X_batch = np.vstack(
            [
                normalize_length(
                    series,
                    target_length=series_length,
                    cutting_probability=self.cutting_probability,
                    stretching_probability=1 - self.padding_probability,
                )
                for series in self.X[index]
            ]
        )
        y_batch = self.y[index]
        X_batch, y_batch = np.array(X_batch, dtype=self.dtype), np.array(y_batch)
        X_batch = self.__augment(X_batch)
        return (np.expand_dims(X_batch, axis=-1), y_batch)

    # def on_epoch_end(self):
    #     self.indices = range(self.X.shape[0])
    #     self.log()
    #
    # def log(self, ignore=None):
    #     if ignore is None:
    #         ignore = ["X", "y", "_ConstantLengthDataGenerator__y_inverse_probabilities"]
    #     if self.logging_call:
    #         self.logging_call(
    #             {"gen_" + key: value for key, value in vars(self).items() if key not in ignore}
    #         )
    #     else:
    #         logging.warning("Not logging to mlflow")


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
