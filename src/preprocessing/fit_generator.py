import numpy as np
import keras

from preprocessing.utils import TargetEncoder


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X: np.array,
                 y: np.array,
                 shuffle=True):
        """Initialization"""
        self.shuffle = shuffle
        self.X = X
        self.lengths = self.__get_lengths()
        self.batches = np.unique(self.lengths).tolist()
        self.encoder = TargetEncoder(y)
        self.y = self.encoder.categorical
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
        X_batch = self.X[index]
        y_batch = self.y[index]
        return np.vstack(X_batch), y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.batches)


