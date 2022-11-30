import numpy as np
import keras
import sklearn
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
        self.encoder = sklearn.preprocessing.OneHotEncoder(categories='auto')
        self.y = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()
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
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.batches)


