import logging
import os
from typing import Optional

import mlflow
import numpy as np
import sklearn
import tensorflow as tf
from frozendict import frozendict
from keras.callbacks import EarlyStopping
from keras.models import clone_model
from models import Encoder_model
from models import FCN_model
from preprocessing import ConstantLengthDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras


class BaseExperiment:
    def __init__(
        self, saving_path: Optional[str] = None, use_early_stopping: bool = False
    ):
        self.decay = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5,
            decay_steps=10000,
            decay_rate=0.75,
        )
        self.callbacks = []
        if use_early_stopping:
            self.callbacks += [EarlyStopping(monitor="val_loss", patience=3)]
        if saving_path:
            self.output_directory = f"./data/models/{saving_path}"
            self.callbacks += [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.output_directory,
                    monitor="val_accuracy",
                    save_best_only=True,
                )
            ]
            os.makedirs(self.output_directory, exist_ok=True)
        self.y_encoder = sklearn.preprocessing.OneHotEncoder(categories="auto")

    def get_number_of_classes(self):
        return len(self.y_encoder.categories_[0])

    def prepare_generators(
        self,
        X: np.array,
        y: np.array,
        train_args: dict = frozendict(),
        test_args: dict = frozendict(),
    ):
        y = self.y_encoder.fit_transform(y.reshape(-1, 1)).toarray()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y
        )
        data_generator_train = ConstantLengthDataGenerator(
            X_train, y_train, **train_args
        )
        validation_data = next(
            ConstantLengthDataGenerator(
                X_val, y_val, batch_size=len(y_val), **test_args
            )
        )
        mlflow.log_param("y.shape", y.shape)
        return data_generator_train, validation_data

    def swap_last_layer(
        self, source_model: keras.models.Model, number_of_classes, compile_=True
    ) -> keras.models.Model:
        source_model.layers.pop()
        last = keras.layers.Dense(
            units=number_of_classes, activation="softmax", name="dense_appended"
        )(source_model.layers[-2].output)
        dest_model = keras.models.Model(inputs=source_model.input, outputs=last)
        if compile_:
            dest_model.compile(
                loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(self.decay),
                metrics=["accuracy"],
            )
        return dest_model

    def clean_weights(self, source_model: keras.models.Model):
        dest_model = clone_model(source_model)

        dest_model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(self.decay),
            metrics=["accuracy"],
        )
        return dest_model

    def prepare_FCN_model(self, scale: float = 1) -> keras.models.Model:
        number_of_classes = self.get_number_of_classes()
        input_layer = keras.layers.Input(shape=(None, 1))
        encoder_model = FCN_model(
            number_of_classes=number_of_classes, parameters=scale
        )(input_layer)
        model = keras.models.Model(inputs=input_layer, outputs=encoder_model)

        try:
            with open(os.path.join(self.output_directory, "model.json"), "w") as f:
                f.write(model.to_json())
        except AttributeError:
            logging.warning("Not saving model json")
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(self.decay),
            metrics=["accuracy"],
        )
        return model

    def prepare_encoder_classifier(self, input_length: int) -> keras.models.Model:
        number_of_classes = self.get_number_of_classes()
        input_layer = keras.layers.Input(shape=(input_length, 1))
        encoder_model = Encoder_model(number_of_classes=number_of_classes)(input_layer)
        model = keras.models.Model(inputs=input_layer, outputs=encoder_model)

        with open(os.path.join(self.output_directory, "model.json"), "w") as f:
            f.write(model.to_json())

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(self.decay),
            metrics=["accuracy"],
        )
        return model
