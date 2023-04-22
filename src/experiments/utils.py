import logging
import os
from typing import Literal
from typing import Optional
from typing import Tuple

import mlflow
import numpy as np
import sklearn
import tensorflow as tf
from experiments.ensemble_category import mlflow_logging
from frozendict import frozendict
from keras import Model
from keras.callbacks import EarlyStopping
from keras.utils import Sequence
from models import Encoder_model
from models import FCN_model
from preprocessing import ConstantLengthDataGenerator
from preprocessing.fit_generator import VariableLengthDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

# TODO log here as in data generator
class BaseExperiment:
    name = "base"

    def __init__(
        self,
        model: Literal["fcn", "encoder"],
        input_length: int = 256,
        batch_size: int = 256,
        saving_path: Optional[str] = None,
        use_early_stopping: bool = False,
    ):
        self.model = model
        self.input_length = input_length
        self.batch_size
        self.transfer_learning_decay = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5,
            decay_steps=10000,
            decay_rate=0.75,
        )
        self.normal_decay = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=100000,
            decay_rate=0.96,
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
        if self.model == "encoder":
            data_generator_train = ConstantLengthDataGenerator(
                X_train, y_train, length=self.input_length, **train_args
            )
            validation_data = next(
                ConstantLengthDataGenerator(
                    X_val,
                    y_val,
                    length=self.input_length,
                    batch_size=len(y_val),
                    **test_args,
                )
            )
        elif self.model == "fcn":
            data_generator_train = VariableLengthDataGenerator(
                X_train, y_train, **train_args
            )
            validation_data = next(
                VariableLengthDataGenerator(
                    X_val, y_val, batch_size=len(y_val), **test_args
                )
            )
        else:
            raise KeyError()
        mlflow.log_param("y.shape", y.shape)
        return data_generator_train, validation_data

    def swap_last_layer(
        self, source_model: keras.models.Model, number_of_classes, compile_=True
    ) -> keras.models.Model:
        source_model.layers.pop()
        last = keras.layers.Dense(
            units=number_of_classes, activation="softmax", name="dense_appended"
        )(source_model.layers[-2].output)
        destination_model = keras.models.Model(inputs=source_model.input, outputs=last)
        if compile_:
            destination_model.compile(
                loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(self.transfer_learning_decay),
                metrics=["accuracy"],
            )
        return destination_model

    def prepare_classifier(self) -> keras.models.Model:
        number_of_classes = self.get_number_of_classes()
        if self.model == "encoder":
            input_layer = keras.layers.Input(shape=(self.input_length, 1))
            selected_model = Encoder_model(number_of_classes=number_of_classes)(
                input_layer
            )
        elif self.model == "fcn":
            input_layer = keras.layers.Input(shape=(None, 1))
            selected_model = FCN_model(number_of_classes=number_of_classes)(input_layer)
        else:
            raise KeyError()

        try:
            with open(os.path.join(self.output_directory, "model.json"), "w") as f:
                f.write(model.to_json())
        except AttributeError:
            logging.warning("Not saving model json")
        model = keras.models.Model(inputs=input_layer, outputs=selected_model)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(self.normal_decay),
            metrics=["accuracy"],
        )
        return model

    def log(
        self,
        data_generator_train: Sequence,
        validation_data: Tuple[np.array, np.array],
        model: Model,
        y_encoder: OneHotEncoder,
        history: dict,
    ):
        mlflow_logging.log_confusion_matrix(
            *validation_data, classifier=model, y_encoder=self.y_encoder
        )
        history = {f"{self.name}_{key}": value for key, value in history.items()}
        mlflow_logging.log_history(
            history,
        )
        mlflow_logging.log_example_data(*next(data_generator_train), encoder=y_encoder)
        return history
