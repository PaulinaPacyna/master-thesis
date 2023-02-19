import logging
import os
from typing import Optional

import mlflow
import numpy as np
import sklearn
import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import EarlyStopping
from keras.models import clone_model
from sklearn.model_selection import train_test_split

from mlflow_logging import MlFlowLogging
from models import Encoder_model
from preprocessing import ConstantLengthDataGenerator
from reading import ConcatenatedDataset

logging.getLogger().setLevel(logging.INFO)

mlflow_logging = MlFlowLogging()


class Experiment:
    def __init__(self, saving_path: Optional[str] = None, use_early_stopping=True):
        self.decay = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=100000,
            decay_rate=0.96,
        )
        self.callbacks = []
        if use_early_stopping:
            self.callbacks += [EarlyStopping(monitor='val_loss', patience=3)]
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

    def prepare_generators(
        self, X: np.array, y: np.array, train_args: dict = {}, test_args: dict = {}
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

    def prepare_encoder_classifier(
        self, number_of_classes: int, input_length: int
    ) -> keras.models.Model:
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

    def swap_last_layer(self, source_model: keras.models.Model, number_of_classes):
        source_model.layers.pop()
        last = keras.layers.Dense(units=number_of_classes, activation="softmax")(
            source_model.layers[-2].output
        )
        dest_model = keras.models.Model(inputs=source_model.input, outputs=last)

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


def train_source_model(
    category: str,
    dataset: str,
    batch_size: int = 256,
    input_length: int = 256,
    number_of_epochs: int = 10,
):
    with mlflow.start_run(nested=True, run_name="Source model"):
        X, y = ConcatenatedDataset().read_dataset(category=category)
        mask = np.char.startswith(y.ravel(), prefix=f"{dataset}_")
        X, y = X[~mask], y[~mask]

        experiment = Experiment(
            saving_path=f"encoder_same_cat_other_datasets/source/category={category}/dataset={dataset}"
        )
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={
                "batch_size": batch_size,
                "min_length": input_length,
                "max_length": input_length,
            },
            test_args={"min_length": input_length, "max_length": input_length},
        )
        model = experiment.prepare_encoder_classifier(
            number_of_classes=len(experiment.y_encoder.categories_[0]),
            input_length=input_length,
        )
        history = model.fit(
            data_generator_train,
            epochs=number_of_epochs,
            validation_data=validation_data,
            callbacks=experiment.callbacks,
        )
        mlflow_logging.log_confusion_matrix(
            *validation_data, classifier=model, y_encoder=experiment.y_encoder
        )
        mlflow_logging.log_history(
            {f"source_{key}": value for key, value in history.history.items()},
        )
        mlflow_logging.log_example_data(
            *next(data_generator_train), encoder=experiment.y_encoder
        )
        return model


def train_destination_model(
    dataset: str,
    source_model: keras.models.Model,
    number_of_epochs=10,
    batch_size=256,
) -> dict:
    with mlflow.start_run(nested=True, run_name="Destination"):
        X, y = ConcatenatedDataset().read_dataset(dataset=dataset)
        experiment = Experiment(
            saving_path=f"encoder_same_cat_other_datasets/dest/dataset={dataset}",
            use_early_stopping=False
        )
        input_length = source_model.inputs[0].shape[1]
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={
                "batch_size": batch_size,
                "min_length": input_length,
                "max_length": input_length,
            },
            test_args={"min_length": input_length, "max_length": input_length},
        )
        model = experiment.swap_last_layer(
            source_model=source_model,
            number_of_classes=len(experiment.y_encoder.categories_[0]),
        )
        history = model.fit(
            data_generator_train,
            epochs=number_of_epochs,
            validation_data=validation_data,
            callbacks=experiment.callbacks,
        )

        history = {f"dest_{key}": value for key, value in history.history.items()}
        mlflow_logging.log_confusion_matrix(
            *validation_data, classifier=model, y_encoder=experiment.y_encoder
        )
        mlflow_logging.log_history(history)
        mlflow_logging.log_example_data(
            *next(data_generator_train), encoder=experiment.y_encoder
        )
        return {"history": history, "model": model}


def train_dest_model_no_weights(
    model,
    dataset: str,
    number_of_epochs=10,
    batch_size=256,
):
    with mlflow.start_run(nested=True, run_name="Destination plain"):
        X, y = ConcatenatedDataset().read_dataset(dataset=dataset)
        experiment = Experiment(
            saving_path=f"encoder_same_cat_other_datasets/dest_plain/dataset={dataset}",
            use_early_stopping=False
        )
        model = experiment.clean_weights(
            source_model=model,
        )
        input_length = model.inputs[0].shape[1]
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={
                "batch_size": batch_size,
                "min_length": input_length,
                "max_length": input_length,
            },
            test_args={"min_length": input_length, "max_length": input_length},
        )
        history = model.fit(
            data_generator_train,
            epochs=number_of_epochs,
            validation_data=validation_data,
            callbacks=experiment.callbacks,
        )
        mlflow_logging.log_confusion_matrix(
            *validation_data, classifier=model, y_encoder=experiment.y_encoder
        )
        history = {
            f"dest_no_weights_{key}": value for key, value in history.history.items()
        }
        mlflow_logging.log_history(
            history,
        )
        mlflow_logging.log_example_data(
            *next(data_generator_train), encoder=experiment.y_encoder
        )
        return {"history": history}


if __name__ == "__main__":
    mlflow.set_experiment("Transfer learning - same category, other datasets")
    mlflow_logging = MlFlowLogging()
    run = mlflow.start_run()
    mlflow.tensorflow.autolog()
    category = "ECG"
    dataset = "ECG200"
    source_model = train_source_model(category=category, dataset=dataset)
    destination = train_destination_model(dataset=dataset, source_model=source_model)
    plain_destination = train_dest_model_no_weights(
        model=destination["model"], dataset=dataset
    )
    history = {**destination["history"], **plain_destination["history"]}
    mlflow_logging.log_history(history)
