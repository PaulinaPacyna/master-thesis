import logging
import os

import mlflow
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
from experiments import BaseExperiment
from models import FCN_model
from preprocessing.fit_generator import (
    SelfLearningDataGenerator,
    ConstantLengthDataGenerator,
)
from reading import ConcatenatedDataset


class SelfLearningExperiment(BaseExperiment):
    def prepare_FCN_model(self) -> keras.models.Model:
        number_of_classes = self.get_number_of_classes()
        input_layer = keras.layers.Input(shape=(None, 1))
        encoder_model = FCN_model(number_of_classes=number_of_classes)(input_layer)
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

    def prepare_generators(
        self, X: np.array, y: np.array, train_args: dict = {}, test_args: dict = {}
    ):
        if "X_self_learning" not in train_args:
            raise KeyError("Please specify X_self_learning in training_args")
        y = self.y_encoder.fit_transform(y.reshape(-1, 1)).toarray()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y
        )
        data_generator_train = SelfLearningDataGenerator(X_train, y_train, **train_args)
        validation_data = next(
            ConstantLengthDataGenerator(
                X_val, y_val, batch_size=len(y_val), **test_args
            )
        )
        mlflow.log_param("y.shape", y.shape)
        return data_generator_train, validation_data


if __name__ == "__main__":
    os.chdir("..")
    number_of_epochs = 3
    mlflow.set_experiment("Self learning - FCN")
    with mlflow.start_run(run_name="", nested=True):
        dataset = "ECG200"
        category = "ECG"
        X, y = ConcatenatedDataset().read_dataset(dataset=dataset)
        X_self_learning, _ = ConcatenatedDataset().read_dataset(category=category)
        experiment = SelfLearningExperiment(use_early_stopping=False)
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={
                "X_self_learning": X_self_learning,
                "self_learning_cold_start": 1,
            },
        )
        model = experiment.prepare_FCN_model()
        data_generator_train.add_model(model)
        model.fit(
            data_generator_train,
            epochs=number_of_epochs,
            validation_data=validation_data,
            callbacks=experiment.callbacks,
        )
