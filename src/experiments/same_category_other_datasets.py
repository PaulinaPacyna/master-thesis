import logging
import os

import mlflow
import numpy as np
import tensorflow.keras as keras

from experiments import BaseExperiment
from mlflow_logging import MlFlowLogging
from models import Encoder_model
from reading import ConcatenatedDataset

logging.getLogger().setLevel(logging.INFO)

mlflow_logging = MlFlowLogging()


class EncoderExperiment(BaseExperiment):
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

        experiment = EncoderExperiment(
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
        experiment = EncoderExperiment(
            saving_path=f"encoder_same_cat_other_datasets/dest/dataset={dataset}",
            use_early_stopping=False,
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
        experiment = EncoderExperiment(
            saving_path=f"encoder_same_cat_other_datasets/dest_plain/dataset={dataset}",
            use_early_stopping=False,
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
