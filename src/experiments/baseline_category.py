import logging

import mlflow
import numpy as np
from experiments.utils import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import Reading
from tensorflow import keras

logging.getLogger().setLevel(logging.INFO)

mlflow_logging = MlFlowLogging()

# TODO: remove if not needed
class BaselineExperiment(BaseExperiment):
    name = "baseline"


def train_source_model(
    category: str,
    dataset: str,
    batch_size: int = 256,
    input_length: int = 256,
    number_of_epochs: int = 10,
):
    with mlflow.start_run(nested=True, run_name="Source model"):
        X, y = Reading().read_dataset(category=category)
        mask = np.char.startswith(y.ravel(), prefix=f"{dataset}_")
        X, y = X[~mask], y[~mask]

        experiment = BaselineExperiment(
            saving_path=f"baseline_approach/source/category={category}/dataset={dataset}"
        )
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={
                "batch_size": batch_size,
                "length": input_length,
            },
            test_args={"length": input_length},
        )
        model = experiment.prepare_encoder_classifier(
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
        X, y = Reading().read_dataset(dataset=dataset)
        experiment = BaselineExperiment(
            saving_path=f"baseline_approach/dest/dataset={dataset}",
            use_early_stopping=False,
        )
        input_length = source_model.inputs[0].shape[1]
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={
                "batch_size": batch_size,
                "length": input_length,
            },
            test_args={"length": input_length},
        )
        model = experiment.swap_last_layer(
            source_model=source_model,
            number_of_classes=experiment.get_number_of_classes(),
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


def main(category):
    mlflow.set_experiment("Transfer learning - same category, other datasets - Encoder")
    mlflow_logging = MlFlowLogging()  # pylint: disable=redefined-outer-name
    mlflow.tensorflow.autolog(log_models=False)
    for dataset in Reading().return_datasets_for_category(category):
        with mlflow.start_run(run_name=dataset):
            source_model = train_source_model(
                category=category, dataset=dataset, number_of_epochs=10
            )
            destination = train_destination_model(
                dataset=dataset, source_model=source_model, number_of_epochs=10
            )
            history = destination["history"]
            mlflow_logging.log_history(history)


main(category="IMAGE")
main(category="ECG")
