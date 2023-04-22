import logging
from typing import Literal

import mlflow
import numpy as np
from experiments.utils import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import Reading
from selecting import DBASelector
from selecting import RandomSelector
from tensorflow import keras

logging.getLogger().setLevel(logging.INFO)

mlflow_logging = MlFlowLogging()

# TODO: remove if not needed
class BaselineExperiment(BaseExperiment):
    name = "baseline"


def train_source_model(
    dataset: str,
    selection_method=Literal["random", "similarity"],
    model=Literal["fcn", "encoder"],
    number_of_epochs: int = 10,
):
    with mlflow.start_run(nested=True, run_name="Source model"):
        if selection_method == "random":
            selector = RandomSelector()
        elif selection_method == "similarity":
            selector = DBASelector()
        else:
            raise KeyError()
        datasets = selector.select(dataset=dataset)

        X, y = Reading().read_dataset(dataset=datasets)

        experiment = BaselineExperiment(model=model)
        data_generator_train, validation_data = experiment.prepare_generators(X, y)
        model = experiment.prepare_classifier()
        history = model.fit(
            data_generator_train,
            epochs=number_of_epochs,
            validation_data=validation_data,
            callbacks=experiment.callbacks,
        )
        experiment.log(
            data_generator_train=data_generator_train,
            validation_data=validation_data,
            model=model,
            y_encoder=experiment.y_encoder,
            history=history.history,
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
