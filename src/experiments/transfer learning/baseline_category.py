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
    model_type: Literal["fcn", "encoder"],
    source_model: keras.models.Model,
    number_of_epochs=10,
) -> dict:
    with mlflow.start_run(nested=True, run_name="Destination"):
        X, y = Reading().read_dataset(dataset=dataset)
        experiment = BaselineExperiment(model=model_type)
        data_generator_train, validation_data = experiment.prepare_generators(X, y)
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

        experiment.log(
            data_generator_train=data_generator_train,
            validation_data=validation_data,
            model=model,
            y_encoder=experiment.y_encoder,
            history=history.history,
        )
        return history.history


def main(category, selection_method, model_type, number_of_epochs=10):
    mlflow.set_experiment("Transfer learning - same category, other datasets - Encoder")
    mlflow_logging = MlFlowLogging()  # pylint: disable=redefined-outer-name
    mlflow.tensorflow.autolog(log_models=False)
    for dataset in Reading().return_datasets_for_category(category):
        with mlflow.start_run(run_name=dataset):
            source_model = train_source_model(
                dataset=dataset,
                selection_method=selection_method,
                model=model_type,
                number_of_epochs=number_of_epochs,
            )
            destination = train_destination_model(
                dataset=dataset,
                model_type=model_type,
                source_model=source_model,
                number_of_epochs=number_of_epochs,
            )
            history = destination["history"]
            mlflow_logging.log_history(history)
        # TODO log accuracy for plain, history, etc


main(category="IMAGE")
main(category="ECG")
