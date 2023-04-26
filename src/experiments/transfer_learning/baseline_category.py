import json
import logging
from typing import Literal
from typing import Tuple

import mlflow
import numpy as np
from experiments.utils import BaseExperiment
from keras import Model
from mlflow import MlflowClient
from mlflow_logging import MlFlowLogging
from reading import Reading
from selecting import DBASelector
from selecting import RandomSelector
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

logging.getLogger().setLevel(logging.INFO)
mlflow_logging = MlFlowLogging()


# TODO: remove if not needed
class BaselineExperiment(BaseExperiment):
    name = "baseline"

    def get_source_experiment_metrics(
        self,
        dataset,
        param: str,
        experiment_id: str,
        type_: Literal["param", "metric"] = "param",
    ):
        all_runs = self._get_or_cache_runs_for_experiment(experiment_id)
        runs = [
            run
            for run in all_runs
            if run.data.params["dataset_train"] == dataset
            and run.info.run_name == "Source model"
        ]
        assert len(runs) == 1
        run = runs[0]
        if type_ == "param":
            return run.data.params[param]
        if type_ == "metric":
            return run.data.params[param]
        raise KeyError(type_)

    def extended_log(
        self,
        data_generator_train,
        validation_data: Tuple[np.array, np.array],
        model: Model,
        y_encoder: OneHotEncoder,
        history: dict,
        no_transfer_learning_history: dict,
    ):
        no_transfer_learning_history = {
            f"no_transfer_learning_{k}": v
            for k, v in no_transfer_learning_history.items()
        }
        self.log(
            data_generator_train=data_generator_train,
            validation_data=validation_data,
            model=model,
            y_encoder=y_encoder,
            history={**history, **no_transfer_learning_history},
        )


def train_source_model(
    dataset: str,
    selection_method: Literal["random", "similarity"],
    model: Literal["fcn", "encoder"],
    single_model_experiment_id: str,
    number_of_epochs: int = 10,
):
    with mlflow.start_run(run_name=f"Source model: {dataset}"):
        if selection_method == "random":
            selector = RandomSelector()
        elif selection_method == "similarity":
            selector = DBASelector()
        else:
            raise KeyError()
        datasets = selector.select(dataset=dataset)

        X, y = Reading().read_dataset(dataset=datasets)
        experiment = BaselineExperiment(model=model)
        mlflow.log_param(
            "mean_accuracies_source",
            experiment.get_mean_accuracies_from_experiment(
                experiment_id=single_model_experiment_id, datasets=datasets
            ),
        )
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
    single_model_experiment_id: str,
    number_of_epochs=10,
) -> dict:
    with mlflow.start_run(run_name=f"Destination model: {dataset}"):
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
        comparison_history = experiment.get_param_from_mlflow(
            experiment_id=single_model_experiment_id,
            dataset=dataset,
            param="history.json",
            type_="artifact",
        )
        comparison_history = json.load(open(comparison_history))
        experiment.extended_log(
            data_generator_train=data_generator_train,
            validation_data=validation_data,
            model=model,
            y_encoder=experiment.y_encoder,
            history=history.history,
            no_transfer_learning_history=comparison_history,
        )
        mlflow.log_metric(
            "no_transfer_learning_val_acc",
            experiment.get_param_from_mlflow(
                experiment_id=single_model_experiment_id,
                dataset=dataset,
                param="val_accuracy",
                type_="metric",
            ),
        )
        return history.history


def main(
    category,
    selection_method,
    model_type,
    single_model_experiment_id,
    this_experiment_id,
    number_of_epochs=10,
):
    mlflow.set_experiment(experiment_id=this_experiment_id)
    mlflow.tensorflow.autolog(log_models=False)
    for dataset in Reading().return_datasets_for_category(category):
        source_model = train_source_model(
            dataset=dataset,
            selection_method=selection_method,
            model=model_type,
            single_model_experiment_id=single_model_experiment_id,
            number_of_epochs=number_of_epochs,
        )
        train_destination_model(
            dataset=dataset,
            model_type=model_type,
            source_model=source_model,
            single_model_experiment_id=single_model_experiment_id,
            number_of_epochs=number_of_epochs,
        )
