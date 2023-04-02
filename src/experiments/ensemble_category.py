from typing import List

import mlflow
import numpy as np
from mlflow import MlflowClient
from tensorflow import keras

from experiments import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import Reading

mlflow_logging = MlFlowLogging()


class EnsembleExperiment(BaseExperiment):
    def prepare_ensemble_model(
        self,
        source_models: List[keras.models.Model],
    ) -> keras.models.Model:
        target_number_of_classes = self.get_number_of_classes()
        first = keras.layers.Input(source_models[0].inputs[0].shape[1:])
        outputs = []
        for model in source_models:
            model = self.swap_last_layer(
                model, number_of_classes=target_number_of_classes, compile_=False
            )
            model = model(first)
            outputs.append(model)
        last = keras.layers.Add()(outputs) / len(source_models)
        model = keras.models.Model(inputs=first, outputs=last)
        return model


def read_or_train_model(
    dataset, component_experiment_id: str, root_path: str = "data/models/components"
) -> keras.models.Model:
    saving_path = f"{root_path}/{component_experiment_id}/dataset={dataset}"
    try:
        return keras.models.load_model(saving_path)
    except OSError as error:
        raise FileNotFoundError(
            f"Cannot find model for  dataset {dataset} and experiment {component_experiment_id}"
        ) from error


def get_accuracies_from_experiment(experiment_id: str, datasets: List[str]) -> float:
    all_runs = MlflowClient().search_runs([experiment_id])
    runs = [run for run in all_runs if run.data.params["dataset_train"] in datasets]
    accuracies = [run.data.metrics["val_accuracy"] for run in runs]
    return np.mean(accuracies)


def train_ensemble_model(
    target_dataset: str, category: str, component_experiment_id: str, epochs: int = 10
):
    with mlflow.start_run(run_name="ensemble", nested=True):
        reading = Reading()
        X, y = reading.read_dataset(dataset=target_dataset)
        all_datasets = reading.return_datasets_for_category(category=category)
        datasets = np.random.choice(all_datasets, 5, False)
        mlflow.log_param("Datasets used for ensemble", ", ".join(datasets))
        mlflow.log_param(
            "Mean accuracy of models used for ensemble",
            get_accuracies_from_experiment(
                experiment_id=component_experiment_id, datasets=datasets
            ),
        )
        experiment = EnsembleExperiment(
            saving_path=f"encoder_ensemble/ensemble/dataset={target_dataset}",
            use_early_stopping=False,
        )
        models = [
            read_or_train_model(
                dataset=dataset, component_experiment_id=component_experiment_id
            )
            for dataset in datasets
            if dataset != target_dataset
        ]
        input_len = models[0].input.shape[1]
        # TODO either set length in fcn or change this logic
        if input_len is None:
            input_len = 2**8
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={"min_length": input_len, "max_length": input_len},
            test_args={"min_length": input_len, "max_length": input_len},
        )
        ensemble_model = experiment.prepare_ensemble_model(models)
        ensemble_model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(experiment.decay),
            metrics=["accuracy"],
        )
        history = ensemble_model.fit(
            data_generator_train,
            epochs=epochs,
            validation_data=validation_data,
            use_multiprocessing=True,
        )
        mlflow_logging.log_confusion_matrix(
            *validation_data, classifier=ensemble_model, y_encoder=experiment.y_encoder
        )
        history = {f"ensemble_{key}": value for key, value in history.history.items()}
        mlflow_logging.log_history(
            history,
        )
        mlflow_logging.log_example_data(
            *next(data_generator_train), encoder=experiment.y_encoder
        )
        return {"history": history, "model": ensemble_model}


def train_plain_model(
    source_model: keras.models.Model, target_dataset: str, epochs: int = 10
) -> dict:
    with mlflow.start_run(run_name="plain model", nested=True):
        reading = Reading()
        X, y = reading.read_dataset(dataset=target_dataset)
        experiment = BaseExperiment(
            use_early_stopping=False,
        )
        model = experiment.clean_weights(source_model=source_model)

        input_len = model.input.shape[1]
        if input_len is None:
            input_len = 2**8
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={"min_length": input_len, "max_length": input_len},
            test_args={"min_length": input_len, "max_length": input_len},
        )

        history = model.fit(
            data_generator_train,
            epochs=epochs,
            validation_data=validation_data,
            use_multiprocessing=True,
        )
        mlflow_logging.log_confusion_matrix(
            *validation_data, classifier=model, y_encoder=experiment.y_encoder
        )
        history = {
            f"ensemble_no_weights_{key}": value
            for key, value in history.history.items()
        }
        mlflow_logging.log_history(
            history,
        )
        mlflow_logging.log_example_data(
            *next(data_generator_train), encoder=experiment.y_encoder
        )
        return {"history": history}


def main(category, component_experiment_id):
    mlflow.set_experiment("Transfer learning - same category, ensemble")
    mlflow.tensorflow.autolog(log_models=False)
    mlflow_logging = MlFlowLogging()  # pylint: disable=redefined-outer-name
    for target_dataset in Reading().return_datasets_for_category(category):
        with mlflow.start_run(run_name=f"Parent run - {target_dataset}"):
            ensemble_training_results = train_ensemble_model(
                category=category,
                target_dataset=target_dataset,
                component_experiment_id=component_experiment_id,
            )
            plain_training_results = train_plain_model(
                ensemble_training_results["model"], target_dataset=target_dataset
            )
            history = {
                **ensemble_training_results["history"],
                **plain_training_results["history"],
            }
            mlflow_logging.log_history(history)


main(category="ECG", component_experiment_id="861748084231733287")
