import logging
import os
from typing import Literal

import mlflow
from experiments import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import Reading
from tensorflow import keras

mlflow_logging = MlFlowLogging()


def train_component(
    dataset_name: str,
    experiment_id: str,
    saving_path: str,
    model: Literal["fcn", "encoder", "inception"],
) -> None:
    with mlflow.start_run(
        run_name=dataset_name, experiment_id=experiment_id, nested=True
    ):
        mlflow.tensorflow.autolog(log_models=False)
        experiment = BaseExperiment(saving_path=saving_path, use_early_stopping=False)
        X, y = Reading().read_dataset(dataset=dataset_name)
        train_gen, val_data = experiment.prepare_generators(
            X, y, train_args={"augmentation_probability": 0.3}
        )
        experiment.decay = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=100000,
            decay_rate=0.96,
        )
        if model == "fcn":
            model: keras.models.Model = experiment.prepare_FCN_model(scale=1)
        elif model == "encoder":
            model: keras.model.Model = experiment.prepare_encoder_classifier(2**8)
        else:
            raise NotImplementedError()
        history = model.fit(
            train_gen,
            epochs=10,
            callbacks=experiment.callbacks,
            validation_data=val_data,
            use_multiprocessing=True,
        )
        mlflow_logging.log_confusion_matrix(
            *val_data, classifier=model, y_encoder=experiment.y_encoder
        )
        mlflow_logging.log_history(
            history.history,
        )
        mlflow_logging.log_example_data(*next(train_gen), encoder=experiment.y_encoder)


def main(experiment_id: str, category: str):
    for dataset in Reading().return_datasets_for_category(category):
        saving_path = f"components/{experiment_id}/dataset={dataset}"
        if os.path.exists(os.path.join("data", "models", saving_path)):
            logging.info("Skipping %s", saving_path)
        else:
            train_component(
                dataset_name=dataset,
                experiment_id=experiment_id,
                saving_path=saving_path,
                model="encoder",
            )


main(experiment_id="103427775450294357", category="IMAGE")
main(experiment_id="103427775450294357", category="ECG")
