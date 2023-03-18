import logging
import os

import mlflow
from tensorflow import keras

from experiments import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import ConcatenatedDataset

mlflow_logging = MlFlowLogging()


def train_component(dataset_name: str, experiment_id: str, saving_path: str) -> None:
    with mlflow.start_run(
        run_name=dataset_name, experiment_id=experiment_id, nested=True
    ):
        mlflow.tensorflow.autolog(log_models=False)
        experiment = BaseExperiment(saving_path=saving_path, use_early_stopping=False)
        X, y = ConcatenatedDataset().read_dataset(dataset=dataset_name)
        train_gen, val_data = experiment.prepare_generators(
            X, y, train_args={"augmentation_probability": 0.3}
        )
        model: keras.models.Model = experiment.prepare_FCN_model(scale=0.5)
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
    for dataset in ConcatenatedDataset().return_datasets_for_category(category):
        saving_path = f"components/{experiment_id}/dataset={dataset}"
        if os.path.exists(os.path.join("data", "models", saving_path)):
            logging.info(f"Skipping {saving_path}")
        else:
            train_component(
                dataset_name=dataset,
                experiment_id=experiment_id,
                saving_path=saving_path,
            )


main(experiment_id="835719718053923699", category="IMAGE")
# TODO change decay for this
