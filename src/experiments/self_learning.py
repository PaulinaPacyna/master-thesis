import os

import mlflow
from sklearn.model_selection import train_test_split
import numpy as np
from experiments import BaseExperiment
from mlflow_logging import MlFlowLogging
from preprocessing.fit_generator import (
    SelfLearningDataGenerator,
    ConstantLengthDataGenerator,
)
from reading import ConcatenatedDataset

mlflow_logging = MlFlowLogging()

class SelfLearningExperiment(BaseExperiment):

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


def train_self_learning(dataset, category, number_of_epochs=10):
    with mlflow.start_run(run_name="self learning", nested=True):
        concatenated_dataset = ConcatenatedDataset()
        X, y = concatenated_dataset.read_dataset(dataset=dataset)
        X_self_learning, _ = concatenated_dataset.read_dataset(
            category=category, exclude_dataset=dataset
        )
        experiment = SelfLearningExperiment(use_early_stopping=False)
        self_learning_params = {
                "X_self_learning": X_self_learning,
                "self_learning_cold_start": 2,
                "self_learning_threshold": 0.95,
            }
        mlflow.log_params(self_learning_params)
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args=self_learning_params,
        )
        model = experiment.prepare_FCN_model()
        data_generator_train.add_model(model)
        history = model.fit(
            data_generator_train,
            epochs=number_of_epochs,
            validation_data=validation_data,
            callbacks=experiment.callbacks,
        )
        history = {f"self_learning_{key}": value for key, value in history.history.items()}
        mlflow_logging.log_history(history)
        mlflow_logging.log_confusion_matrix(*validation_data, classifier=model, y_encoder=experiment.y_encoder)
        mlflow_logging.log_example_data(*next(data_generator_train), encoder=experiment.y_encoder)
        return {"history": history}


def train_fcn(dataset, number_of_epochs=10):
    with mlflow.start_run(run_name="plain", nested=True):
        concatenated_dataset = ConcatenatedDataset()
        X, y = concatenated_dataset.read_dataset(dataset=dataset)
        experiment = BaseExperiment(use_early_stopping=False)
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
        )
        model = experiment.prepare_FCN_model()
        data_generator_train.add_model(model)
        history = model.fit(
            data_generator_train,
            epochs=number_of_epochs,
            validation_data=validation_data,
            callbacks=experiment.callbacks,
        )

        history = {f"plain_{key}": value for key, value in history.history.items()}
        mlflow_logging.log_history(history)
        mlflow_logging.log_confusion_matrix(*validation_data, classifier=model, y_encoder=experiment.y_encoder)
        mlflow_logging.log_example_data(*next(data_generator_train), encoder=experiment.y_encoder)
        return {"history": history}

if __name__ == "__main__":
    os.chdir("..")
    mlflow.set_experiment("Self learning - FCN")
    dataset = "ECG200"
    category = "ECG"
    with mlflow.start_run(run_name="test"):
        self_learning_results = train_self_learning(dataset=dataset, category=category)
        plain_results = train_fcn(dataset=dataset)
        history = {**self_learning_results["history"], **plain_results["history"]}
        mlflow_logging.log_history(history)