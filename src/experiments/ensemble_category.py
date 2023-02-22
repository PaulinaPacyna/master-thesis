from typing import List
import numpy as np
import mlflow
from tensorflow import keras

from experiments import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import ConcatenatedDataset


class EnsembleExperiment(BaseExperiment):
    def prepare_ensemble_model(
        self, source_models: List[keras.models.Model], target_number_of_classes: int
    ) -> keras.models.Model:
        first = keras.layers.Input(source_models[0].inputs[0].shape[1:])
        outputs = []
        for model in source_models:
            model = self.swap_last_layer(
                model, number_of_classes=target_number_of_classes, compile=False
            )
            model = model(first)
            outputs.append(model)
        last = keras.layers.Add()(outputs) / len(source_models)
        model = keras.models.Model(inputs=first, outputs=last)
        return model


def train_ensemble_model(target_dataset: str, category: str, epochs: int = 10):
    with mlflow.start_run(run_name="ensemble", nested=True):
        concatenated_dataset = ConcatenatedDataset()
        X, y = concatenated_dataset.read_dataset(dataset=target_dataset)
        all_datasets = concatenated_dataset.return_datasets_for_category(
            category=category
        )
        datasets = np.random.choice(all_datasets, 5, False)
        mlflow.log_param("Datasets used for ensemble", ", ".join(datasets))
        experiment = EnsembleExperiment(
            saving_path=f"encoder_ensemble/ensemble/dataset={target_dataset}",
            use_early_stopping=False,
        )
        models = [
            keras.models.load_model(
                f"data/models/encoder_same_cat_other_datasets/dest_plain/dataset={dataset}"
            )
            for dataset in datasets
            if dataset != target_dataset
        ]
        input_len = models[0].input.shape[1]
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={"min_length": input_len, "max_length": input_len},
            test_args={"min_length": input_len, "max_length": input_len},
        )
        ensemble_model = experiment.prepare_ensemble_model(
            models, len(experiment.y_encoder.categories_[0])
        )
        ensemble_model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(experiment.decay),
            metrics=["accuracy"],
        )
        history = ensemble_model.fit(
            data_generator_train, epochs=epochs, validation_data=validation_data
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
        concatenated_dataset = ConcatenatedDataset()
        X, y = concatenated_dataset.read_dataset(dataset=target_dataset)
        experiment = BaseExperiment(
            saving_path=f"encoder_ensemble/plain/dataset={target_dataset}",
            use_early_stopping=False,
        )
        model = experiment.clean_weights(source_model=source_model)

        input_len = model.input.shape[1]
        data_generator_train, validation_data = experiment.prepare_generators(
            X,
            y,
            train_args={"min_length": input_len, "max_length": input_len},
            test_args={"min_length": input_len, "max_length": input_len},
        )

        history = model.fit(
            data_generator_train, epochs=epochs, validation_data=validation_data
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


mlflow_logging = MlFlowLogging()

if __name__ == "__main__":
    mlflow.set_experiment("Transfer learning - same category, ensemble")
    mlflow.tensorflow.autolog()
    category = "ECG"
    target_dataset = "CinCECGTorso"
    with mlflow.start_run(run_name=target_dataset):
        ensemble_training_results = train_ensemble_model(
            category=category, target_dataset=target_dataset
        )
        plain_training_results = train_plain_model(
            ensemble_training_results["model"], target_dataset=target_dataset
        )
        history = {**ensemble_training_results["history"]}
        mlflow_logging.log_history(history)
