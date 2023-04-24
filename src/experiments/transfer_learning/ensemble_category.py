from typing import List
from typing import Literal

import mlflow
import numpy as np
from experiments.utils import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import Reading
from selecting import DBASelector
from tensorflow import keras

mlflow_logging = MlFlowLogging()


class EnsembleExperiment(BaseExperiment):
    name = "ensemble"

    def prepare_ensemble_model(
        self,
        source_models: List[keras.models.Model],
        compile_=True,
        mode=Literal["normal", "transfer_learning"],
    ) -> keras.models.Model:
        if mode == "normal":
            decay = self.normal_decay
        elif mode == "transfer_learning":
            decay = self.transfer_learning_decay
        else:
            raise KeyError('mode should be either "normal" or "transfer_learning"')
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
        if compile_:
            model.compile(
                loss="categorical_crossentropy",
                optimizer=decay,
                metrics=["accuracy"],
            )
        return model

    @staticmethod
    def read_component_model(
        dataset, component_experiment_id: str, root_path: str = "data/models/components"
    ) -> keras.models.Model:
        saving_path = f"{root_path}/{component_experiment_id}/dataset={dataset}"
        try:
            return keras.models.load_model(saving_path)
        except OSError as error:
            raise FileNotFoundError(
                f"Cannot find model for dataset {dataset} and experiment {component_experiment_id}"
            ) from error


def train_ensemble_model(
    target_dataset: str, component_experiment_id: str, epochs: int = 10
):
    reading = Reading()
    X, y = reading.read_dataset(dataset=target_dataset)
    experiment = EnsembleExperiment(model="fcn")
    selector = DBASelector()
    datasets = selector.select(dataset=target_dataset)

    mlflow.log_param(
        "Mean accuracy of models used for ensemble",
        experiment.get_accuracies_from_experiment(
            experiment_id=component_experiment_id, datasets=datasets
        ),
    )
    models = [
        experiment.read_component_model(
            dataset=dataset, component_experiment_id=component_experiment_id
        )
        for dataset in datasets
        if dataset != target_dataset
    ]
    data_generator_train, validation_data = experiment.prepare_generators(X, y)
    ensemble_model = experiment.prepare_ensemble_model(models)
    history = ensemble_model.fit(
        data_generator_train,
        epochs=epochs,
        validation_data=validation_data,
        use_multiprocessing=True,
    )
    history = experiment.log(
        data_generator_train=data_generator_train,
        validation_data=validation_data,
        model=ensemble_model,
        y_encoder=experiment.y_encoder,
        history=history.history,
    )
    return {"history": history, "model": ensemble_model}


def main(category, component_experiment_id):
    mlflow.set_experiment("Transfer learning - same category, ensemble")
    mlflow.tensorflow.autolog(log_models=False)
    for target_dataset in Reading().return_datasets_for_category(category):
        with mlflow.start_run(run_name=f"Parent run - {target_dataset}"):
            train_ensemble_model(
                target_dataset=target_dataset,
                component_experiment_id=component_experiment_id,
            )


main(category="ECG", component_experiment_id="861748084231733287")
