import json
from typing import List
from typing import Literal

import mlflow
from experiments.utils import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import Reading
from selecting import DBASelector
from selecting import RandomSelector
from tensorflow import keras

mlflow_logging = MlFlowLogging()


class EnsembleExperiment(BaseExperiment):
    name = "ensemble"

    def prepare_ensemble_model(
        self,
        source_models: List[keras.models.Model],
        mode: Literal["normal", "transfer_learning"],
        compile_=True,
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
                optimizer=keras.optimizers.Adam(decay),
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
    target_dataset: str,
    selection_method: Literal["random", "similarity"],
    component_experiment_id: str,
    no_tr_experiment_id: str,
    epochs: int = 10,
    number_of_datasets: int = 5,
):
    reading = Reading()
    X, y = reading.read_dataset(dataset=target_dataset)
    experiment = EnsembleExperiment(model="fcn")

    if selection_method == "random":
        selector = RandomSelector()
    elif selection_method == "similarity":
        selector = DBASelector()
    else:
        raise KeyError()
    datasets = selector.select(dataset=target_dataset, size=number_of_datasets)

    mlflow.log_param(
        "Mean accuracy of models used for ensemble",
        experiment.get_mean_accuracies_from_experiment(
            experiment_id=component_experiment_id, datasets=datasets
        ),
    )
    models = [
        experiment.read_component_model(
            dataset=dataset, component_experiment_id=component_experiment_id
        )
        for dataset in datasets
    ]
    data_generator_train, validation_data = experiment.prepare_generators(X, y)
    ensemble_model = experiment.prepare_ensemble_model(models, mode="transfer_learning")
    history = ensemble_model.fit(
        data_generator_train,
        epochs=epochs,
        validation_data=validation_data,
        use_multiprocessing=True,
    )
    comparison_history = experiment.get_param_from_mlflow(
        experiment_id=no_tr_experiment_id,
        dataset=target_dataset,
        param="history.json",
        type_="artifact",
    )
    comparison_history = json.load(open(comparison_history))
    experiment.extended_log(
        data_generator_train=data_generator_train,
        validation_data=validation_data,
        model=ensemble_model,
        y_encoder=experiment.y_encoder,
        history=history.history,
        no_transfer_learning_history=comparison_history,
    )

    return {"history": history, "model": ensemble_model}


def main(
    category: str,
    selection_method: Literal["random", "similarity"],
    this_experiment_id: str,
    component_experiment_id: str,
    no_tr_experiment_id: str,
    number_of_datasets: int = 5,
):
    mlflow.set_experiment(experiment_id=this_experiment_id)
    mlflow.tensorflow.autolog(log_models=False)
    for target_dataset in Reading().return_datasets_for_category(category):
        print(f"Target dataset: {target_dataset}")
        with mlflow.start_run(run_name=target_dataset):
            train_ensemble_model(
                target_dataset=target_dataset,
                selection_method=selection_method,
                component_experiment_id=component_experiment_id,
                no_tr_experiment_id=no_tr_experiment_id,
                number_of_datasets=number_of_datasets,
            )


if __name__ == "main":
    main(
        category="MOTION",
        selection_method="similarity",
        this_experiment_id="554900821027531839",
        component_experiment_id="183382388301527558",
        no_tr_experiment_id="541913567164685548",
    )
