from typing import Literal

import mlflow
from experiments.transfer_learning.ensemble_dba_selection_5 import EnsembleExperiment
from reading import Reading


def train_plain_model(
    target_dataset: str,
    model: Literal["fcn", "encoder"],
    number_of_models: int = 5,
    epochs: int = 10,
) -> dict:
    with mlflow.start_run(run_name=target_dataset, nested=True):
        reading = Reading()
        X, y = reading.read_dataset(dataset=target_dataset)
        experiment = EnsembleExperiment(
            model=model,
        )
        data_generator_train, validation_data = experiment.prepare_generators(X, y)

        model = experiment.prepare_ensemble_model(
            source_models=[
                experiment.prepare_classifier() for _ in range(number_of_models)
            ],
            mode="normal",
        )

        history = model.fit(
            data_generator_train,
            epochs=epochs,
            validation_data=validation_data,
            use_multiprocessing=True,
        )
        experiment.log(
            data_generator_train=data_generator_train,
            validation_data=validation_data,
            model=model,
            history=history.history,
            y_encoder=experiment.y_encoder,
        )


def main(category: str, model: Literal["fcn", "encoder"], experiment_id: str):
    mlflow.set_experiment(experiment_id=experiment_id)
    for dataset in Reading().return_datasets_for_category(category):
        train_plain_model(target_dataset=dataset, model=model)


if __name__ == "__main__":
    main(category="MOTION", model="fcn", experiment_id="541913567164685548")
