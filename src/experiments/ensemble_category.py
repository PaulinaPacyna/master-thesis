from typing import List

from tensorflow import keras

from experiments import BaseExperiment
from reading import ConcatenatedDataset
import tensorflow as tf


class EnsembleExperiment(BaseExperiment):
    @staticmethod
    def prepare_ensemble_model(
        source_models: List[keras.models.Model], target_number_of_classes: int
    ) -> keras.models.Model:
        first = keras.layers.Input(source_models[0].inputs[0].shape[1:])
        outputs = []
        for model in source_models:
            model = experiment.swap_last_layer(
                model, number_of_classes=target_number_of_classes, compile=False
            )
            model = model(first)
            outputs.append(model)
        last = keras.layers.Add()(outputs)
        model = keras.models.Model(inputs=first, outputs=last)
        return model


if __name__ == "__main__":
    category = "ECG"
    target_dataset = "CinCECGTorso"
    concatenated_dataset = ConcatenatedDataset()
    X, y = concatenated_dataset.read_dataset(dataset=target_dataset)
    experiment = EnsembleExperiment(
        saving_path="encoder_ensemble/category={category}/dataset={dataset}"
    )
    datasets = concatenated_dataset.return_datasets_for_category(category=category)
    models = []
    for dataset in datasets:
        if dataset != target_dataset:
            models += [
                keras.models.load_model(
                    f"data/models/encoder_same_cat_other_datasets/dest_plain/dataset={dataset}"
                )
            ]
    input_len = models[0].input.shape[1]
    train_gen, val_data = experiment.prepare_generators(X, y, train_args={"min_length": input_len, "max_length": input_len})
    ensemble_model = experiment.prepare_ensemble_model(
        models, len(experiment.y_encoder.categories_[0])
    )
    ensemble_model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(experiment.decay),
        metrics=["accuracy"],
    )
    history = ensemble_model.fit(train_gen, epochs=2, validation_data=val_data)
