import mlflow
from tensorflow import keras

from experiments import BaseExperiment
from mlflow_logging import MlFlowLogging
from reading import ConcatenatedDataset

mlflow_logging = MlFlowLogging()


def train_component(dataset_name: str, experiment_id: str, saving_path: str) -> None:
    mlflow.set_experiment(experiment_id=experiment_id)
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
        history,
    )
    mlflow_logging.log_example_data(*next(train_gen), encoder=experiment.y_encoder)
