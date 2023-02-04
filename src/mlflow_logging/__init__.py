import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from keras.engine.training_v1 import Model
from sklearn.metrics import confusion_matrix

from preprocessing import plot


# logger = mlflow_logging.getLogger()


class MlFlowLogging:
    def __init__(
        self, artifacts_saving_root=os.getenv("ARTIFACT_SAVING_PATH", "./artifats")
    ):
        self.artifacts_saving_root = artifacts_saving_root

    def log_example_data(
        self,
        X,
        y,
        encoder: Optional[sklearn.preprocessing.OneHotEncoder] = None,
    ) -> None:
        if encoder:
            fig = plot(X, encoder.inverse_transform(y))
        else:
            fig = plot(X, y)
        mlflow.log_figure(fig, os.path.join(self.artifacts_saving_root, "train_data.png"))

    def log_history(
        self,
        history: dict,
    ):
        losses = {key for key in history if "loss" in key}
        accuracies = {key for key in history if "acc" in key}
        mlflow.log_text(json.dumps(history), "history.json")
        for key in losses:
            figure, ax = plt.subplots(figsize=(20, 20))
            ax.plot(history[key], data=key)
            figure.suptitle("Model loss")
            ax.set_ylabel("loss")
            ax.set_xlabel("epoch")
            ax.legend(loc="upper left")
            mlflow.log_figure(figure,  os.path.join(self.artifacts_saving_root,"loss.png"))
        for key in accuracies:
            figure, ax = plt.subplots(figsize=(20, 20))
            ax.plot(history[key], data=key)
            figure.suptitle("Model accuracy")
            ax.set_ylabel("accuracy")
            ax.set_xlabel("epoch")
            ax.legend(loc="upper left")
            mlflow.log_figure(figure,  os.path.join(self.artifacts_saving_root,"acc.png"))

    def log_confusion_matrix(
        self,
        X_true,
        y_true,
        classifier: Model,
        y_encoder: Optional[sklearn.preprocessing.OneHotEncoder] = None,
        figsize=(20, 20),
    ) -> None:
        y_predicted = classifier.predict(X_true)
        if y_encoder:
            y_predicted = y_encoder.inverse_transform(y_predicted)
            y_true = y_encoder.inverse_transform(y_true)
            labels = y_encoder.categories_[0]
        else:
            labels = None
        cm = confusion_matrix(y_true, y_predicted, labels=labels)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pd.DataFrame(cm, columns=labels, index=labels), ax=ax)
        mlflow.log_figure(fig,  os.path.join(self.artifacts_saving_root, "confusion_matrix.png"))

        np.fill_diagonal(cm, 0)
        idx_1d = cm.flatten().argsort()[:-10:-1]
        x_idx, y_idx = np.unravel_index(idx_1d, cm.shape)
        if labels is not None:
            misclassified_summary = ""
            for (
                x,
                y,
            ) in zip(x_idx, y_idx):
                misclassified_summary += (
                    f"True: {labels[x]}, predicted {labels[y]}, {cm[x, y]} times \n"
                )
            mlflow.log_text(misclassified_summary,  os.path.join(self.artifacts_saving_root, "confusion_matrix_summary.txt"))
