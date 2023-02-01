import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from keras.engine.training_v1 import Model
from sklearn.metrics import confusion_matrix

from preprocessing import plot


# logger = logging.getLogger()


def read_dataset(
    root_data_path: str = "./data",
    category: str = None,
    dataset: str = None,
    return_cat: bool = False,
    logging_call: callable = None,
) -> List[np.array]:
    X: np.array = np.load(f"{root_data_path}/X.npy", allow_pickle=True)
    y: np.array = np.load(f"{root_data_path}/y.npy")
    categories: np.array = np.load(f"{root_data_path}/categories.npy")
    if category and not dataset:
        logging.info("Loading only one category: %s", category)
        mask = (categories == category).reshape(-1)
        y = y[mask, :]
        X = X[mask]
        if logging_call:
            logging_call("category", category)
        return X, y
    if dataset:
        logging.info("Loading only one dataset: %s", dataset)
        mask = (np.char.startswith(y, dataset)).reshape(-1)
        y = y[mask, :]
        X = X[mask]
        if logging_call:
            logging_call("dataset", dataset)
        return X, y
    if logging_call:
        logging_call("y.unique", ", ".join(np.unique(y))[:500])
    if return_cat:
        return X, y, categories
    return X, y


def log_example_data(
    X,
    y,
    encoder: Optional[sklearn.preprocessing.OneHotEncoder] = None,
    logging_call: Optional[callable] = None,
    path="train_data.png",
) -> None:
    if encoder:
        fig = plot(X, encoder.inverse_transform(y))
    else:
        fig = plot(X, y)
    logging_call(fig, path)


def log_history(
    history: dict,
    logging_figures_call: callable,
    logging_param_call: Optional[callable] = None,
):
    losses = {key for key in history if "loss" in key}
    accuracies = {key for key in history if "acc" in key}
    if logging_param_call:
        logging_param_call("history", history)
    figure, ax = plt.subplots(figsize=(20, 20))
    for key in losses:
        ax.plot(history[key], data=key)
        figure.suptitle("Model loss")
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        ax.legend(loc="upper left")
        logging_figures_call(figure, "loss.png")
    for key in accuracies:
        ax.plot(history[key], data=key)
        figure.suptitle("Model accuracy")
        ax.set_ylabel("accuracy")
        ax.set_xlabel("epoch")
        ax.legend(loc="upper left")
        logging_figures_call(figure, "acc.png")


def log_confusion_matrix(
    X_true,
    y_true,
    classifier: Model,
    logging_figures_call: callable,
    logging_text_call: Optional[callable],
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
    logging_figures_call(fig, "confusion_matrix.png")
    if logging_text_call:
        np.fill_diagonal(cm, 0)
        idx_1d = cm.flatten().argsort()[:-10:-1]
        x_idx, y_idx = np.unravel_index(idx_1d, cm.shape)
        misclassified_summary = ""
        for (
            x,
            y,
        ) in zip(x_idx, y_idx):
            misclassified_summary += (
                f"True: {labels[x]}, predicted {labels[y]}, {cm[x, y]} times \n"
            )
        logging_text_call(misclassified_summary, "confusion_matrix_summary.txt")
