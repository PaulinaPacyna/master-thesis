import json
import os
from abc import ABCMeta
from abc import abstractmethod
from collections import Counter
from typing import List
from typing import Literal

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mlflow import MlflowClient
from mlflow.entities import Run


class Results(metaclass=ABCMeta):
    @property
    @abstractmethod
    def results_root_path(self):
        pass

    @property
    @abstractmethod
    def approach_name(self):
        pass

    @property
    @abstractmethod
    def transfer_learning_key_name_loss(self):
        pass

    @property
    @abstractmethod
    def transfer_learning_key_name_val_loss(self):
        pass

    @property
    @abstractmethod
    def transfer_learning_key_name_acc(self):
        pass

    @property
    @abstractmethod
    def transfer_learning_key_name_val_acc(self):
        pass

    @property
    @abstractmethod
    def no_transfer_learning_key_name_loss(self):
        pass

    @property
    @abstractmethod
    def no_transfer_learning_key_name_val_loss(self):
        pass

    @property
    @abstractmethod
    def no_transfer_learning_key_name_acc(self):
        pass

    @property
    @abstractmethod
    def no_transfer_learning_key_name_val_acc(self):
        pass

    def __init__(
        self,
        transfer_learning_experiment_id: str,
        no_transfer_learning_experiment_id: str,
    ):
        self.no_transfer_learning_experiment_id = no_transfer_learning_experiment_id
        self.transfer_learning_experiment_id = transfer_learning_experiment_id
        self.client = MlflowClient()
        self.transfer_learning_runs: List[Run] = self._get_transfer_learning_runs()
        self.no_transfer_learning_runs: List[
            Run
        ] = self._get_no_transfer_learning_runs()
        self._assert_histories_equal()
        matplotlib.rc("font", size=12)

    def _get_no_transfer_learning_runs(self):
        return self._get_history_per_experiment(self.no_transfer_learning_experiment_id)

    def _get_transfer_learning_runs(self):
        return self._get_history_per_experiment(self.transfer_learning_experiment_id)

    def _get_history_per_experiment(self, experiment_id):
        runs = self.client.search_runs([experiment_id])
        for run in runs:
            if run.info.status == "FINISHED" and run.info.lifecycle_stage == "active":
                run.data.metrics["history"] = json.load(
                    open(run.info.artifact_uri + "/history.json")
                )
                assert min([len(x) for x in run.data.metrics["history"].values()]) == 10
        return runs

    def _assert_histories_equal(self):
        transfer_learning_datasets = [
            run.data.params["dataset_train"] for run in self.transfer_learning_runs
        ]
        no_transfer_learning_datasets = [
            run.data.params["dataset_train"] for run in self.transfer_learning_runs
        ]
        transfer_learning_datasets_counts = Counter(transfer_learning_datasets)
        for dataset_name in transfer_learning_datasets_counts:
            if transfer_learning_datasets_counts[dataset_name] > 1:
                raise ValueError(
                    f"More than one experiment for {dataset_name} "
                    f"for {self.transfer_learning_experiment_id}"
                )
        del transfer_learning_datasets_counts
        no_transfer_learning_datasets_counts = Counter(no_transfer_learning_datasets)
        for dataset_name in no_transfer_learning_datasets_counts:
            if no_transfer_learning_datasets_counts[dataset_name] > 1:
                raise ValueError(
                    f"More than one experiment for {dataset_name} "
                    f"for {self.no_transfer_learning_experiment_id}"
                )

        del no_transfer_learning_datasets_counts
        no_transfer_learning_datasets = set(no_transfer_learning_datasets)
        transfer_learning_datasets = set(transfer_learning_datasets)
        if no_transfer_learning_datasets != transfer_learning_datasets:
            raise ValueError(
                f"The following datasets missing for {self.transfer_learning_experiment_id}: "
                f"{no_transfer_learning_datasets-transfer_learning_datasets}.\n"
                f"The following datasets missing for {self.no_transfer_learning_experiment_id}: "
                f"{transfer_learning_datasets-no_transfer_learning_datasets}.\n"
            )

    @abstractmethod
    def _prepare_legend(self, text):
        pass

    def get_mean_loss_acc_per_epoch(self, metric: Literal["loss", "acc"]):
        history_summarized = self._get_history_summarized_per_epoch()
        history_summarized = {
            key: history_summarized[key]
            for key in history_summarized
            if metric in key.lower()
        }
        full_metric_name = "accuracy" if metric == "acc" else metric
        figure, ax = plt.subplots(figsize=(5.5, 5.5))
        for metric_name in sorted(history_summarized):
            plt.plot(
                np.arange(len(history_summarized[metric_name])) + 1,
                history_summarized[metric_name],
                label=metric_name,
                color="red" if "train" in metric_name else "green",
                linestyle="--" if "no transfer" in metric_name.lower() else "-",
                axes=ax,
            )
        figure.suptitle(f"Model {full_metric_name} - {self.approach_name} approach")
        ax.set_ylabel(full_metric_name)
        ax.set_xlabel("epoch")
        ax.legend()
        plt.ylim(bottom=0)
        plt.savefig(os.path.join(self.results_root_path, f"{metric}.png"))
        plt.close(figure)

    def _get_history_summarized_per_epoch(self):
        metrics_names = self.transfer_learning_runs[0].data.metrics["history"].keys()
        metrics_per_epoch = {
            self._prepare_legend(metric): [] for metric in metrics_names
        }
        for run in self.transfer_learning_runs:
            history = run.data.metrics["history"]
            for metric_name in metrics_names:
                metrics_per_epoch[self._prepare_legend(metric_name)].append(
                    history[metric_name]
                )
        history_summarized = {
            metric: np.array(metrics_per_epoch[metric]).mean(0)
            for metric in metrics_per_epoch
        }
        return history_summarized

    def win_tie_loss_diagram(self, epoch):
        epoch_acc_pairs = [
            [
                run.data.metrics["history"][self.no_transfer_learning_key_name_val_acc][
                    epoch - 1
                ],
                run.data.metrics["history"][self.transfer_learning_key_name_val_acc][
                    epoch - 1
                ],
            ]
            for run in self.transfer_learning_runs
        ]
        win = sum(acc[0] < acc[1] for acc in epoch_acc_pairs)
        tie = sum(acc[0] == acc[1] for acc in epoch_acc_pairs)
        lose = sum(acc[0] > acc[1] for acc in epoch_acc_pairs)
        figure, ax = plt.subplots(figsize=(5.5, 5.5))
        plt.scatter(*list(zip(*epoch_acc_pairs)), s=8)
        plt.plot([-10, 10], [-10, 10], color="black")
        figure.suptitle(
            f"Win/tie/lose plot - {self.approach_name} approach (epoch {epoch})"
        )
        ax.set_ylabel("With transfer learning")
        ax.set_xlabel("Without transfer learning")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        ax.set_aspect("equal")
        ax.text(
            0.1,
            0.9,
            f"Win / tie / loss\n{win} / {tie} / {lose}",
            bbox={"ec": "black", "color": "white"},
        )
        plt.savefig(
            os.path.join(self.results_root_path, f"win_tie_lose_epoch_{epoch}.png")
        )
        plt.close(figure)
