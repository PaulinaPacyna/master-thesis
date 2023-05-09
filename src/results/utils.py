import json
import os
from abc import ABCMeta
from abc import abstractmethod
from collections import Counter
from typing import Dict
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
    def first_result_key_name_loss(self):
        pass

    @property
    @abstractmethod
    def first_result_key_name_val_loss(self):
        pass

    @property
    @abstractmethod
    def first_result_key_name_acc(self):
        pass

    @property
    @abstractmethod
    def first_result_key_name_val_acc(self):
        pass

    @property
    @abstractmethod
    def second_result_key_name_loss(self):
        pass

    @property
    @abstractmethod
    def second_result_key_name_val_loss(self):
        pass

    @property
    @abstractmethod
    def second_result_key_name_acc(self):
        pass

    @property
    @abstractmethod
    def second_result_key_name_val_acc(self):
        pass

    def __init__(
        self,
        first_experiment_id: str,
        second_experiment_id: str,
        assert_: bool = True,
    ):
        self.second_experiment_id = second_experiment_id
        self.first_experiment_id = first_experiment_id
        self.client = MlflowClient()
        self.first_experiment_runs: Dict[str, Run] = self._get_first_experiment_runs()
        self.second_experiment_runs: Dict[str, Run] = self._get_second_experiment_runs()
        if assert_:
            self._assert_histories_equal()
        self.datasets = self._get_common_datasets()
        matplotlib.rc("font", size=12)

    def _get_second_experiment_runs(self) -> Dict[str, Run]:
        return self._get_history_per_experiment(self.second_experiment_id)

    def _get_first_experiment_runs(self) -> Dict[str, Run]:
        return self._get_history_per_experiment(self.first_experiment_id)

    def _get_history_per_experiment(self, experiment_id) -> Dict[str, Run]:
        runs = self.client.search_runs([experiment_id])
        for run in runs:
            if run.info.status == "FINISHED" and run.info.lifecycle_stage == "active":
                run.data.metrics["history"] = json.load(
                    open(run.info.artifact_uri + "/history.json")
                )
                assert min([len(x) for x in run.data.metrics["history"].values()]) == 10
        datasets_counts = Counter([run.data.params["dataset_train"] for run in runs])
        for dataset_name in datasets_counts:
            if datasets_counts[dataset_name] > 1:
                raise ValueError(
                    f"More than one experiment for {dataset_name} "
                    f"for {self.first_experiment_id}"
                )
        return {run.data.params["dataset_train"]: run for run in runs}

    def _assert_histories_equal(self):
        second_datasets = set(self.second_experiment_runs.keys())
        first_datasets = set(self.first_experiment_runs.keys())
        if second_datasets != first_datasets:
            raise ValueError(
                f"The following datasets missing for {self.first_experiment_id}: "
                f"{second_datasets-first_datasets}.\n"
                f"The following datasets missing for {self.second_experiment_id}: "
                f"{first_datasets-second_datasets}.\n"
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
        metrics_names_1 = [
            self.first_result_key_name_acc,
            self.first_result_key_name_val_acc,
            self.first_result_key_name_loss,
            self.first_result_key_name_val_loss,
        ]
        metrics_names_2 = [
            self.second_result_key_name_acc,
            self.second_result_key_name_val_acc,
            self.second_result_key_name_loss,
            self.second_result_key_name_val_loss,
        ]
        metrics_per_epoch = {
            self._prepare_legend(metric): []
            for metric in metrics_names_1 + metrics_names_2
        }
        for dataset in self.datasets:
            history_1 = self.first_experiment_runs[dataset].data.metrics["history"]
            history_2 = self.second_experiment_runs[dataset].data.metrics["history"]
            for metric_name in metrics_names_1:
                metrics_per_epoch[self._prepare_legend(metric_name)].append(
                    history_1[metric_name]
                )
            for metric_name in metrics_names_2:
                metrics_per_epoch[self._prepare_legend(metric_name)].append(
                    history_2[metric_name]
                )
        history_summarized = {
            metric: np.array(metrics_per_epoch[metric]).mean(0)
            for metric in metrics_per_epoch
        }
        return history_summarized

    def win_tie_loss_diagram(self, epoch):
        epoch_acc_pairs = [
            [
                self.second_experiment_runs[dataset].data.metrics["history"][
                    self.second_result_key_nameval_acc
                ][epoch - 1],
                self.first_experiment_runs[dataset].data.metrics["history"][
                    self.first_result_key_name_val_acc
                ][epoch - 1],
            ]
            for dataset in self.datasets
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

    def _get_common_datasets(self) -> list:
        first_datasets = self.first_experiment_runs.keys()
        second_datasets = self.second_experiment_runs.keys()
        return list(set(first_datasets).intersection(second_datasets))
