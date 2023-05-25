import json
import os
from abc import ABCMeta
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict
from typing import Literal

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.entities import Run

cm = 1 / 2.54


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
        matplotlib.rc("font", size=9)
        plt.rcParams["figure.dpi"] = 400

    def _save_fig(self, fig: Figure, path: str):
        latex_dir = os.path.join(
            Path(os.path.abspath(".")).parent, "latex", "2. thesis", "imgs"
        )
        dirname = self.results_root_path.split("/")[-1]
        fig.savefig(os.path.join(self.results_root_path, path), transparent=True)
        fig.savefig(os.path.join(latex_dir, dirname, path), transparent=True)
        plt.close(fig)

    def _get_second_experiment_runs(self) -> Dict[str, Run]:
        return self._get_history_per_experiment(self.second_experiment_id)

    def _get_first_experiment_runs(self) -> Dict[str, Run]:
        return self._get_history_per_experiment(self.first_experiment_id)

    def _get_history_per_experiment(
        self,
        experiment_id,
        add_prefix="",
        exclude_from_name=None,
        dataset_name_from_run_name=False,
    ) -> Dict[str, Run]:
        result = []
        for run in self.client.search_runs([experiment_id]):
            if run.info.status == "FINISHED" and run.info.lifecycle_stage == "active":
                if not exclude_from_name or (
                    exclude_from_name not in run.info.run_name
                ):
                    history = json.load(open(run.info.artifact_uri + "/history.json"))
                    history = {
                        add_prefix + key: value for key, value in history.items()
                    }
                    run.data.metrics["history"] = history
                    assert (
                        min([len(x) for x in run.data.metrics["history"].values()])
                        == 10
                    )
                    result.append(run)
        if dataset_name_from_run_name:
            datasets_counts = Counter([run.info.run_name for run in result])
            for dataset_name in datasets_counts:
                if datasets_counts[dataset_name] > 1:
                    raise ValueError(
                        f"More than one experiment for {dataset_name} "
                        f"for {self.first_experiment_id}"
                    )
            return {run.info.run_name: run for run in result}
        datasets_counts = Counter([run.data.params["dataset_train"] for run in result])
        for dataset_name in datasets_counts:
            if datasets_counts[dataset_name] > 1:
                raise ValueError(
                    f"More than one experiment for {dataset_name} "
                    f"for {self.first_experiment_id}"
                )
        return {run.data.params["dataset_train"]: run for run in result}

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

    def get_mean_loss_acc_per_epoch(self):
        figure, axes = plt.subplots(1, 2, figsize=(18 * cm, 13 * cm))
        figure.tight_layout(rect=[0.025, 0.01, 0.975, 0.95])
        figure.suptitle(f"Model accuracy and loss - {self.approach_name} approach")
        for metric, ax in zip(["loss", "acc"], axes):
            self._get_mean_loss_ax_acc_per_epoch(metric=metric, ax=ax)
        self._save_fig(figure, "loss_acc.png")
        plt.close(figure)

    def _get_mean_loss_ax_acc_per_epoch(self, metric: Literal["loss", "acc"], ax: Axis):
        history_summarized = self._get_history_summarized_per_epoch()
        history_summarized = {
            key: history_summarized[key]
            for key in history_summarized
            if metric in key.lower()
        }
        full_metric_name = "accuracy" if metric == "acc" else metric
        for metric_name in sorted(history_summarized):
            plot_kwargs = self._get_plot_kwargs(metric_name)
            ax.plot(
                np.arange(len(history_summarized[metric_name])) + 1,
                history_summarized[metric_name],
                label=metric_name,
                **plot_kwargs,
            )
        ax.set_ylabel(full_metric_name)
        ax.set_xlabel("epoch")
        if metric == "loss":
            plt.ylim(bottom=0)
        else:
            plt.ylim([0, 1])
        ax.legend()
        return ax

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

    def win_tie_loss_diagram(self):
        figure, axes = plt.subplots(1, 2, figsize=(18 * cm, 10 * cm))
        figure.tight_layout(rect=[0.025, 0.01, 0.975, 0.95])
        figure.suptitle(f"Win/tie/lose plot - {self.approach_name} approach")
        for epoch, ax in zip([5, 10], axes):
            self._win_tie_loss_diagram(epoch=epoch, ax=ax)
        self._save_fig(figure, "win_tie_lose_epoch.png")
        plt.close(figure)

    def _win_tie_loss_diagram(self, epoch: int, ax):
        epoch_acc_pairs = [
            [
                self.second_experiment_runs[dataset].data.metrics["history"][
                    self.second_result_key_name_val_acc
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
        ax.scatter(*list(zip(*epoch_acc_pairs)), s=8)
        ax.plot([-10, 10], [-10, 10], color="black")
        ax.set_title(f"Epoch {epoch}")
        ax.set_ylabel("With transfer learning")
        ax.set_xlabel("Without transfer learning")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        ax.text(
            0.1,
            0.9,
            f"Win / tie / loss\n{win} / {tie} / {lose}",
            bbox={"ec": "black", "color": "white"},
        )
        return ax

    def _get_common_datasets(self) -> list:
        first_datasets = self.first_experiment_runs.keys()
        second_datasets = self.second_experiment_runs.keys()
        return list(set(first_datasets).intersection(second_datasets))

    @staticmethod
    def _get_plot_kwargs(metric_name) -> dict:
        return {
            "color": "red" if "train" in metric_name else "green",
            "linestyle": "--" if "no transfer" in metric_name.lower() else "-",
        }

    def dba_vs_accuracy(self) -> dict:
        results = self.first_experiment_runs
        sim_acc_pairs = [
            (
                results[dataset].data.metrics["Mean DBA similarity"],
                results[dataset].data.metrics["val_accuracy"],
            )
            for dataset in self.datasets
        ]
        figure, ax = plt.subplots(figsize=(14 * cm, 14 * cm))
        plt.scatter(*list(zip(*sim_acc_pairs)), s=8)
        figure.suptitle("Accuracy versus mean DBA similarity of source datasets")
        ax.set_ylabel("Accuracy - validation split")
        ax.set_xlabel("Mean DBA similarity of source datasets to target dataset")
        plt.ylim([0, 1])
        plt.xscale("log")
        self._save_fig(figure, "accuracy_vs_mean_dba_sim.png")
