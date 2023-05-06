import json
from abc import ABC
from collections import Counter
from typing import List
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mlflow import MlflowClient
from mlflow.entities import Run


class Results(ABC):
    def __init__(
        self,
        transfer_learning_experiment_id: str,
        no_transfer_learning_experiment_id: str,
    ):
        self.no_transfer_learning_experiment_id = no_transfer_learning_experiment_id
        self.transfer_learning_experiment_id = transfer_learning_experiment_id
        self.client = MlflowClient()
        self.transfer_learning_runs: List[Run] = self.get_transfer_learning_runs()
        self.no_transfer_learning_runs: List[Run] = self.get_no_transfer_learning_runs()
        self.assert_histories_equal()
        matplotlib.rc("font", size=12)

    def get_no_transfer_learning_runs(self):
        return self._get_history_per_experiment(self.no_transfer_learning_experiment_id)

    def get_transfer_learning_runs(self):
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

    def assert_histories_equal(self):
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


class BaselineResults(Results):
    results_root_path = "results/baseline"
    transfer_learning_key_name_loss = "baseline_loss"
    transfer_learning_key_name_val_loss = "baseline_val_loss"
    transfer_learning_key_name_acc = "baseline_accuracy"
    transfer_learning_key_name_val_acc = "baseline_val_accuracy"
    no_transfer_learning_key_name_loss = "baseline_no_transfer_learning_base_loss"
    no_transfer_learning_key_name_val_loss = (
        "baseline_no_transfer_learning_base_val_loss"
    )
    no_transfer_learning_key_name_acc = "baseline_no_transfer_learning_base_accuracy"
    no_transfer_learning_key_name_val_acc = (
        "baseline_no_transfer_learning_base_val_accuracy"
    )

    def get_transfer_learning_runs(self):
        hist = self._get_history_per_experiment(self.transfer_learning_experiment_id)
        return [run for run in hist if run.info.run_name.startswith("Destination")]

    def prepare_legend(self, text: str):
        mapping = {
            self.transfer_learning_key_name_loss: "Loss - train split",
            self.transfer_learning_key_name_val_loss: "Loss - validation split",
            self.transfer_learning_key_name_acc: "Accuracy - train split",
            self.transfer_learning_key_name_val_acc: "Accuracy - validation split",
            self.no_transfer_learning_key_name_loss: "No transfer learning - loss - train split",
            self.no_transfer_learning_key_name_val_loss: "No transfer learning - loss - validation split",
            self.no_transfer_learning_key_name_acc: "No transfer learning - accuracy - train split",
            self.no_transfer_learning_key_name_val_acc: "No transfer learning - accuracy - validation split",
        }
        return mapping[text]

    def get_mean_loss_acc_per_epoch(self, metric: Literal["loss", "acc"]):
        history_summarized = self.get_history_summarized_per_epoch()
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
                axes=ax,
            )
        figure.suptitle(f"Model {full_metric_name}")
        ax.set_ylabel(full_metric_name)
        ax.set_xlabel("epoch")
        ax.legend()
        plt.ylim(bottom=0)
        plt.savefig(f"{self.results_root_path}/{metric}.png")
        plt.close(figure)

    def get_history_summarized_per_epoch(self):
        metrics_names = self.transfer_learning_runs[0].data.metrics["history"].keys()
        metrics_per_epoch = {
            self.prepare_legend(metric): [] for metric in metrics_names
        }
        for run in self.transfer_learning_runs:
            history = run.data.metrics["history"]
            for metric_name in metrics_names:
                metrics_per_epoch[self.prepare_legend(metric_name)].append(
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
        figure.suptitle(f"Win/tie/lose plot (epoch {epoch})")
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
        plt.savefig(f"{self.results_root_path}/win_tie_lose_epoch_{epoch}.png")
        plt.close(figure)


results = BaselineResults(
    transfer_learning_experiment_id="743133642334170939",
    no_transfer_learning_experiment_id="183382388301527558",
)
results.get_mean_loss_acc_per_epoch("loss")
results.get_mean_loss_acc_per_epoch("acc")
results.win_tie_loss_diagram(epoch=10)
results.win_tie_loss_diagram(epoch=5)

# 0e3a43fd4ff7438fa058e34ecd41976e 94ae7f67fd5d47e5bb619f387fbf999e a9ecfeec183840a081003d948760d0b9 b724266be26945ed879a45cf77c1b54765e051337fdb469e8a34ecbefdbadacf 9d4d895e6bbd4d56924895e93ff28aeb 31f3811c731c4808bc7e62b8113514fe b731659de4aa4e8bb97268318650b6b3 50446ee07c424506b98f8c3e8d9c8984 b4199b445dc84c78a94c1b96a89ab921 b4199b445dc84c78a94c1b96a89ab921 88f9a2e41b62486db84eecdff8afa1fe 7a43259282a54d4fa428f684b72cb096 85bd436c8d594500a17be073e0dd7a01 c79534a6b7674e1e81bcdc4a82db8f63 52aa18868d914075a00869d84d873408
