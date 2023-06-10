import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from mlflow.entities import Run
from preprocessing import get_lengths
from reading import Reading
from results.utils import cm
from results.utils import Results


class BaselineVsEnsembleResults(Results):
    approach_name = "ensemble approach vs baseline"
    distribution_names = ("ensemble", "baseline")
    results_root_path = os.path.dirname(__file__)
    first_result_key_name_loss = "ensemble_loss"
    first_result_key_name_val_loss = "ensemble_val_loss"
    first_result_key_name_acc = "ensemble_accuracy"
    first_result_key_name_val_acc = "ensemble_val_accuracy"
    second_result_key_name_loss = "baseline_loss"
    second_result_key_name_val_loss = "baseline_val_loss"
    second_result_key_name_acc = "baseline_accuracy"
    second_result_key_name_val_acc = "baseline_val_accuracy"
    x_label_win_tie_loss = "baseline approach"
    y_label_win_tie_loss = "ensemble approach"

    def __init__(
        self,
        first_experiment_id: str,
        second_experiment_id: str,
        component_experiment_id: str,
        assert_=True,
    ):
        super().__init__(
            first_experiment_id=first_experiment_id,
            second_experiment_id=second_experiment_id,
            assert_=assert_,
        )
        self.component_experiment_id = component_experiment_id
        self.X, self.y = Reading().read_dataset()

    def _get_second_experiment_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(
            self.second_experiment_id, exclude_from_name="Source"
        )
        return hist

    def _prepare_legend(self, text: str):
        mapping = {
            self.first_result_key_name_loss: "Ensemble - loss - train split",
            self.first_result_key_name_val_loss: "Ensemble - loss - validation \nsplit",
            self.first_result_key_name_acc: "Ensemble - accuracy - train split",
            self.first_result_key_name_val_acc: "Ensemble - accuracy - validation \nsplit",
            self.second_result_key_name_loss: "Baseline - loss - train split",
            self.second_result_key_name_val_loss: "Baseline - loss - validation \nsplit",
            self.second_result_key_name_acc: "Baseline - accuracy - train split",
            self.second_result_key_name_val_acc: "Baseline - accuracy - validation \nsplit",
        }
        return mapping[text]

    @staticmethod
    def _get_plot_kwargs(metric_name) -> dict:
        return {
            "color": "pink" if "ensemble" in metric_name.lower() else "blue",
            "linestyle": "--" if "train" in metric_name.lower() else "-",
        }

    def times_comparison(self):
        time_and_length = [
            (
                self._get_total_training_time_ensemble(dataset),
                self._get_total_training_time_baseline(dataset),
                self._get_series_total_length(dataset),
            )
            for dataset in self.datasets
        ]
        figure, ax = plt.subplots(figsize=(16 * cm, 14 * cm))
        x, y, colors = zip(*time_and_length)
        sc = ax.scatter(
            x,
            y,
            c=colors,
            cmap="plasma",
        )
        plt.colorbar(sc, ax=ax, label="Mean length of the series")

        lim = max(*x, *y)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.plot([0, 100], [0, 100], c="black")
        ax.set_ylabel("Training time for baseline approach")
        ax.set_xlabel("Training time for ensemble approach")
        ax.set_aspect("equal")
        ax.set_xticks(ax.get_xticks(), [f"{int(tick)} min" for tick in ax.get_xticks()])
        ax.set_yticks(ax.get_yticks(), [f"{int(tick)} min" for tick in ax.get_yticks()])
        self._save_fig(figure, "times_comparison.png")

    def _get_series_total_length(self, dataset_name):
        run = self.first_experiment_runs[dataset_name]
        source_datasets = run.data.params["Datasets used for ensemble"].split(",")
        all_datasets = source_datasets + [dataset_name]
        mask = [dataset.split("_")[0] in all_datasets for dataset in self.y.ravel()]
        mask = np.array(mask)
        return get_lengths(self.X[mask]).sum()

    def _get_total_training_time_ensemble(self, dataset):
        run = self.first_experiment_runs[dataset]
        finetuning_time = self.__extract_training_time(run)
        source_datasets = run.data.params["Datasets used for ensemble"].split(",")
        source_datasets = [name.strip(" ") for name in source_datasets]
        component_experiment = self._get_history_per_experiment(
            self.component_experiment_id
        )
        components_training_time = [
            self.__extract_training_time(component_experiment[source_dataset])
            for source_dataset in source_datasets
        ]
        return finetuning_time + sum(components_training_time)

    @staticmethod
    def __extract_training_time(run):
        return (run.info.end_time - run.info.start_time) / 1000 / 60

    def _get_total_training_time_baseline(self, dataset):
        target_run = self.second_experiment_runs[dataset]
        source_run = self._get_history_per_experiment(
            self.second_experiment_id,
            exclude_from_name="Destination",
            dataset_name_from_run_name=True,
        )[f"Source model: {dataset}"]
        return self.__extract_training_time(target_run) + self.__extract_training_time(
            source_run
        )


results = BaselineVsEnsembleResults(
    first_experiment_id="554900821027531839",
    second_experiment_id="743133642334170939",
    component_experiment_id="183382388301527558",
    assert_=False,
)
results.save_wilcoxon_test()
results.times_comparison()
results.get_mean_loss_acc_per_epoch()
results.win_tie_loss_diagram()
