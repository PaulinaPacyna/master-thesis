import os
from typing import Dict

from mlflow.entities import Run
from results.utils import Results


class NumberOfDatasets(Results):
    approach_name = "3 vs 5 vs 8 datasets used. Ensemble "
    distribution_names = ("3 datasets", "5 datasets")
    results_root_path = os.path.dirname(__file__).replace("\\", "/")
    first_result_key_name_loss = "3_ensemble_loss"
    first_result_key_name_val_loss = "3_ensemble_val_loss"
    first_result_key_name_acc = "3_ensemble_accuracy"
    first_result_key_name_val_acc = "3_ensemble_val_accuracy"
    second_result_key_name_loss = "5_ensemble_loss"
    second_result_key_name_val_loss = "5_ensemble_val_loss"
    second_result_key_name_acc = "5_ensemble_accuracy"
    second_result_key_name_val_acc = "5_ensemble_val_accuracy"
    third_result_key_name_loss = "8_ensemble_loss"
    third_result_key_name_val_loss = "8_ensemble_val_loss"
    third_result_key_name_acc = "8_ensemble_accuracy"
    third_result_key_name_val_acc = "8_ensemble_val_accuracy"
    x_label_win_tie_loss = "using 5 source datasets"
    y_label_win_tie_loss = "using 3 source datasets"

    def __init__(
        self,
        first_experiment_id: str,
        second_experiment_id: str,
        third_experiment_id: str,
        assert_: bool = False,
    ):
        super().__init__(
            first_experiment_id=first_experiment_id,
            second_experiment_id=second_experiment_id,
            assert_=assert_,
        )
        self.third_experiment_id = third_experiment_id
        self.third_experiment_runs: Dict[str, Run] = self._get_third_experiment_runs()

    def _get_first_experiment_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(
            self.first_experiment_id, add_prefix="3_"
        )
        return hist

    def _get_second_experiment_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(
            self.second_experiment_id, add_prefix="5_"
        )
        return hist

    def _get_third_experiment_runs(self):
        hist = self._get_history_per_experiment(
            self.third_experiment_id, add_prefix="8_"
        )
        return hist

    def _get_history_summarized_per_epoch_comparison(self):
        history_summarized_1 = self._get_history_summarized_per_epoch(
            experiment_run=self.first_experiment_runs,
            metrics_names=[
                self.first_result_key_name_acc,
                self.first_result_key_name_val_acc,
                self.first_result_key_name_loss,
                self.first_result_key_name_val_loss,
            ],
        )

        history_summarized_2 = self._get_history_summarized_per_epoch(
            experiment_run=self.second_experiment_runs,
            metrics_names=[
                self.second_result_key_name_acc,
                self.second_result_key_name_val_acc,
                self.second_result_key_name_loss,
                self.second_result_key_name_val_loss,
            ],
        )
        history_summarized_3 = self._get_history_summarized_per_epoch(
            experiment_run=self.third_experiment_runs,
            metrics_names=[
                self.third_result_key_name_acc,
                self.third_result_key_name_val_acc,
                self.third_result_key_name_loss,
                self.third_result_key_name_val_loss,
            ],
        )
        return {**history_summarized_1, **history_summarized_2, **history_summarized_3}

    def _prepare_legend(self, text: str):
        mapping = {
            self.first_result_key_name_loss: "Ensemble with 3 datasets \n- loss - train split",
            self.first_result_key_name_val_loss: "Ensemble with 3 datasets \n- loss - validation split",
            self.first_result_key_name_acc: "Ensemble with 3 datasets \n- accuracy - train split",
            self.first_result_key_name_val_acc: "Ensemble with 3 datasets \n- accuracy - validation split",
            self.second_result_key_name_loss: "Ensemble with 5 datasets \n- loss - train split",
            self.second_result_key_name_val_loss: "Ensemble with 5 datasets \n- loss - validation split",
            self.second_result_key_name_acc: "Ensemble with 5 datasets \n- accuracy - train split",
            self.second_result_key_name_val_acc: "Ensemble with 5 datasets \n- accuracy - validation split",
            self.third_result_key_name_loss: "Ensemble with 8 datasets \n- loss - train split",
            self.third_result_key_name_val_loss: "Ensemble with 8 datasets \n- loss - validation split",
            self.third_result_key_name_acc: "Ensemble with 8 datasets \n- accuracy - train split",
            self.third_result_key_name_val_acc: "Ensemble with 8 datasets \n- accuracy - validation split",
        }
        return mapping[text]

    @staticmethod
    def _get_plot_kwargs(metric_name) -> dict:
        return {
            "color": "pink"
            if "3" in metric_name.lower()
            else "blue"
            if "5" in metric_name.lower()
            else "yellow",
            "linestyle": "--" if "train" in metric_name.lower() else "-",
        }


results = NumberOfDatasets(
    first_experiment_id="879346711569312185",
    second_experiment_id="554900821027531839",
    third_experiment_id="528548208530565493",
    assert_=False,
)
results.save_wilcoxon_test()
results.get_mean_loss_acc_per_epoch()
results.win_tie_loss_diagram()
