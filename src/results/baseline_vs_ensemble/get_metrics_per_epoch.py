import os
from typing import Dict

from mlflow.entities import Run
from results.utils import Results


class BaselineVsEnsembleResults(Results):
    approach_name = "ensemble approach vs baseline"
    results_root_path = os.path.dirname(__file__)
    first_result_key_name_loss = "ensemble_loss"
    first_result_key_name_val_loss = "ensemble_val_loss"
    first_result_key_name_acc = "ensemble_accuracy"
    first_result_key_name_val_acc = "ensemble_val_accuracy"
    second_result_key_name_loss = "baseline_loss"
    second_result_key_name_val_loss = "baseline_val_loss"
    second_result_key_name_acc = "baseline_accuracy"
    second_result_key_name_val_acc = "baseline_val_accuracy"

    def _get_second_experiment_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(
            self.second_experiment_id, exclude_from_name="Source"
        )
        return hist

    def _prepare_legend(self, text: str):
        mapping = {
            self.first_result_key_name_loss: "Ensemble - loss - train split",
            self.first_result_key_name_val_loss: "Ensemble - loss - validation split",
            self.first_result_key_name_acc: "Ensemble - accuracy - train split",
            self.first_result_key_name_val_acc: "Ensemble - accuracy - validation split",
            self.second_result_key_name_loss: "Baseline - loss - train split",
            self.second_result_key_name_val_loss: "Baseline - loss - validation split",
            self.second_result_key_name_acc: "Baseline - accuracy - train split",
            self.second_result_key_name_val_acc: "Baseline - accuracy - validation split",
        }
        return mapping[text]

    @staticmethod
    def _get_plot_kwargs(metric_name) -> dict:
        return {
            "color": "pink" if "ensemble" in metric_name.lower() else "blue",
            "linestyle": "--" if "train" in metric_name.lower() else "-",
        }


results = BaselineVsEnsembleResults(
    first_experiment_id="554900821027531839",
    second_experiment_id="743133642334170939",
    assert_=False,
)
results.get_mean_loss_acc_per_epoch()
results.win_tie_loss_diagram(epoch=10)
results.win_tie_loss_diagram(epoch=5)
