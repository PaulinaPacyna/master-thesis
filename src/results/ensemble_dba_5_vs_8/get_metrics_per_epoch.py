import os
from typing import Dict

from mlflow.entities import Run
from results.utils import Results


class BaselineVsEnsembleResults(Results):
    approach_name = "8 datasets vs 5 datasets used. Ensemble "
    results_root_path = os.path.dirname(__file__)
    first_result_key_name_loss = "8_ensemble_loss"
    first_result_key_name_val_loss = "8_ensemble_val_loss"
    first_result_key_name_acc = "8_ensemble_accuracy"
    first_result_key_name_val_acc = "8_ensemble_val_accuracy"
    second_result_key_name_loss = "5_ensemble_loss"
    second_result_key_name_val_loss = "5_ensemble_val_loss"
    second_result_key_name_acc = "5_ensemble_accuracy"
    second_result_key_name_val_acc = "5_ensemble_val_accuracy"

    def _get_first_experiment_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(
            self.first_experiment_id, add_prefix="8_"
        )
        return hist

    def _get_second_experiment_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(
            self.second_experiment_id, add_prefix="5_"
        )
        return hist

    def _prepare_legend(self, text: str):
        mapping = {
            self.first_result_key_name_loss: "Ensemble with 5 datasets - loss - train split",
            self.first_result_key_name_val_loss: "Ensemble with 5 datasets - loss - validation split",
            self.first_result_key_name_acc: "Ensemble with 5 datasets - accuracy - train split",
            self.first_result_key_name_val_acc: "Ensemble with 5 datasets - accuracy - validation split",
            self.second_result_key_name_loss: "Ensemble with 8 datasets - loss - train split",
            self.second_result_key_name_val_loss: "Ensemble with 8 datasets - loss - validation split",
            self.second_result_key_name_acc: "Ensemble with 8 datasets - accuracy - train split",
            self.second_result_key_name_val_acc: "Ensemble with 8 datasets - accuracy - validation split",
        }
        return mapping[text]

    @staticmethod
    def _get_plot_kwargs(metric_name) -> dict:
        return {
            "color": "pink" if "8" in metric_name.lower() else "blue",
            "linestyle": "--" if "train" in metric_name.lower() else "-",
        }


results = BaselineVsEnsembleResults(
    first_experiment_id="528548208530565493",
    second_experiment_id="554900821027531839",
    assert_=False,
)
results.get_mean_loss_acc_per_epoch("loss")
results.get_mean_loss_acc_per_epoch("acc")
results.win_tie_loss_diagram(epoch=10)
results.win_tie_loss_diagram(epoch=5)
