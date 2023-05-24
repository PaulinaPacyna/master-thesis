import os
from typing import Dict

from mlflow.entities import Run
from results.utils import Results


class BaselineResults(Results):
    approach_name = "Baseline"
    results_root_path = os.path.dirname(__file__)
    first_result_key_name_loss = "baseline_loss"
    first_result_key_name_val_loss = "baseline_val_loss"
    first_result_key_name_acc = "baseline_accuracy"
    first_result_key_name_val_acc = "baseline_val_accuracy"
    second_result_key_name_loss = "base_loss"
    second_result_key_name_val_loss = "base_val_loss"
    second_result_key_name_acc = "base_accuracy"
    second_result_key_name_val_acc = "base_val_accuracy"

    def _get_first_experiment_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(
            self.first_experiment_id, exclude_from_name="Source"
        )
        return hist

    def _prepare_legend(self, text: str):
        mapping = {
            self.first_result_key_name_loss: "Loss - train split",
            self.first_result_key_name_val_loss: "Loss - validation split",
            self.first_result_key_name_acc: "Accuracy - train split",
            self.first_result_key_name_val_acc: "Accuracy - validation split",
            self.second_result_key_name_loss: "No transfer learning \n- loss - train split",
            self.second_result_key_name_val_loss: "No transfer learning \n- loss - validation split",
            self.second_result_key_name_acc: "No transfer learning \n- accuracy - train split",
            self.second_result_key_name_val_acc: "No transfer learning \n- accuracy - validation split",
        }
        return mapping[text]


results = BaselineResults(
    first_experiment_id="743133642334170939",
    second_experiment_id="183382388301527558",
    assert_=False,
)
results.get_mean_loss_acc_per_epoch()
results.win_tie_loss_diagram()
