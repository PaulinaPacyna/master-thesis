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
    second_result_key_nameloss = "baseline_no_transfer_learning_base_loss"
    second_result_key_nameval_loss = "baseline_no_transfer_learning_base_val_loss"
    second_result_key_nameacc = "baseline_no_transfer_learning_base_accuracy"
    second_result_key_nameval_acc = "baseline_no_transfer_learning_base_val_accuracy"

    def _get_transfer_learning_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(self.transfer_learning_experiment_id)
        return {
            key: run
            for key, run in hist.items()
            if run.info.run_name.startswith("Destination")
        }

    def _prepare_legend(self, text: str):
        mapping = {
            self.first_result_key_name_loss: "Loss - train split",
            self.first_result_key_name_val_loss: "Loss - validation split",
            self.first_result_key_name_acc: "Accuracy - train split",
            self.first_result_key_name_val_acc: "Accuracy - validation split",
            self.second_result_key_nameloss: "No transfer learning - loss - train split",
            self.second_result_key_nameval_loss: "No transfer learning - loss - validation split",
            self.second_result_key_nameacc: "No transfer learning - accuracy - train split",
            self.second_result_key_nameval_acc: "No transfer learning - accuracy - validation split",
        }
        return mapping[text]


results = BaselineResults(
    transfer_learning_experiment_id="743133642334170939",
    no_transfer_learning_experiment_id="183382388301527558",
)
results.get_mean_loss_acc_per_epoch("loss")
results.get_mean_loss_acc_per_epoch("acc")
results.win_tie_loss_diagram(epoch=10)
results.win_tie_loss_diagram(epoch=5)
