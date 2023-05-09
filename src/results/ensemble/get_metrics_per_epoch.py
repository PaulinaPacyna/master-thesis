import os
from typing import Dict

from mlflow.entities import Run
from results.utils import Results


class EnsembleResults(Results):
    approach_name = "Ensemble"
    results_root_path = os.path.dirname(__file__)
    first_result_key_name_loss = "ensemble_loss"
    first_result_key_name_val_loss = "ensemble_val_loss"
    first_result_key_name_acc = "ensemble_accuracy"
    first_result_key_name_val_acc = "ensemble_val_accuracy"
    second_result_key_name_loss = "no_transfer_learning_ensemble_loss"
    second_result_key_name_val_loss = "no_transfer_learning_ensemble_val_loss"
    second_result_key_name_acc = "no_transfer_learning_ensemble_accuracy"
    second_result_key_name_val_acc = "no_transfer_learning_ensemble_val_accuracy"

    def _get_second_experiment_runs(self) -> Dict[str, Run]:
        return self._get_history_per_experiment(
            self.second_experiment_id, add_prefix="no_transfer_learning_"
        )

    def _prepare_legend(self, text):
        mapping = {
            self.first_result_key_name_loss: "Loss - train split",
            self.first_result_key_name_val_loss: "Loss - validation split",
            self.first_result_key_name_acc: "Accuracy - train split",
            self.first_result_key_name_val_acc: "Accuracy - validation split",
            self.second_result_key_name_loss: "No transfer learning - loss - train split",
            self.second_result_key_name_val_loss: "No transfer learning - loss - validation split",
            self.second_result_key_name_acc: "No transfer learning - accuracy - train split",
            self.second_result_key_name_val_acc: "No transfer learning - accuracy - validation split",
        }
        return mapping[text]


results = EnsembleResults(
    first_experiment_id="554900821027531839",
    second_experiment_id="541913567164685548",
    assert_=False,
)
results.get_mean_loss_acc_per_epoch("loss")
results.get_mean_loss_acc_per_epoch("acc")
results.win_tie_loss_diagram(epoch=10)
results.win_tie_loss_diagram(epoch=5)
