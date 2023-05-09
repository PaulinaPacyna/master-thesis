import os

from results.utils import Results


class BaselineResults(Results):
    approach_name = "Baseline"
    results_root_path = os.path.dirname(__file__)
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

    def _get_transfer_learning_runs(self):
        hist = self._get_history_per_experiment(self.transfer_learning_experiment_id)
        return [run for run in hist if run.info.run_name.startswith("Destination")]

    def _prepare_legend(self, text: str):
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


results = BaselineResults(
    transfer_learning_experiment_id="743133642334170939",
    no_transfer_learning_experiment_id="183382388301527558",
)
results.get_mean_loss_acc_per_epoch("loss")
results.get_mean_loss_acc_per_epoch("acc")
results.win_tie_loss_diagram(epoch=10)
results.win_tie_loss_diagram(epoch=5)
