import os

from results.utils import Results


class EnsembleResults(Results):
    results_root_path = os.path.dirname(__file__)
    transfer_learning_key_name_loss = "ensemble_loss"
    transfer_learning_key_name_val_loss = "ensemble_val_loss"
    transfer_learning_key_name_acc = "ensemble_accuracy"
    transfer_learning_key_name_val_acc = "ensemble_val_accuracy"
    no_transfer_learning_key_name_loss = "ensemble_no_transfer_learning_ensemble_loss"
    no_transfer_learning_key_name_val_loss = (
        "ensemble_no_transfer_learning_ensemble_val_loss"
    )
    no_transfer_learning_key_name_acc = (
        "ensemble_no_transfer_learning_ensemble_accuracy"
    )
    no_transfer_learning_key_name_val_acc = (
        "ensemble_no_transfer_learning_ensemble_val_accuracy"
    )

    def _prepare_legend(self, text):
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


results = EnsembleResults(
    transfer_learning_experiment_id="554900821027531839",
    no_transfer_learning_experiment_id="541913567164685548",
)
results.get_mean_loss_acc_per_epoch("loss")
results.get_mean_loss_acc_per_epoch("acc")
results.win_tie_loss_diagram(epoch=10)
results.win_tie_loss_diagram(epoch=5)
