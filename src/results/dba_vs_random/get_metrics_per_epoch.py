import os
from typing import Dict

from mlflow.entities import Run
from results.utils import Results


class DbaSimiliarityVsRandomResults(Results):
    approach_name = "selecting datasets with DBA similarity vs random"
    distribution_names = ("DBA similarity", "random")
    results_root_path = os.path.dirname(__file__).replace("\\", "/")
    first_result_key_name_loss = "ensemble_loss"
    first_result_key_name_val_loss = "ensemble_val_loss"
    first_result_key_name_acc = "ensemble_accuracy"
    first_result_key_name_val_acc = "ensemble_val_accuracy"
    second_result_key_name_loss = "random_ensemble_loss"
    second_result_key_name_val_loss = "random_ensemble_val_loss"
    second_result_key_name_acc = "random_ensemble_accuracy"
    second_result_key_name_val_acc = "random_ensemble_val_accuracy"
    x_label_win_tie_loss = "random approach"
    y_label_win_tie_loss = "DBA similarity approach"

    def _get_second_experiment_runs(self) -> Dict[str, Run]:
        hist = self._get_history_per_experiment(
            self.second_experiment_id, add_prefix="random_"
        )
        return hist

    def _prepare_legend(self, text: str):
        mapping = {
            self.first_result_key_name_loss: "DBA similarity approach \n- loss - train split",
            self.first_result_key_name_val_loss: "DBA similarity approach \n- loss - validation split",
            self.first_result_key_name_acc: "DBA similarity approach \n- accuracy - train split",
            self.first_result_key_name_val_acc: "DBA similarity approach \n- accuracy - validation split",
            self.second_result_key_name_loss: "Random approach \n- loss - train split",
            self.second_result_key_name_val_loss: "Random approach \n- loss - validation split",
            self.second_result_key_name_acc: "Random approach \n- accuracy - train split",
            self.second_result_key_name_val_acc: "Random approach \n- accuracy - validation split",
        }
        return mapping[text]

    @staticmethod
    def _get_plot_kwargs(metric_name) -> dict:
        return {
            "color": "pink" if "random" not in metric_name.lower() else "blue",
            "linestyle": "--" if "train" in metric_name.lower() else "-",
        }


results = DbaSimiliarityVsRandomResults(
    first_experiment_id="554900821027531839",
    second_experiment_id="303093782117013484",
    assert_=False,
)
results.save_wilcoxon_test()
results.get_mean_loss_acc_per_epoch()
results.win_tie_loss_diagram()
