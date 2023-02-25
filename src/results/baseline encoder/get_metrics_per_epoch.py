import json
import matplotlib.pyplot as plt
import numpy as np
from mlflow import MlflowClient

client = MlflowClient()
runs = client.search_runs(["674599303712758429"])

history_detailed = []
for run in runs:
    if (
        run.info.status == "FINISHED"
        and run.info.run_name
        not in ("Destination", "Destination plain", "Source model")
        and run.info.lifecycle_stage == "active"
    ):
        history_entry = json.load(open(run.info.artifact_uri + "/history.json"))
        if min([len(x) for x in history_entry.values()]) == 10:
            print(run.info)
            history_detailed.append(history_entry)

history = dict()
for key in history_detailed[0]:
    history[key] = np.array([stats[key] for stats in history_detailed]).mean(0).tolist()

losses = {key for key in history if "loss" in key}
accuracies = {key for key in history if "acc" in key}


def prepare_legend(text: str):
    text = text.replace("dest_", "")
    text = text.replace("_", " ")
    text = text.replace("weights", "transfer learning -")
    if "val" in text:
        text = text.replace("val ", "")
        text += " - validation split"
    else:
        text += " - train split"
    text = text.capitalize()
    return text


figure, ax = plt.subplots(figsize=(7, 7))
for key in sorted(losses, key=prepare_legend):
    plt.plot(history[key], label=prepare_legend(key), axes=ax)
figure.suptitle("Model loss")
ax.set_ylabel("loss")
ax.set_xlabel("epoch")
ax.legend()
plt.ylim(bottom=0)
plt.savefig("results/baseline encoder/loss.png")
plt.close(figure)


figure, ax = plt.subplots(figsize=(7, 7))
for key in sorted(accuracies, key=prepare_legend):
    plt.plot(history[key], label=prepare_legend(key), axes=ax)
ax.set_ylabel("accuracy")
figure.suptitle("Model accuracy")
ax.set_xlabel("epoch")
ax.legend()
plt.savefig("results/baseline encoder/acc.png")
plt.close(figure)
