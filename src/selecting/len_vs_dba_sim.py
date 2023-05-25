import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import get_lengths
from reading import Reading
from results.utils import cm
from selecting.utils import DBASelector

matplotlib.rc("font", size=9)
plt.rcParams["figure.dpi"] = 400


reading = Reading()
X, y = reading.read_dataset()
selecting = DBASelector()
lengths = get_lengths(X)
len_sim_pairs = []

for dataset in reading.categories:
    mask = np.char.startswith(y.ravel(), dataset + "_")
    category = reading.categories[dataset]
    mean_length = lengths[mask].mean()
    similarities = selecting.similarity_matrices[category][dataset]
    similarities = similarities[similarities.index != dataset]
    mean_sim = similarities.mean()
    len_sim_pairs.append((mean_length, mean_sim))
figure, ax = plt.subplots(figsize=(14 * cm, 14 * cm))
ax.scatter(*list(zip(*len_sim_pairs)), s=8)
ax.set_xscale("log")
ax.set_yscale("log")
figure.savefig("../latex/2. thesis/imgs/len_vs_dba_sim.png", transparent=True)
plt.close(figure)
