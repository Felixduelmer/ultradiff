"""Plot the target and fitted B-mode for the middle view of a saved sweep."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
args = parser.parse_args()
with np.load(args.run / "sweep.npz") as sweep:
    targets = sweep["target"]
    index = len(targets) // 2 - 1
    images = [targets[index].squeeze(), sweep["reconstruction"][index].squeeze()]
limit = max(images[0].max(), images[1].max())
fig, axes = plt.subplots(1, 2, constrained_layout=True)
for ax, image, title in zip(axes, images, ["Reference B-mode", "Optimized SDF"]):
    ax.imshow(image, cmap="gray", vmin=0, vmax=limit)
    ax.set_title(title)
    ax.axis("off")
fig.savefig(args.run / "comparison.png", dpi=180)
