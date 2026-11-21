import matplotlib.pyplot as plt
import torch

FILEPATH = "../data/All_spikes.pt"
OUTPUT_FILE = "../docs/figures/spikes.png"
TRIAL = 0

data = torch.load(FILEPATH)
spikes = data["spikes"]  # [trials, time, 256]
print({k: tuple(v.shape) for k, v in data.items()})

conditions = ["Pronounced", "Inner", "Visualized"]
classes = ["Up", "Down", "Right", "Left"]
time = torch.arange(spikes.shape[1]) / 256 - 0.5  # seconds, cue at 0


def mean_rate(x, dim):
    # mean in chunks of 200 trials, so we never convert the whole tensor to float at once
    total = sum(chunk.float().sum(dim) for chunk in x.split(200))
    count = x.numel() / total.numel()
    return total / count


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1. Raster plot of one trial
t, neuron = spikes[TRIAL].nonzero(as_tuple=True)
up = neuron < 128
axes[0, 0].scatter(time[t[up]], neuron[up], s=0.5, c="tab:red", label="up neurons")
axes[0, 0].scatter(time[t[~up]], neuron[~up], s=0.5, c="tab:blue", label="down neurons")
cond, label = data["condition"][TRIAL], data["label"][TRIAL]
axes[0, 0].set_title(f"Trial {TRIAL}: {conditions[cond]} / {classes[label]}")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Neuron")
axes[0, 0].legend(loc="upper right", markerscale=10)

# 2. Average spike rate over time for each condition
for c, name in enumerate(conditions):
    rate = mean_rate(spikes[data["condition"] == c], dim=(0, 2))
    axes[0, 1].plot(time, rate, label=name)
axes[0, 1].axvline(0, color="k", ls="--")
axes[0, 1].set_title("Average spike rate over time")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Spikes per neuron per sample")
axes[0, 1].legend()

# 3. Average spike rate of each neuron
rate_per_neuron = mean_rate(spikes, dim=(0, 1))
axes[1, 0].bar(range(256), rate_per_neuron)
axes[1, 0].set_title("Average spike rate per neuron (0-127 up, 128-255 down)")
axes[1, 0].set_xlabel("Neuron")
axes[1, 0].set_ylabel("Spikes per sample")

# 4. Total number of spikes per trial, per subject
spikes_per_trial = torch.cat([chunk.sum(dim=(1, 2)) for chunk in spikes.split(200)])
per_subject = [spikes_per_trial[data["subject"] == s] for s in range(10)]
axes[1, 1].boxplot(per_subject)
axes[1, 1].set_xticklabels([f"sub-{s + 1:02d}" for s in range(10)])
axes[1, 1].set_title("Spikes per trial")
axes[1, 1].set_ylabel("Number of spikes")

plt.tight_layout()
plt.savefig(OUTPUT_FILE)
print(f"Saved {OUTPUT_FILE}")
plt.show()
