

"""
To spikes:
Each EEG channel becomes one up and one down neuron (256 input neurons)
Go through each sample and calculate delta. If delta is less than threshold register a down spike. If delta is more than threshold register an up spike.

Creates:
All_spikes.pt:
[spikes([N_trials(0-5639)*N_samples(4.5*256Hz=0-1153)*num_neurons(0-255)],
session(0-2),
subject(0-9),
condition(0-2),
class(0-3)]
"""

import glob
import re
import sys

import mne
import numpy as np
import torch



# The "derivatives" folder. I keep it .gitignored. Ask Ludvig if you want the raw files.
DATASET_FOLDER = "../data/derivatives"

OUTPUT_FILE = "../data/All_spikes.pt"


THRESHOLD = 0.5



def load_session(fif_file):
    #The EEG signal. We keep only the 128 EEG channels.
    epochs = mne.read_epochs(fif_file, verbose="error")
    epochs.pick("eeg")
    eeg = epochs.get_data()               # shape: [trials, 128 channels, 1153 samples]

    #The labels. One row per trial.
    events_file = fif_file.replace("_eeg-epo.fif", "_events.dat")
    events = np.load(events_file, allow_pickle=True)
    labels = events[:, 1].astype(int)       # which word (0-3)
    conditions = events[:, 2].astype(int)   # which task (0-2)

    return eeg, labels, conditions



def delta_encode(trial):
    n_channels, n_samples = trial.shape

    up_spikes = np.zeros((n_samples, n_channels), dtype=bool)
    down_spikes = np.zeros((n_samples, n_channels), dtype=bool)

    for channel in range(n_channels):
        threshold = THRESHOLD * trial[channel].std()
        signal = trial[channel].tolist()   
        last_value = signal[0]

        for t in range(1, n_samples):
            change = signal[t] - last_value

            if change >= threshold:
                up_spikes[t, channel] = True
                last_value = signal[t]
            elif change <= -threshold:
                down_spikes[t, channel] = True
                last_value = signal[t]

    # Put UP neurons first (0-127), then DOWN neurons (128-255)
    return np.concatenate([up_spikes, down_spikes], axis=1)   # [1153, 256]



all_spikes, all_subjects, all_sessions, all_conditions, all_labels = [], [], [], [], []

fif_files = sorted(glob.glob(f"{DATASET_FOLDER}/**/*_eeg-epo.fif", recursive=True))
if not fif_files:
    sys.exit(f"No *_eeg-epo.fif files found in {DATASET_FOLDER}")

for fif_file in fif_files:
    # Read subject and session number from the file name, e.g. "sub-01_ses-02"
    subject_nr = int(re.search(r"sub-(\d+)", fif_file).group(1)) - 1   # make it 0-9
    session_nr = int(re.search(r"ses-(\d+)", fif_file).group(1)) - 1   # make it 0-2

    eeg, labels, conditions = load_session(fif_file)
    print(f"Subject {subject_nr}, session {session_nr}: {len(eeg)} trials")

    for i in range(len(eeg)):
        all_spikes.append(delta_encode(eeg[i]))
        all_subjects.append(subject_nr)
        all_sessions.append(session_nr)
        all_conditions.append(conditions[i])
        all_labels.append(labels[i])



data = {
    "spikes": torch.tensor(np.array(all_spikes)),   # [N, 1153, 256]
    "subject": torch.tensor(all_subjects),
    "session": torch.tensor(all_sessions),
    "condition": torch.tensor(all_conditions),
    "label": torch.tensor(all_labels),
}
torch.save(data, OUTPUT_FILE)

print(f"\nSaved {OUTPUT_FILE}")
print("spikes shape:", tuple(data["spikes"].shape))
print("label values:", data["label"].unique().tolist(), "(should be 0-3)")
print("condition values:", data["condition"].unique().tolist(), "(should be 0-2)")