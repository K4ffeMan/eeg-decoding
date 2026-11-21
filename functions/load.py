"""

Function for loading data.

Solution 1(inter-subject)(35-50% SOTA)
Session 1 and 2 = train
Session 3 = val and test

Solution 2 (leave-one-subject-out)(25% SOTA)
Subject 1-8 = train
Subject 9 = val
Subject 10 = test

start and end time is to be able to crop to just the signal when the subject is thinking/pronouncing or visualising the direction



Load data:
set = load.spikes(filepath, set_type, solution, condition,start_time, end_time)
ex.
train = load.spikes(“home/usr/all_spikes.pt”, train, 1,0)
val = load.spikes(“home/usr/all_spikes.pt”,val 1,0)
test = load.spikes(“home/usr/all_spikes.pt”, test, 1,0)


Or (I'll fix this later)

filepath = config.All_spikes_filepath
set_type = config.All_spikes_set_type
solution = config.All_spikes_set_type


"""

import torch
from torch.utils.data import TensorDataset

SAMPLE_RATE = 256   # Hz

_cache = {}


def spikes(filepath, set_type, solution, condition=1, start_time=None, end_time=None):
    """
    filepath   : path to All_spikes.pt
    set_type   : "train", "val" or "test"
    solution   : 1 (across sessions) or 2 (across subjects)
    condition  : 0 = pronounced, 1 = inner speech (default), 2 = visualized
    start_time,
    end_time   : optional, in seconds from the start of the trial, to keep only
                 part of each trial (e.g. the action interval). None = keep all.

    Returns a TensorDataset where each item is (spikes [time, 256], label).
    """


    if filepath not in _cache:
        _cache[filepath] = torch.load(filepath)
    data = _cache[filepath]

    subject = data["subject"]
    session = data["session"]


    if solution == 1:
        #session 3 is split in two halves: every other trial goes to val, the rest to test.
        trial_nr = torch.arange(len(session))
        if set_type == "train":
            chosen = (session == 0) | (session == 1)
        elif set_type == "val":
            chosen = (session == 2) & (trial_nr % 2 == 0)
        elif set_type == "test":
            chosen = (session == 2) & (trial_nr % 2 == 1)

    elif solution == 2:
        if set_type == "train":
            chosen = subject <= 7
        elif set_type == "val":
            chosen = subject == 8
        elif set_type == "test":
            chosen = subject == 9

    else:
        raise ValueError("solution must be 1 or 2")

    # Only keep the condition we want (inner speech by default)
    chosen = chosen & (data["condition"] == condition)

    x = data["spikes"][chosen]
    y = data["label"][chosen]


    if start_time is not None or end_time is not None:
        first = 0 if start_time is None else int(start_time * SAMPLE_RATE)
        last = x.shape[1] if end_time is None else int(end_time * SAMPLE_RATE)
        x = x[:, first:last, :]

    print(f"{set_type}: {len(y)} trials, spikes shape {tuple(x.shape)}")

    return TensorDataset(x.float(), y)