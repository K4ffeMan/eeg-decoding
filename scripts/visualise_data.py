
#Figures are written to ../docs/figures by default.


import argparse
import pickle
from pathlib import Path

import matplotlib

import numpy as np

derivatives_filepath = "../data/derivatives"

CLASSES = {0: "Up (Arriba)", 1: "Down (Abajo)", 2: "Right (Derecha)", 3: "Left (Izquierda)"}
CONDITIONS = {0: "Pronounced", 1: "Inner", 2: "Visualized"}

ACTION_INTERVAL = (1.5, 3.5)



def session_prefix(root, sub, ses):
    return Path(root) / f"sub-{sub:02d}" / f"ses-{ses:02d}" / f"sub-{sub:02d}_ses-{ses:02d}_"


def load_events(root, sub, ses):
    return np.load(f"{session_prefix(root, sub, ses)}events.dat", allow_pickle=True)


def load_report(root, sub, ses):
    with open(f"{session_prefix(root, sub, ses)}report.pkl", "rb") as f:
        return pickle.load(f)


def load_epochs(root, sub, ses, kind="eeg"):
    import mne

    epochs = mne.read_epochs(f"{session_prefix(root, sub, ses)}{kind}-epo.fif", verbose="error")
    if kind in ("eeg", "baseline"):
        # Channel names A1..D32 match MNE's standard BioSemi 128 layout
        try:
            epochs.set_montage("biosemi128", on_missing="ignore", verbose="error")
        except ValueError:
            pass
    return epochs


def list_sessions(root):
    sessions = []
    for sub_dir in sorted(Path(root).glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            sub, ses = int(sub_dir.name[4:]), int(ses_dir.name[4:])
            if Path(f"{session_prefix(root, sub, ses)}events.dat").exists():
                sessions.append((sub, ses))
    return sessions


def plot_overview(root, out_dir):
    """Dataset-wide summary using only the small events/report files."""
    import matplotlib.pyplot as plt

    sessions = list_sessions(root)
    subjects = sorted({s for s, _ in sessions})
    n_ses = max(s for _, s in sessions)

    trials = np.zeros((len(subjects), n_ses))
    cond_counts = np.zeros((len(subjects), len(CONDITIONS)))
    class_counts = np.zeros((len(CONDITIONS), len(CLASSES)))
    accuracy = np.full((len(subjects), n_ses), np.nan)
    emg_trials = np.zeros((len(subjects), n_ses))
    ages, genders = {}, {}

    for sub, ses in sessions:
        i = subjects.index(sub)
        ev = load_events(root, sub, ses)
        trials[i, ses - 1] = len(ev)
        for c in CONDITIONS:
            cond_counts[i, c] += np.sum(ev[:, 2] == c)
            for k in CLASSES:
                class_counts[c, k] += np.sum((ev[:, 2] == c) & (ev[:, 1] == k))
        rep = load_report(root, sub, ses)
        total = rep.get("Ans_R", 0) + rep.get("Ans_W", 0)
        if total:
            accuracy[i, ses - 1] = 100 * rep["Ans_R"] / total
        emg_trials[i, ses - 1] = len(np.atleast_1d(rep.get("EMG_trials", [])))
        ages[sub], genders[sub] = rep.get("Age"), rep.get("Gender")

    labels = [f"sub-{s:02d}\n{ages[s]}{genders[s]}" for s in subjects]
    x = np.arange(len(subjects))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"Inner Speech dataset overview: {len(subjects)} subjects, {len(sessions)} sessions, "
                 f"{int(trials.sum())} trials")

    ax = axes[0, 0]
    bottom = np.zeros(len(subjects))
    for c, name in CONDITIONS.items():
        ax.bar(x, cond_counts[:, c], bottom=bottom, label=name)
        bottom += cond_counts[:, c]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Trials")
    ax.set_title("Trials per subject by condition")
    ax.legend()

    ax = axes[0, 1]
    w = 0.8 / len(CLASSES)
    for k, name in CLASSES.items():
        ax.bar(np.arange(len(CONDITIONS)) + (k - 1.5) * w, class_counts[:, k], w, label=name)
    ax.set_xticks(np.arange(len(CONDITIONS)), list(CONDITIONS.values()))
    ax.set_ylabel("Trials")
    ax.set_title("Class balance per condition (all subjects)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    w = 0.8 / n_ses
    for s in range(n_ses):
        ax.bar(x + (s - (n_ses - 1) / 2) * w, accuracy[:, s], w, label=f"ses-{s + 1:02d}")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% correct")
    ax.set_title("Attention-check accuracy (Ans_R / (Ans_R + Ans_W))")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    im = ax.imshow(emg_trials.T, cmap="Reds", aspect="auto")
    for i in range(len(subjects)):
        for s in range(n_ses):
            ax.text(i, s, int(emg_trials[i, s]), ha="center", va="center", fontsize=8)
    ax.set_xticks(x, [f"sub-{s:02d}" for s in subjects], rotation=45)
    ax.set_yticks(range(n_ses), [f"ses-{s + 1:02d}" for s in range(n_ses)])
    ax.set_title("Trials flagged for EMG contamination")
    fig.colorbar(im, ax=ax, label="Trials")

    _save(fig, out_dir, "overview.png")



def _shade_trial_phases(ax):
    ax.axvspan(*ACTION_INTERVAL, color="tab:green", alpha=0.08, lw=0)
    ax.axvline(0, color="k", lw=0.8, ls="--")


def plot_single_trial(epochs, events, tag, out_dir, trial=0, n_channels=16):
    """Stacked traces of one trial on evenly spread channels."""
    import matplotlib.pyplot as plt

    picks = np.linspace(0, len(epochs.ch_names) - 1, n_channels).astype(int)
    data = epochs.get_data(picks=picks)[trial] * 1e6  # V -> µV
    t = epochs.times
    spacing = 3 * np.median(np.std(data, axis=1))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    for j, ch in enumerate(picks):
        ax.plot(t, data[j] - j * spacing, lw=0.7, color="k")
    ax.set_yticks(-np.arange(len(picks)) * spacing, [epochs.ch_names[c] for c in picks])
    _shade_trial_phases(ax)
    cls, cond = CLASSES[events[trial, 1]], CONDITIONS[events[trial, 2]]
    ax.set_title(f"{tag} trial {trial}: {cond} / {cls}   (spacing {spacing:.0f} µV; "
                 f"dashed = cue onset, green = decoding window)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    _save(fig, out_dir, f"{tag}_single_trial.png")


def plot_erp_by_condition(epochs, events, tag, out_dir):

    import matplotlib.pyplot as plt

    data = epochs.get_data() * 1e6
    t = epochs.times
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for c, name in CONDITIONS.items():
        sel = events[:, 2] == c
        if not sel.any():
            continue
        erp = data[sel].mean(axis=1)  # average over channels -> (trials, times)
        m, sem = erp.mean(0), erp.std(0) / np.sqrt(sel.sum())
        axes[0].plot(t, m, label=f"{name} (n={sel.sum()})")
        axes[0].fill_between(t, m - sem, m + sem, alpha=0.25)
        gfp = data[sel].mean(axis=0).std(axis=0)  # std across channels of the evoked
        axes[1].plot(t, gfp, label=name)
    for ax in axes:
        _shade_trial_phases(ax)
        ax.legend()
    axes[0].set_ylabel("Mean amplitude (µV)")
    axes[0].set_title(f"{tag}: average over all channels, mean ± SEM over trials")
    axes[1].set_ylabel("GFP (µV)")
    axes[1].set_title("Global field power of the evoked response")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(t[0], t[-1])
    _save(fig, out_dir, f"{tag}_erp_by_condition.png")


def plot_erp_by_class(epochs, events, tag, out_dir):
    """Evoked response per class for each condition (channel average)."""
    import matplotlib.pyplot as plt

    data = epochs.get_data() * 1e6
    t = epochs.times
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(16, 4.5), sharey=True,
                             constrained_layout=True)
    for ax, (c, cname) in zip(axes, CONDITIONS.items()):
        for k, kname in CLASSES.items():
            sel = (events[:, 2] == c) & (events[:, 1] == k)
            if sel.any():
                ax.plot(t, data[sel].mean(axis=(0, 1)), label=f"{kname} (n={sel.sum()})", lw=1)
        _shade_trial_phases(ax)
        ax.set_title(cname)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Mean amplitude (µV)")
    fig.suptitle(f"{tag}: class-wise evoked response (channel average)")
    _save(fig, out_dir, f"{tag}_erp_by_class.png")


def plot_psd(epochs, baseline, events, tag, out_dir, fmax=60):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    action = epochs.copy().crop(*ACTION_INTERVAL)
    for c, name in CONDITIONS.items():
        sel = np.where(events[:, 2] == c)[0]
        if len(sel) == 0:
            continue
        spec = action[sel].compute_psd(fmin=1, fmax=fmax, verbose="error")
        psd, f = spec.get_data(return_freqs=True)
        ax.semilogy(f, psd.mean(axis=(0, 1)) * 1e12, label=f"{name} ({ACTION_INTERVAL[0]}-{ACTION_INTERVAL[1]} s)")
    if baseline is not None:
        spec = baseline.compute_psd(fmin=1, fmax=fmax, verbose="error")
        psd, f = spec.get_data(return_freqs=True)
        ax.semilogy(f, psd.mean(axis=(0, 1)) * 1e12, "k--", label="Resting baseline")
    for lo, hi, band in [(4, 8, "θ"), (8, 13, "α"), (13, 30, "β")]:
        ax.axvspan(lo, hi, color="grey", alpha=0.07 if band != "α" else 0.15, lw=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], band, ha="center", va="top")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz, channel average)")
    ax.set_title(f"{tag}: power spectrum")
    ax.legend()
    _save(fig, out_dir, f"{tag}_psd.png")


def plot_topomaps(epochs, events, tag, out_dir):

    import matplotlib.pyplot as plt
    import mne

    if epochs.get_montage() is None:
        print("  (no montage available, skipping topomaps)")
        return
    times = [0.2, 0.75, 1.5, 2.5, 3.5]
    fig, axes = plt.subplots(len(CONDITIONS), len(times) + 1, figsize=(15, 8),
                             constrained_layout=True)
    for row, (c, name) in enumerate(CONDITIONS.items()):
        sel = np.where(events[:, 2] == c)[0]
        if len(sel) == 0:
            continue
        evoked = epochs[sel].average()
        evoked.plot_topomap(times=times, axes=axes[row, :len(times)], show=False,
                            colorbar=False, vlim=(-3, 3), units="µV", time_format="%.2f s")
        spec = epochs[sel].copy().crop(*ACTION_INTERVAL).compute_psd(fmin=8, fmax=13,
                                                                     verbose="error")
        alpha = 10 * np.log10(spec.get_data().mean(axis=(0, 2)) * 1e12)  # dB re 1 µV²/Hz
        mne.viz.plot_topomap(alpha, epochs.info, axes=axes[row, -1], show=False, cmap="viridis",
                             vlim=(np.percentile(alpha, 5), np.percentile(alpha, 95)))
        axes[row, 0].set_ylabel(name, fontsize=12)
        axes[row, -1].set_title(f"α power {ACTION_INTERVAL[0]}-{ACTION_INTERVAL[1]} s\n"
                                f"{alpha.min():.0f}..{alpha.max():.0f} dB", fontsize=9)
    fig.suptitle(f"{tag}: evoked topographies (±3 µV) and alpha power")
    _save(fig, out_dir, f"{tag}_topomaps.png")


def plot_exg(exg, events, tag, out_dir, report=None):
    """External channels (eye / mouth EMG) averaged per condition."""
    import matplotlib.pyplot as plt

    data = exg.get_data() * 1e6
    t = exg.times
    n = len(exg.ch_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        for c, name in CONDITIONS.items():
            sel = events[:, 2] == c
            if sel.any():
                # mean rectified signal shows muscle activity better than a signed mean
                ax.plot(t, np.abs(data[sel, j]).mean(0), lw=0.8, label=name)
        _shade_trial_phases(ax)
        ax.set_ylabel(exg.ch_names[j], rotation=0, ha="right")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(t[0], t[-1])
    title = f"{tag}: rectified EXG activity per condition (µV)"
    if report is not None and len(np.atleast_1d(report.get("EMG_trials", []))):
        title += f"   EMG-flagged trials: {np.atleast_1d(report['EMG_trials']).astype(int).tolist()}"
    fig.suptitle(title)
    _save(fig, out_dir, f"{tag}_exg.png")


def plot_session(root, sub, ses, out_dir):
    tag = f"sub-{sub:02d}_ses-{ses:02d}"
    print(f"Loading {tag} ...")
    events = load_events(root, sub, ses)
    report = load_report(root, sub, ses)
    eeg = load_epochs(root, sub, ses, "eeg")
    exg = load_epochs(root, sub, ses, "exg")
    try:
        baseline = load_epochs(root, sub, ses, "baseline").pick(eeg.ch_names)
    except (FileNotFoundError, ValueError):
        baseline = None
    assert len(events) == len(eeg), "events.dat and epochs disagree on trial count"
    print(f"  {eeg}\n  report: {report}")

    plot_single_trial(eeg, events, tag, out_dir)
    plot_erp_by_condition(eeg, events, tag, out_dir)
    plot_erp_by_class(eeg, events, tag, out_dir)
    plot_psd(eeg, baseline, events, tag, out_dir)
    plot_topomaps(eeg, events, tag, out_dir)
    plot_exg(exg, events, tag, out_dir, report)


# --------------------------------------------------------------------------- main
def _save(fig, out_dir, name):
    import matplotlib.pyplot as plt

    path = Path(out_dir) / name
    fig.savefig(path, dpi=120)
    print(f"  saved {path}")
    if not SHOW:
        plt.close(fig)


SHOW = False


def main():
    global SHOW
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=derivatives_filepath)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--out", default="../docs/figures")
    parser.add_argument("--overview-only", action="store_true")
    parser.add_argument("--show", action="store_true", help="open figure windows")
    args = parser.parse_args()

    SHOW = args.show
    if not SHOW:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(args.out).mkdir(parents=True, exist_ok=True)
    plot_overview(args.root, args.out)
    if not args.overview_only:
        plot_session(args.root, args.subject, args.session, args.out)
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
