import mne
import numpy as np
from scipy.signal import butter, sosfilt

def get_total_number_trials(fif_fnames: list[str]) -> int:
    total_trials_nb = 0
    for fif_fname in fif_fnames:
        epochs = mne.read_epochs(
            fif_fname,
            preload=False,
            verbose="error",
        )
        total_trials_nb += len(epochs)

    return total_trials_nb
        

def load_raw_data(fif_fname):
    epochs = mne.read_epochs(fif_fname, verbose="error")
    epochs.pick("eeg")
    eeg = epochs.get_data()               # shape: [trials, 128 channels, 1153 samples]

    # The labels. One row per trial.
    events_file = fif_fname.replace("_eeg-epo.fif", "_events.dat")
    events = np.load(events_file, allow_pickle=True)
    labels = events[:, 1].astype(int)       # which word (0-3)
    conditions = events[:, 2].astype(int)   # which task (0-2)

    return eeg, labels, conditions

def causal_filtering(eeg: np.ndarray, representations: list[str], bands: list[str], target_sfreq: float = 256, filter_order: int = 2, lowpass_critical_freq: float = 5.0):

    bands_freq = {
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (12.0, 30.0),
        }

    filtered = []
    for band in bands:
        if band not in bands_freq:
            valid_bands = ", ".join(bands_freq)
            raise ValueError(
                f"Unknown band {band!r}. Expected one of: {valid_bands}."
            )
        
        sos_bp = butter(
            N=filter_order,
            Wn=bands_freq[band],
            btype="bandpass",
            fs=target_sfreq,
            output="sos",
        )

        bp_waveform = sosfilt(sos_bp, eeg, axis=-1) # axis = -1 because eeg shape is [trials, 128 channels, 1153 samples]

        if "bandpass" in representations:
            filtered.append(bp_waveform)

        if "envelope" in representations or "diff_enveloppe" in representations or "power" in representations:
            sos_lp = butter(
                N=filter_order,
                Wn=lowpass_critical_freq,
                btype="lowpass",
                fs=target_sfreq,
                output="sos"
            )

            if "envelope" in representations or "diff_enveloppe" in representations:
                env = sosfilt(sos_lp, np.abs(bp_waveform), axis=-1)
                if "envelope" in representations:
                    filtered.append(env)

                if "diff_enveloppe" in representations:
                    filtered.append(np.diff(env, axis=-1, prepend=env[..., :1]))
        
            if "power" in representations:
                filtered.append(sosfilt(sos_lp, bp_waveform**2, axis=-1))

        if "diff_bp" in representations:
            filtered.append(np.diff(bp_waveform, axis=-1, prepend=bp_waveform[..., :1]))

    return np.concatenate(filtered, axis=1)