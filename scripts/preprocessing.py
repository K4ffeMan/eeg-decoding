import glob
import argparse
import json
import re
import sys
import os

import mne
import numpy as np
import torch

from preprocessing_tools import load_raw_data, causal_filtering, get_total_number_trials

def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "--dataset-dir-path",
        type=str,
        help="Path to the dataset root directory.",
    )
    
    arg_parser.add_argument(
        "--output-file-name",
        type=str,
        help="Name of the ouput file without extension.",
    )

    arg_parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        help="list of the bands (theta, alpha, beta)."
    )

    arg_parser.add_argument(
        "--representations",
        type=str,
        nargs="+",
        help="list of causal filters (bandpass, envelope, diff_envelope, power, diff_bp)"
    )

    arg_parser.add_argument(
        "--target-sfreq",
        type=float,
        default=256,
        help="Target sampling frequency.",
    )

    arg_parser.add_argument(
        "--filter-order",
        type=int,
        default=2,
        help="Filter order.",
    )

    arg_parser.add_argument(
        "--lowpass-critical-freq",
        type=float,
        default=5.0,
        help="Low-pass critical frequency.",
    )

    return arg_parser.parse_args()


def launch_preprocessing(bands:list[str], representations:list[str], target_sfreq:float, filter_order:int, lowpass_critical_freq:float):

    all_subject_numbers, all_session_numbers, all_conditions, all_labels = [], [], [], []

    fif_files = sorted(glob.glob(f"{DATASET_FOLDER}/**/*_eeg-epo.fif", recursive=True))
    
    if not fif_files:
        sys.exit(f"No *_eeg-epo.fif files found in {DATASET_FOLDER}")

    print(f"Found {len(fif_files)} epochs files to process. Starting...")

    N_trials_tot = get_total_number_trials(fif_files)
    N_bands = len(bands)
    N_representations = len(representations)

    eeg_shape = load_raw_data(fif_files[0])[0].shape # [trials, 128 channels, 1153 samples]

    N_channels = eeg_shape[1] * N_bands * N_representations

    N_samples = eeg_shape[2]

    os.makedirs(os.path.dirname(OUTPUT_FILE + '.npy'), exist_ok=True)
    all_eeg = np.memmap(shape=(N_trials_tot, N_channels, N_samples), filename=OUTPUT_FILE + '.npy', mode='w+', dtype=np.float32,)

    trial_offset = 0

    for i, fif_file in enumerate(fif_files):
        print(f"[{i+1}/{len(fif_files)}] Processing: {fif_file}")
        
        eeg, labels, conditions = load_raw_data(fif_file)

        eeg = causal_filtering(eeg, representations, bands, target_sfreq, filter_order, lowpass_critical_freq)

        sub_id = int(re.search(r"sub-(\d+)", fif_file).group(1)) - 1
        subject_numbers = [sub_id] * len(eeg)

        ses_id = int(re.search(r"ses-(\d+)", fif_file).group(1)) - 1
        session_numbers = [ses_id] * len(eeg)

        n_trials = len(eeg)
        all_eeg[trial_offset:trial_offset + n_trials] = torch.as_tensor(eeg)
        trial_offset += n_trials
        all_subject_numbers.extend(subject_numbers)
        all_session_numbers.extend(session_numbers)
        all_conditions.extend(conditions)
        all_labels.extend(labels)

    print("Concatenating data and converting to tensors...")
    
    data = {
        "eeg": torch.from_numpy(all_eeg),  
        "subject": torch.tensor(all_subject_numbers),
        "session": torch.tensor(all_session_numbers),
        "condition": torch.tensor(all_conditions),
        "label": torch.tensor(all_labels),
    }
    torch.save(data, OUTPUT_FILE + '.pt')

    print(f"\nSaved {OUTPUT_FILE + 'pt'}")
    print("eeg shape:", tuple(data["eeg"].shape))
    print("label values:", data["label"].unique().tolist(), "(should be 0-3)")
    print("condition values:", data["condition"].unique().tolist(), "(should be 0-2)")

if __name__ == "__main__":

    args = parse_arguments()
    
    DATASET_FOLDER = args.dataset_dir_path
    OUTPUT_FILE = '../data/' + args.output_file_name

    parameters = {
        "dataset_dir_path": args.dataset_dir_path,
        "bands": args.bands,
        "representations": args.representations,
        "target_sfreq": args.target_sfreq,
        "filter_order": args.filter_order,
        "lowpass_critical_freq": args.lowpass_critical_freq,
    }

    print("launching preprocessing")
    
    launch_preprocessing(
        bands=args.bands,
        representations=args.representations,
        target_sfreq=args.target_sfreq,
        filter_order=args.filter_order,
        lowpass_critical_freq=args.lowpass_critical_freq,
    )

    print("preprocessing ended: saving parameters")

    parameters_file = OUTPUT_FILE + ".json"
    with open(parameters_file, "w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2)