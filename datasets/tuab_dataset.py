import os
from copy import deepcopy

import numpy as np
import pyedflib
import torch
from torch.utils.data import Dataset


class TuabDataset(Dataset):
    """Dataset wrapper for TUAB EDF files."""

    def __init__(self, data_list: list, channel_labels=None, transform=None):
        self.data_list = data_list
        self.channel_labels = channel_labels
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = deepcopy(self.data_list[idx])
        channel_indices = None
        present_labels = None
        if self.channel_labels is not None:
            signal_headers = pyedflib.highlevel.read_edf_header(sample["edf_path"])["SignalHeaders"]
            label_to_index = {header["label"]: i for i, header in enumerate(signal_headers)}
            present_labels = [label for label in self.channel_labels if label in label_to_index]
            channel_indices = [label_to_index[label] for label in present_labels]

        signal, _, _ = pyedflib.highlevel.read_edf(sample["edf_path"], ch_nrs=channel_indices)
        signal = np.asarray(signal)

        if self.channel_labels is not None and present_labels is not None and len(present_labels) != len(self.channel_labels):
            fixed_signal = np.zeros((len(self.channel_labels), signal.shape[-1]), dtype=signal.dtype)
            label_to_row = {label: i for i, label in enumerate(self.channel_labels)}
            for row_idx, label in enumerate(present_labels):
                fixed_signal[label_to_row[label]] = signal[row_idx]
            signal = fixed_signal

        sample["signal"] = signal

        if self.transform:
            sample = self.transform(sample)

        return sample


def normalize_tuab_root(dataset_path: str) -> str:
    """Resolve either `/path/to/tuh` or `/path/to/tuh/tuab` to the TUAB root."""
    if os.path.isdir(os.path.join(dataset_path, "edf")):
        return dataset_path

    candidate = os.path.join(dataset_path, "tuab")
    if os.path.isdir(os.path.join(candidate, "edf")):
        return candidate

    raise FileNotFoundError(
        f"Could not find TUAB EDF directory under '{dataset_path}'. "
        "Expected either '<dataset_path>/edf' or '<dataset_path>/tuab/edf'."
    )


def resolve_tuab_channel_metadata(edf_path: str):
    """Select channels with the dominant sampling frequency."""
    signal_headers = pyedflib.highlevel.read_edf_header(edf_path)["SignalHeaders"]
    sample_rates = [header.get("sample_frequency", header.get("sample_rate")) for header in signal_headers]
    dominant_rate = max(sample_rates)
    channel_indices = [i for i, rate in enumerate(sample_rates) if rate == dominant_rate]
    channel_labels = [signal_headers[i]["label"] for i in channel_indices]
    return channel_indices, channel_labels
