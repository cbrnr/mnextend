# © MNEXTEND developers
#
# License: BSD (3-clause)

from pathlib import Path

import numpy as np
from numpy.rec import fromarrays
from scipy.io import savemat

from mnextend.io.utils import split_name_ext


def write_fif(fname, raw):
    raw.save(fname, overwrite=True)


def write_set(fname, raw):
    """Export raw to EEGLAB .set file."""
    data = raw.get_data() * 1e6  # convert to microvolts
    fs = raw.info["sfreq"]
    times = raw.times
    ch_names = raw.info["ch_names"]
    chanlocs = fromarrays([ch_names], names=["labels"])
    events = fromarrays(
        [
            raw.annotations.description,
            raw.annotations.onset * fs + 1,
            raw.annotations.duration * fs,
        ],
        names=["type", "latency", "duration"],
    )
    savemat(
        fname,
        {
            "EEG": {
                "data": data,
                "setname": str(fname),
                "nbchan": data.shape[0],
                "pnts": data.shape[1],
                "trials": 1,
                "srate": fs,
                "xmin": times[0],
                "xmax": times[-1],
                "chanlocs": chanlocs,
                "event": events,
                "icawinv": [],
                "icasphere": [],
                "icaweights": [],
            }
        },
        appendmat=False,
    )


def write_bdf_edf(fname, raw):
    """Export raw to EDF file."""
    raw.export(fname, overwrite=True)


def write_bv(fname, raw):
    """Export data to BrainVision EEG/VHDR/VMRK file (requires pybv)."""
    raw.export(fname=Path(fname).with_suffix(".vhdr"), overwrite=True)


# These dicts contain each supported file extension as a key; the corresponding value is
# a list with two elements: (1) the writer function and (2) the full file format name.
raw_writers = {
    ".bdf": [write_bdf_edf, "Biosemi Data Format"],
    ".edf": [write_bdf_edf, "European Data Format"],
    ".eeg": [write_bv, "BrainVision"],
    ".fif": [write_fif, "Elekta Neuromag"],
    ".fif.gz": [write_fif, "Elekta Neuromag"],
    ".set": [write_set, "EEGLAB"],
}


def write_epochs_set(fname, epochs):
    """Export epochs to EEGLAB .set file."""
    data = epochs.get_data() * 1e6  # (n_epochs, n_channels, n_times), convert to µV
    data = data.transpose(1, 2, 0)  # EEGLAB expects (n_channels, n_times, n_epochs)

    n_epochs = len(epochs)
    n_times = len(epochs.times)
    fs = epochs.info["sfreq"]

    chanlocs = fromarrays([epochs.ch_names], names=["labels"])

    id_to_name = {v: k for k, v in epochs.event_id.items()}
    event_types = np.array(
        [id_to_name.get(eid, str(eid)) for eid in epochs.events[:, 2]]
    )
    # latency in samples (1-based) within the concatenated epoch data
    offset = round(abs(epochs.tmin) * fs)
    latencies = (np.arange(n_epochs) * n_times + offset + 1).astype(float)
    epoch_indices = np.arange(1, n_epochs + 1, dtype=float)

    events = fromarrays(
        [event_types, latencies, np.zeros(n_epochs), epoch_indices],
        names=["type", "latency", "duration", "epoch"],
    )
    # per-epoch struct: eventlatency in ms (0 = time-locking event)
    epoch_struct = fromarrays(
        [epoch_indices, event_types, np.zeros(n_epochs), np.zeros(n_epochs)],
        names=["event", "eventtype", "eventlatency", "eventduration"],
    )

    savemat(
        fname,
        {
            "EEG": {
                "data": data,
                "setname": str(fname),
                "nbchan": data.shape[0],
                "pnts": n_times,
                "trials": n_epochs,
                "srate": fs,
                "xmin": epochs.tmin,
                "xmax": epochs.tmax,
                "chanlocs": chanlocs,
                "event": events,
                "epoch": epoch_struct,
                "icawinv": [],
                "icasphere": [],
                "icaweights": [],
            }
        },
        appendmat=False,
    )


epochs_writers = {
    ".fif": [write_fif, "Elekta Neuromag"],
    ".fif.gz": [write_fif, "Elekta Neuromag"],
    ".set": [write_epochs_set, "EEGLAB"],
}


def _write(fname, data, writer_dict):
    """Write data using appropriate writer based on file extension."""
    fname = Path(fname).expanduser()
    _, ext = split_name_ext(fname, writer_dict)
    if ext is not None:
        return writer_dict[ext][0](fname, data)
    ext = "".join(Path(fname).suffixes).lower()
    raise ValueError(
        f"Unsupported file type ({ext})." if ext else "Unsupported file type."
    )


def write_raw(fname, raw):
    return _write(fname, raw, raw_writers)


def write_epochs(fname, epochs):
    return _write(fname, epochs, epochs_writers)
