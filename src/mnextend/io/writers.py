# © MNEXTEND developers
#
# License: BSD (3-clause)

from pathlib import Path

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
                "setname": fname,
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

epochs_writers = {
    ".fif": [write_fif, "Elekta Neuromag"],
    ".fif.gz": [write_fif, "Elekta Neuromag"],
}


def _write(fname, data, writer_dict):
    """Write data using appropriate writer based on file extension."""
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
