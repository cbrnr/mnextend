# © MNEXTEND developers
#
# License: BSD (3-clause)

from functools import partial
from pathlib import Path

import mne
from pybvrf import read_raw_bvrf

from mnextend.io.mat import read_raw_mat
from mnextend.io.npy import read_raw_npy
from mnextend.io.utils import split_name_ext
from mnextend.io.xdf import read_raw_xdf


def _read_unsupported(fname, *, suggest=None, **kwargs):
    ext = "".join(Path(fname).suffixes)
    msg = f"Unsupported file type ({ext})."
    if suggest is not None:
        msg += f" Try reading a {suggest} file instead."
    raise ValueError(msg)


# known file formats for raw (continuous) data
raw_readers = {
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".gdf": mne.io.read_raw_gdf,
    ".vhdr": mne.io.read_raw_brainvision,
    ".fif": mne.io.read_raw_fif,
    ".set": mne.io.read_raw_eeglab,
    ".cnt": mne.io.read_raw_cnt,
    ".mff": mne.io.read_raw_egi,
    ".nxe": mne.io.read_raw_eximia,
    ".hdr": mne.io.read_raw_nirx,
    ".snirf": mne.io.read_raw_snirf,
    ".mat": read_raw_mat,
    ".npy": read_raw_npy,
    **dict.fromkeys([".fif.gz"], mne.io.read_raw_fif),
    **dict.fromkeys([".xdf", ".xdfz", ".xdf.gz"], read_raw_xdf),
    **dict.fromkeys([".bvrh", ".bvrd", ".bvrm", ".bvri"], read_raw_bvrf),
    **dict.fromkeys([".vmrk", ".eeg"], partial(_read_unsupported, suggest=".vhdr")),
}

# known file formats for epochs (segmented) data
epochs_readers = {
    ".fif": mne.read_epochs,
    ".fif.gz": mne.read_epochs,
}



def _read(fname, readers, *args, **kwargs):
    """Read file using appropriate reader based on file extension."""
    _, ext = split_name_ext(fname, readers)
    if ext is not None:
        return readers[ext](fname, *args, **kwargs)
    ext = "".join(Path(fname).suffixes).lower()
    raise ValueError(
        f"Unsupported file type ({ext})." if ext else "Unsupported file type."
    )


def read_raw(fname, *args, **kwargs):
    """Read raw (continuous) data file.

    Parameters
    ----------
    fname : str | Path
        File name to load.

    Returns
    -------
    raw : mne.io.Raw
        Raw object.

    Notes
    -----
    This function supports reading raw data from different file formats. It uses the
    `raw_readers` dict to dispatch the appropriate read function for a supported file
    type.
    """
    return _read(fname, raw_readers, *args, **kwargs)


def read_epochs(fname, *args, **kwargs):
    """Read epochs (segmented) data file.

    Parameters
    ----------
    fname : str | Path
        File name to load.

    Returns
    -------
    epochs : mne.Epochs
        Epochs object.

    Notes
    -----
    This function supports reading epochs data from different file formats. It uses the
    `epochs_readers` dict to dispatch the appropriate read function for a supported file
    type.
    """
    return _read(fname, epochs_readers, *args, **kwargs)
