# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Tests for mnextend.io.writers."""

from pathlib import Path
from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from mnextend.io.readers import read_epochs, read_raw
from mnextend.io.writers import _write, write_epochs, write_raw


@pytest.fixture()
def mock_writers():
    """Minimal writers dict for testing dispatch without real I/O."""
    return {
        ".fif": [MagicMock(), "Elekta Neuromag"],
        ".fif.gz": [MagicMock(), "Elekta Neuromag"],
    }


@pytest.fixture()
def raw():
    """Small toy Raw object (4 ch, 100 samples at 256 Hz)."""
    data = np.random.default_rng(0).standard_normal((4, 100))
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=256, ch_types="eeg")
    return mne.io.RawArray(data, info)


@pytest.fixture()
def epochs():
    """Small toy Epochs object (5 epochs, 4 ch, 50 samples at 256 Hz)."""
    data = np.random.default_rng(0).standard_normal((5, 4, 50))
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=256, ch_types="eeg")
    events = np.column_stack(
        [np.arange(5) * 100, np.zeros(5, dtype=int), np.ones(5, dtype=int)]
    )
    return mne.EpochsArray(data, info, events=events, event_id={"stim": 1}, tmin=-0.1)


def test_write_dispatches_to_correct_writer(mock_writers):
    data = MagicMock()
    _write("recording.fif", data, mock_writers)
    mock_writers[".fif"][0].assert_called_once_with(Path("recording.fif"), data)


def test_write_dispatches_compound_ext(mock_writers):
    data = MagicMock()
    _write("recording.fif.gz", data, mock_writers)
    mock_writers[".fif.gz"][0].assert_called_once_with(Path("recording.fif.gz"), data)


def test_write_raw_unsupported_raises():
    with pytest.raises(ValueError, match=r"Unsupported file type"):
        write_raw("recording.unknown", MagicMock())


def test_write_epochs_unsupported_raises():
    with pytest.raises(ValueError, match=r"Unsupported file type"):
        write_epochs("epochs.edf", MagicMock())  # .edf is raw-only


def test_write_raw_fif(tmp_path, raw):
    fname = tmp_path / "recording-raw.fif"
    write_raw(fname, raw)
    raw2 = read_raw(fname, preload=True)
    # .fif stores EEG data as float32 internally, so exact equality is not expected
    np.testing.assert_allclose(raw.get_data(), raw2.get_data(), rtol=1e-6)
    assert raw.ch_names == raw2.ch_names
    assert raw.info["sfreq"] == raw2.info["sfreq"]


def test_write_epochs_fif(tmp_path, epochs):
    fname = tmp_path / "recording-epo.fif"
    write_epochs(fname, epochs)
    epochs2 = read_epochs(fname)
    # .fif stores EEG data as float32 internally, so exact equality is not expected
    np.testing.assert_allclose(epochs.get_data(), epochs2.get_data(), rtol=1e-6)
    assert epochs.ch_names == epochs2.ch_names
    assert epochs.info["sfreq"] == epochs2.info["sfreq"]
    assert epochs.event_id == epochs2.event_id


def test_write_raw_set(tmp_path, raw):
    fname = tmp_path / "recording.set"
    write_raw(fname, raw)
    raw2 = read_raw(fname, preload=True)
    np.testing.assert_allclose(raw.get_data(), raw2.get_data())
    assert raw.ch_names == raw2.ch_names
    assert raw.info["sfreq"] == raw2.info["sfreq"]


def test_write_epochs_set(tmp_path, epochs):
    fname = tmp_path / "epochs.set"
    write_epochs(fname, epochs)
    epochs2 = read_epochs(fname)
    np.testing.assert_allclose(epochs.get_data(), epochs2.get_data())
    assert epochs.ch_names == epochs2.ch_names
    assert epochs.info["sfreq"] == epochs2.info["sfreq"]
    assert epochs.event_id == epochs2.event_id


def test_write_raw_bdf(tmp_path, raw):
    fname = tmp_path / "recording.bdf"
    with pytest.warns(RuntimeWarning, match="equal-length data blocks"):
        write_raw(fname, raw)
    raw2 = read_raw(fname, preload=True)
    # BDF pads to full data blocks, so compare only the original samples
    np.testing.assert_allclose(
        raw.get_data(), raw2.get_data()[:, : raw.n_times], rtol=1e-4
    )
    assert raw.ch_names == raw2.ch_names
    assert raw.info["sfreq"] == raw2.info["sfreq"]


def test_write_raw_edf(tmp_path, raw):
    fname = tmp_path / "recording.edf"
    with pytest.warns(RuntimeWarning, match="equal-length data blocks"):
        write_raw(fname, raw)
    raw2 = read_raw(fname, preload=True)
    # EDF pads to full data blocks, so compare only the original samples
    np.testing.assert_allclose(
        raw.get_data(), raw2.get_data()[:, : raw.n_times], rtol=1e-2
    )
    assert raw.ch_names == raw2.ch_names
    assert raw.info["sfreq"] == raw2.info["sfreq"]


def test_write_raw_eeg(tmp_path, raw):
    fname = tmp_path / "recording.eeg"
    with pytest.warns(RuntimeWarning, match="Converting to float32"):
        write_raw(fname, raw)
    # write_bv exports to .vhdr (+ .eeg + .vmrk); read back via the header file
    raw2 = read_raw(fname.with_suffix(".vhdr"), preload=True)
    np.testing.assert_allclose(raw.get_data(), raw2.get_data(), rtol=1e-6)
    assert raw.ch_names == raw2.ch_names
    assert raw.info["sfreq"] == raw2.info["sfreq"]
