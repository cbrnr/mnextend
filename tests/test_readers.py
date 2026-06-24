# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Tests for mnextend.io.readers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mnextend.io.readers import (
    _read,
    epochs_readers,
    raw_readers,
    read_epochs,
    read_raw,
)
from mnextend.io.utils import split_name_ext


@pytest.fixture()
def mock_readers():
    """Minimal readers dict for testing args/kwargs forwarding and return values.

    Real readers require actual files on disk. These mocks assert call behavior (which
    reader was called, with which arguments) and inspect return values without any I/O.
    """
    return {
        ".fif": MagicMock(return_value="raw_fif"),
        ".fif.gz": MagicMock(return_value="raw_fif_gz"),
    }


@pytest.mark.parametrize("ext", raw_readers.keys())
def test_split_name_ext_raw(ext):
    assert split_name_ext(f"test{ext}", raw_readers) == ("test", ext)


@pytest.mark.parametrize("ext", epochs_readers.keys())
def test_split_name_ext_epochs(ext):
    assert split_name_ext(f"test{ext}", epochs_readers) == ("test", ext)


def test_split_name_ext_unsupported():
    assert split_name_ext("test.xxx", raw_readers) == ("test.xxx", None)


def test_split_name_ext_no_extension():
    assert split_name_ext("recording", raw_readers) == ("recording", None)


def test_split_name_ext_hidden_file():
    """A hidden file (starting with a dot) must not be mistaken for a suffix."""
    assert split_name_ext(".fif", raw_readers) == (".fif", None)


def test_split_name_ext_case_insensitive():
    assert split_name_ext("recording.FIF", raw_readers) == ("recording", ".fif")


def test_split_name_ext_dot_in_stem():
    """A dot in the stem must not be mistaken for a suffix separator."""
    assert split_name_ext("sub-01.run-1.fif", raw_readers) == ("sub-01.run-1", ".fif")


def test_split_name_ext_compound_wins_over_simple():
    """Compound extension (.fif.gz) must take priority over trailing suffix (.gz)."""
    assert split_name_ext("recording.fif.gz", raw_readers) == ("recording", ".fif.gz")


def test_split_name_ext_path_object():
    assert split_name_ext(Path("data/sub-01/recording.fif"), raw_readers) == (
        "recording",
        ".fif",
    )


def test_read_passes_positional_args(mock_readers):
    _read("recording.fif", mock_readers, "arg1", "arg2")
    mock_readers[".fif"].assert_called_once_with("recording.fif", "arg1", "arg2")


def test_read_passes_kwargs(mock_readers):
    _read("recording.fif", mock_readers, preload=True)
    mock_readers[".fif"].assert_called_once_with("recording.fif", preload=True)


def test_read_returns_reader_result(mock_readers):
    assert _read("recording.fif.gz", mock_readers) == "raw_fif_gz"


def test_read_accepts_path_object(mock_readers):
    _read(Path("data/recording.fif"), mock_readers)
    mock_readers[".fif"].assert_called_once()


def test_read_unsupported_ext_raises():
    with pytest.raises(ValueError, match=r"\.unknown"):
        _read("recording.unknown", raw_readers)


def test_read_no_extension_raises():
    with pytest.raises(ValueError, match=r"^Unsupported file type\.$"):
        _read("recording", raw_readers)


def test_read_raw_unsupported_raises():
    with pytest.raises(ValueError, match=r"Unsupported file type"):
        read_raw("recording.unknown")


def test_read_epochs_unsupported_raises():
    with pytest.raises(ValueError, match=r"Unsupported file type"):
        read_epochs("epochs.edf")  # .edf is raw-only
