# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Tests for adaptive line-noise removal."""

import mne
import numpy as np
import pytest

from mnextend import remove_line_noise


def _raw(data, sfreq, ch_types="eeg"):
    """Create a preloaded RawArray for line-noise tests."""
    if isinstance(ch_types, str):
        ch_types = [ch_types] * data.shape[0]
    info = mne.create_info(data.shape[0], sfreq, ch_types=ch_types)
    return mne.io.RawArray(data.copy(), info, verbose=False)


def _sinusoid_amplitude(data, sfreq, frequency):
    """Return the least-squares amplitude at one known frequency for each channel."""
    times = np.arange(data.shape[-1]) / sfreq
    basis = np.column_stack(
        (np.sin(2 * np.pi * frequency * times), np.cos(2 * np.pi * frequency * times))
    )
    coefficients, *_ = np.linalg.lstsq(basis, data.T, rcond=None)
    return np.linalg.norm(coefficients, axis=0)


@pytest.fixture
def synthetic_data():
    """Return smooth signal with stationary 50 Hz line noise and its harmonic."""
    sfreq = 500.0
    times = np.arange(int(12 * sfreq)) / sfreq
    clean = np.vstack(
        [
            0.8 * np.sin(2 * np.pi * 10 * times + phase)
            + 0.2 * np.sin(2 * np.pi * 23 * times - phase)
            for phase in (0.0, 0.4, 1.2)
        ]
    )
    line = np.vstack(
        [
            (1 + 0.2 * channel) * np.sin(2 * np.pi * 50 * times + 0.3 * channel)
            + 0.7 * np.cos(2 * np.pi * 100 * times + 0.2 * channel)
            for channel in range(clean.shape[0])
        ]
    )
    return clean + line, clean, sfreq


def test_harmonics_can_be_selected(synthetic_data):
    data, _, sfreq = synthetic_data
    without_harmonics = _raw(data, sfreq)
    with_harmonics = _raw(data, sfreq)

    remove_line_noise(
        without_harmonics,
        50,
        include_harmonics=False,
        window_length=2,
    )
    remove_line_noise(
        with_harmonics,
        50,
        include_harmonics=True,
        window_length=2,
    )

    clean_without = without_harmonics.get_data()
    clean_with = with_harmonics.get_data()
    assert np.max(_sinusoid_amplitude(clean_without, sfreq, 50)) < 1e-10
    assert np.max(_sinusoid_amplitude(clean_with, sfreq, 100)) < 1e-10
    assert np.min(_sinusoid_amplitude(clean_without, sfreq, 100)) > 0.65


@pytest.mark.parametrize(
    ("line_freq", "kwargs", "exception", "match"),
    [
        (0, {}, ValueError, "line_freq"),
        (250, {}, ValueError, "Nyquist"),
        ([], {}, ValueError, "at least one"),
        (50, {"window_length": 0}, ValueError, "window_length"),
        (50, {"overlap": 1}, ValueError, "overlap"),
        (50, {"include_harmonics": "yes"}, TypeError, "boolean"),
        (
            [50, 60],
            {"window_length": 0.003},
            ValueError,
            "independently estimable",
        ),
    ],
)
def test_invalid_inputs(line_freq, kwargs, exception, match):
    raw = _raw(np.ones((1, 10)), 500)
    with pytest.raises(exception, match=match):
        remove_line_noise(raw, line_freq, **kwargs)


def test_requires_preloaded_raw():
    raw = _raw(np.ones((1, 10)), 500)
    raw.preload = False

    with pytest.raises(RuntimeError, match="requires raw data to be loaded"):
        remove_line_noise(raw, 50)


def test_requires_raw_instance():
    with pytest.raises(TypeError, match="BaseRaw"):
        remove_line_noise(np.ones((1, 10)), 50)


def test_modifies_raw_in_place_and_returns_it(synthetic_data):
    data, _, sfreq = synthetic_data
    raw = _raw(data, sfreq)
    before = raw.get_data()

    returned = remove_line_noise(raw, 50, window_length=2)

    assert returned is raw
    assert not np.array_equal(raw.get_data(), before)


def test_default_picks_preserve_stimulation_channel():
    sfreq = 500.0
    times = np.arange(int(2 * sfreq)) / sfreq
    eeg = np.sin(2 * np.pi * 50 * times)
    stim = np.zeros_like(eeg)
    stim[[100, 700]] = 7
    raw = _raw(np.vstack((eeg, stim)), sfreq, ch_types=["eeg", "stim"])

    remove_line_noise(raw, 50, window_length=2)

    np.testing.assert_allclose(raw.get_data(picks="eeg"), 0, atol=1e-10)
    np.testing.assert_array_equal(raw.get_data(picks="stim")[0], stim)


def test_line_frequency_is_reduced(synthetic_data):
    data, _, sfreq = synthetic_data
    raw = _raw(data, sfreq)

    remove_line_noise(raw, 50, window_length=2)

    before = _sinusoid_amplitude(data, sfreq, 50)
    after = _sinusoid_amplitude(raw.get_data(), sfreq, 50)
    assert np.all(after < before * 1e-8)


def test_unrelated_sinusoid_is_preserved(synthetic_data):
    data, expected_clean, sfreq = synthetic_data
    raw = _raw(data, sfreq)

    remove_line_noise(raw, 50, window_length=2)

    clean = raw.get_data()
    before = _sinusoid_amplitude(expected_clean, sfreq, 10)
    after = _sinusoid_amplitude(clean, sfreq, 10)
    np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(clean, expected_clean, atol=1e-10)


def test_nearby_sinusoid_is_preserved():
    sfreq = 500.0
    times = np.arange(int(4 * sfreq)) / sfreq
    nearby = 0.5 * np.sin(2 * np.pi * 48 * times)
    raw = _raw((nearby + np.sin(2 * np.pi * 50 * times))[np.newaxis, :], sfreq)

    remove_line_noise(raw, 50, window_length=2)

    np.testing.assert_allclose(raw.get_data()[0], nearby, atol=1e-10)


def test_overlap_add_avoids_window_boundary_artifacts():
    sfreq = 500.0
    times = np.arange(int(10 * sfreq)) / sfreq
    amplitude = 1 + 0.2 * np.sin(2 * np.pi * 0.02 * times)
    phase = 0.1 * np.sin(2 * np.pi * 0.015 * times)
    raw = _raw(
        (amplitude * np.sin(2 * np.pi * 50 * times + phase))[np.newaxis, :], sfreq
    )

    remove_line_noise(raw, 50, window_length=2, overlap=0.5)

    clean = raw.get_data()
    boundaries = np.arange(int(sfreq), clean.shape[-1], int(sfreq))
    assert np.max(np.abs(clean)) < 0.03
    assert np.max(np.abs(np.diff(clean[0])[boundaries - 1])) < 0.002


def test_short_data_uses_the_available_effective_window():
    sfreq = 500.0
    times = np.arange(100) / sfreq
    expected_clean = 0.5 * np.sin(2 * np.pi * 10 * times)
    raw = _raw((expected_clean + np.cos(2 * np.pi * 50 * times))[np.newaxis, :], sfreq)

    remove_line_noise(raw, 50, window_length=10)

    np.testing.assert_allclose(raw.get_data()[0], expected_clean, atol=1e-12)


def test_output_is_deterministic(synthetic_data):
    data, _, sfreq = synthetic_data
    first = _raw(data, sfreq)
    second = _raw(data, sfreq)

    remove_line_noise(first, 50)
    remove_line_noise(second, 50)

    np.testing.assert_array_equal(first.get_data(), second.get_data())
