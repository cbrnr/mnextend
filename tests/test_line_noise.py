# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Tests for line-noise removal."""

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


def _attenuation(line, residual):
    """Return the power of line noise relative to the remaining error in dB."""
    return 10 * np.log10(np.sum(line**2) / np.sum(residual**2))


def _line(phase, n_channels=3):
    """Return line noise with three harmonics and channel-specific amplitude/phase."""
    return np.vstack(
        [
            sum(
                (1 + 0.2 * channel)
                / harmonic
                * np.sin(harmonic * phase + 0.3 * channel * harmonic)
                for harmonic in (1, 2, 3)
            )
            for channel in range(n_channels)
        ]
    )


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
        freq_tolerance=0,
    )
    remove_line_noise(
        with_harmonics,
        50,
        include_harmonics=True,
        window_length=2,
        freq_tolerance=0,
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
        (50, {"freq_tolerance": -0.1}, ValueError, "freq_tolerance"),
        (50, {"freq_tolerance": 50}, ValueError, "freq_tolerance"),
        (50, {"freq_tolerance": "0.1"}, TypeError, "freq_tolerance"),
        (50, {"skip_by_annotation": 1}, TypeError, "skip_by_annotation"),
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

    remove_line_noise(raw, 50, window_length=2, freq_tolerance=0)

    before = _sinusoid_amplitude(data, sfreq, 50)
    after = _sinusoid_amplitude(raw.get_data(), sfreq, 50)
    assert np.all(after < before * 1e-8)


def test_unrelated_sinusoid_is_preserved(synthetic_data):
    data, expected_clean, sfreq = synthetic_data
    raw = _raw(data, sfreq)

    remove_line_noise(raw, 50, window_length=2, freq_tolerance=0)

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

    remove_line_noise(raw, 50, window_length=2, freq_tolerance=0)

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

    remove_line_noise(raw, 50, window_length=10, freq_tolerance=0)

    np.testing.assert_allclose(raw.get_data()[0], expected_clean, atol=1e-12)


def test_output_is_deterministic(synthetic_data):
    data, _, sfreq = synthetic_data
    first = _raw(data, sfreq)
    second = _raw(data, sfreq)

    remove_line_noise(first, 50)
    remove_line_noise(second, 50)

    np.testing.assert_array_equal(first.get_data(), second.get_data())


@pytest.mark.parametrize(
    ("window_length", "duration"), [(2.05, 12), (0.25, 3), (10, 7.33), (10, 1.03)]
)
def test_windows_without_whole_cycles_are_fitted_exactly(window_length, duration):
    sfreq = 500.0
    times = np.arange(int(np.rint(duration * sfreq))) / sfreq
    raw = _raw(_line(2 * np.pi * 50 * times), sfreq)

    remove_line_noise(raw, 50, window_length=window_length, freq_tolerance=0)

    assert np.max(np.abs(raw.get_data())) < 1e-10


def test_zero_tolerance_fits_nominal_frequencies():
    sfreq = 500.0
    times = np.arange(int(2.05 * sfreq)) / sfreq
    data = np.sin(2 * np.pi * 50.07 * times + 0.3)[np.newaxis, :]
    raw = _raw(data, sfreq)
    basis = np.column_stack(
        (np.sin(2 * np.pi * 50 * times), np.cos(2 * np.pi * 50 * times))
    )
    # the offset is fitted jointly with the sinusoids, but it is not subtracted
    design = np.column_stack((basis, np.ones_like(times)))
    coefficients, *_ = np.linalg.lstsq(design, data[0], rcond=None)

    # the recording is shorter than one window, so a single fit is subtracted
    remove_line_noise(raw, 50, include_harmonics=False, freq_tolerance=0)

    np.testing.assert_allclose(raw.get_data()[0], data[0] - basis @ coefficients[:2])


@pytest.mark.parametrize("window_length", [10, 4])
def test_constant_frequency_deviation_is_tracked(window_length):
    sfreq = 500.0
    times = np.arange(int(60 * sfreq)) / sfreq
    line = _line(2 * np.pi * 50.07 * times)
    tracked = _raw(line, sfreq)
    fixed = _raw(line, sfreq)

    remove_line_noise(tracked, 50, window_length=window_length)
    remove_line_noise(fixed, 50, window_length=window_length, freq_tolerance=0)

    assert _attenuation(line, tracked.get_data()) > 90
    assert _attenuation(line, fixed.get_data()) < 20


def test_tracking_matches_fit_at_true_frequency_with_nearby_signals():
    sfreq = 500.0
    times = np.arange(int(60 * sfreq)) / sfreq
    clean = 0.8 * np.sin(2 * np.pi * 10 * times) + 0.5 * np.sin(2 * np.pi * 48 * times)
    data = np.tile(clean, (3, 1)) + _line(2 * np.pi * 50.07 * times)
    tracked = _raw(data, sfreq)
    oracle = _raw(data, sfreq)

    remove_line_noise(tracked, 50)
    remove_line_noise(oracle, 50.07, freq_tolerance=0)

    np.testing.assert_allclose(tracked.get_data(), oracle.get_data(), atol=1e-3)
    np.testing.assert_allclose(
        _sinusoid_amplitude(tracked.get_data(), sfreq, 48), 0.5, rtol=1e-3
    )


@pytest.mark.parametrize("sfreq", [500.0, 512.0])
def test_large_offset_does_not_affect_tracking(sfreq):
    times = np.arange(int(60 * sfreq)) / sfreq
    line = _line(2 * np.pi * 50.04 * times)
    # offsets of unreferenced EEG can be thousands of times larger than line noise
    offset = np.array([[5e3], [-2e3], [8e3]])
    without_offset = _raw(line, sfreq)
    with_offset = _raw(line + offset, sfreq)

    remove_line_noise(without_offset, 50)
    remove_line_noise(with_offset, 50)

    assert _attenuation(line, with_offset.get_data() - offset) > 90
    np.testing.assert_allclose(
        with_offset.get_data() - offset, without_offset.get_data(), atol=1e-6
    )


def test_wandering_frequency_is_tracked():
    sfreq = 500.0
    times = np.arange(int(120 * sfreq)) / sfreq
    rng = np.random.default_rng(0)
    walk = np.convolve(
        np.cumsum(rng.standard_normal(times.size)), np.ones(2500) / 2500, "same"
    )
    deviation = 0.03 * (walk - walk.mean()) / walk.std()
    line = _line(2 * np.pi * np.cumsum(50 + deviation) / sfreq)
    tracked = _raw(line, sfreq)
    fixed = _raw(line, sfreq)

    remove_line_noise(tracked, 50)
    remove_line_noise(fixed, 50, freq_tolerance=0)

    assert _attenuation(line, tracked.get_data()) > 20
    assert _attenuation(line, fixed.get_data()) < 10


def test_shared_harmonics_of_several_fundamentals_are_tracked():
    sfreq = 1000.0
    times = np.arange(int(30 * sfreq)) / sfreq
    # the 50 Hz and 60 Hz families share the harmonic at 300 Hz
    line = _line(2 * np.pi * 50.04 * times) + _line(2 * np.pi * 59.95 * times + 1)
    raw = _raw(line, sfreq)

    remove_line_noise(raw, [50, 60, 100])

    assert _attenuation(line, raw.get_data()) > 60


def test_annotated_nan_span_is_skipped():
    sfreq = 500.0
    times = np.arange(int(30 * sfreq)) / sfreq
    clean = np.tile(0.8 * np.sin(2 * np.pi * 10 * times), (3, 1))
    line = _line(2 * np.pi * 50.03 * times)
    gap = (times >= 12.3) & (times < 13.1)
    data = clean + line
    data[:, gap] = np.nan
    raw = _raw(data, sfreq)
    raw.set_annotations(mne.Annotations(12.3, 0.8, "BAD_ACQ_SKIP"))

    remove_line_noise(raw, 50)

    cleaned = raw.get_data()
    np.testing.assert_array_equal(np.isnan(cleaned), np.isnan(data))
    assert _attenuation(line[:, ~gap], cleaned[:, ~gap] - clean[:, ~gap]) > 60

    unskipped = _raw(data, sfreq)
    unskipped.set_annotations(raw.annotations)
    remove_line_noise(unskipped, 50, skip_by_annotation=[])
    assert np.isnan(unskipped.get_data()).sum() > np.isnan(data).sum()


def test_annotated_span_is_left_unchanged():
    sfreq = 500.0
    times = np.arange(int(20 * sfreq)) / sfreq
    data = _line(2 * np.pi * 50 * times)
    skipped = (times >= 5) & (times < 7)
    raw = _raw(data, sfreq)
    raw.set_annotations(mne.Annotations(5, 2, "bad_acq_skip"))

    remove_line_noise(raw, 50)

    np.testing.assert_array_equal(raw.get_data()[:, skipped], data[:, skipped])
    assert np.max(np.abs(raw.get_data()[:, ~skipped])) < 1e-3


def test_concatenated_recordings_are_cleaned_separately():
    sfreq = 500.0
    times = np.arange(int(15 * sfreq)) / sfreq
    first = _line(2 * np.pi * 50.05 * times)
    second = _line(2 * np.pi * 49.95 * times + 2)
    raws = [_raw(first, sfreq), _raw(second, sfreq)]
    raw = mne.concatenate_raws(raws, verbose=False)
    joined = mne.concatenate_raws(
        [_raw(first, sfreq), _raw(second, sfreq)], verbose=False
    )

    remove_line_noise(raw, 50)
    remove_line_noise(joined, 50, skip_by_annotation=[])

    line = np.hstack((first, second))
    assert _attenuation(line, raw.get_data()) > 60
    assert _attenuation(line, joined.get_data()) < 40


def test_too_short_segment_is_left_unchanged_with_warning():
    sfreq = 500.0
    times = np.arange(int(10 * sfreq)) / sfreq
    data = _line(2 * np.pi * 50 * times)
    raw = _raw(data, sfreq)
    # only the last sample is not annotated
    raw.set_annotations(mne.Annotations(0, times[-1], "BAD_ACQ_SKIP"))

    with pytest.warns(RuntimeWarning, match="too short"):
        remove_line_noise(raw, 50)

    np.testing.assert_array_equal(raw.get_data(), data)
