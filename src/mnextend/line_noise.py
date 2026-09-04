# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Adaptive subtraction of known line-noise sinusoids."""

import numpy as np
from mne._fiff.pick import _picks_to_idx
from mne.io import BaseRaw
from mne.utils import _check_preload

# Keep temporary window arrays bounded for long, high-density recordings. The output
# itself necessarily has the size of the selected data.
_MAX_WORKING_SET_BYTES = 64 * 1024**2
_MAX_WINDOW_BATCH_SIZE = 32


def remove_line_noise(
    raw,
    line_freq,
    *,
    picks=None,
    include_harmonics=True,
    window_length=10.0,
    overlap=0.5,
):
    """Subtract fitted line-noise sinusoids from a Raw object in place.

    The supplied line frequencies are fitted with sine and cosine components in
    overlapping time windows. The fitted components are subtracted with a weighted
    overlap-add reconstruction, allowing line amplitude and phase to vary over time.

    Parameters
    ----------
    raw : instance of mne.io.BaseRaw
        Continuous data to clean. It must be preloaded, for example with `preload=True`
        or `raw.load_data()`. The selected channels are modified in place.
    line_freq : float | array-like of float
        Fundamental line frequency or frequencies in Hz. Every value must be strictly
        between zero and the Nyquist frequency.
    picks : array-like of int | array-like of str | slice | str | None
        Channels to clean. `None` selects MNE data or ICA channels, matching
        `Raw.notch_filter`.
    include_harmonics : bool
        If `True`, fit every positive integer harmonic below Nyquist as well as the
        supplied fundamental frequencies.
    window_length : float
        Regression-window duration in seconds. The effective duration can differ because
        it is rounded to samples and is limited by a shorter recording.
    overlap : float
        Fractional overlap between successive regression windows. It must be in the
        interval [0, 1).

    Returns
    -------
    raw : instance of mne.io.BaseRaw
        The same instance, after modifying the selected channels in place.
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw must be an instance of mne.io.BaseRaw.")
    _check_preload(raw, "remove_line_noise")
    pick_indices = _picks_to_idx(raw.info, picks, exclude=(), none="data_or_ica")
    raw._data[pick_indices] = _remove_line_noise(
        raw._data,
        pick_indices,
        raw.info["sfreq"],
        line_freq,
        include_harmonics=include_harmonics,
        window_length=window_length,
        overlap=overlap,
    )
    return raw


def _remove_line_noise(
    data,
    pick_indices,
    sfreq,
    line_freq,
    *,
    include_harmonics,
    window_length,
    overlap,
):
    """Return cleaned data for the public in-place Raw wrapper."""
    sfreq = _validate_real_scalar(sfreq, "sfreq", positive=True)
    frequencies = _prepare_frequencies(line_freq, sfreq, include_harmonics)
    window_length = _validate_real_scalar(window_length, "window_length", positive=True)
    window_samples = min(int(np.rint(window_length * sfreq)), data.shape[1])
    if window_samples < 2:
        raise ValueError(
            "window_length and sfreq must yield at least two samples per window."
        )

    overlap = _validate_real_scalar(overlap, "overlap")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be greater than or equal to 0 and less than 1.")
    hop = int(np.rint(window_samples * (1 - overlap)))
    if hop < 1:
        raise ValueError("overlap leaves fewer than one sample between window starts.")

    basis = _make_basis(window_samples, sfreq, frequencies)
    q_factor, r_factor = np.linalg.qr(basis, mode="reduced")
    if np.linalg.matrix_rank(r_factor) != basis.shape[1]:
        raise ValueError(
            "The requested frequencies are not independently estimable with this "
            "window_length and sfreq."
        )

    window = np.hanning(window_samples + 2)[1:-1]
    starts = _window_starts(data.shape[1], window_samples, hop)
    return _remove_windows(
        data, pick_indices, basis, q_factor, r_factor, starts, window
    )


def _validate_real_scalar(value, name, *, positive=False):
    """Return a finite real scalar, optionally requiring it to be positive."""
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "fiu" or scalar.dtype.kind == "b":
        raise TypeError(f"{name} must be a real scalar.")
    result = float(scalar)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return result


def _prepare_frequencies(line_freq, sfreq, include_harmonics):
    """Validate fundamental frequencies and expand their valid harmonics."""
    if not isinstance(include_harmonics, (bool, np.bool_)):
        raise TypeError("include_harmonics must be a boolean.")

    values = np.asarray(line_freq)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or values.dtype.kind not in "fiu" or values.dtype.kind == "b":
        raise TypeError("line_freq must be a real scalar or one-dimensional array.")
    if values.size == 0:
        raise ValueError("line_freq must contain at least one frequency.")

    fundamentals = values.astype(float, copy=False)
    if not np.isfinite(fundamentals).all():
        raise ValueError("line_freq must contain only finite values.")
    nyquist = sfreq / 2
    if np.any(fundamentals <= 0) or np.any(fundamentals >= nyquist):
        raise ValueError("line_freq values must be greater than 0 and below Nyquist.")

    frequencies = np.unique(fundamentals)
    if include_harmonics:
        max_frequency = np.nextafter(nyquist, 0.0)
        frequencies = np.unique(
            np.concatenate(
                [
                    fundamental
                    * np.arange(1, int(np.floor(max_frequency / fundamental)) + 1)
                    for fundamental in frequencies
                ]
            )
        )
    return frequencies


def _make_basis(window_samples, sfreq, frequencies):
    """Construct the sinusoid design matrix for one window shape."""
    times = np.arange(window_samples, dtype=float) / sfreq
    angles = 2 * np.pi * times[:, np.newaxis] * frequencies
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


def _window_starts(n_times, window_samples, hop):
    """Return starts for complete windows."""
    if n_times <= window_samples:
        return np.array([0])

    starts = np.arange(0, n_times - window_samples + 1, hop)
    final_start = n_times - window_samples
    if starts[-1] != final_start:
        starts = np.append(starts, final_start)
    return starts


def _remove_windows(data, pick_indices, basis, q_factor, r_factor, starts, window):
    """Fit all channels in bounded window/channel batches using a cached QR basis."""
    n_channels = len(pick_indices)
    n_times = data.shape[1]
    window_samples, n_parameters = basis.shape
    clean = np.zeros((n_channels, n_times), dtype=data.dtype)
    weights = np.zeros(n_times)
    for start in starts:
        stop = min(start + window_samples, n_times)
        weights[start:stop] += window[: stop - start]

    # Four arrays of this size are live at once: data, coefficients, fitted data, and
    # residuals. Split first by windows and then by channels when this estimate is big.
    window_batch_size = min(_MAX_WINDOW_BATCH_SIZE, len(starts))
    channel_batch_size = max(
        1,
        _MAX_WORKING_SET_BYTES
        // (4 * window_batch_size * window_samples * data.dtype.itemsize),
    )
    sample_offsets = np.arange(window_samples)

    for first_window in range(0, len(starts), window_batch_size):
        batch_starts = starts[first_window : first_window + window_batch_size]
        sample_indices = batch_starts[:, np.newaxis] + sample_offsets
        for first_channel in range(0, n_channels, channel_batch_size):
            last_channel = min(first_channel + channel_batch_size, n_channels)
            channel_indices = pick_indices[first_channel:last_channel]
            segments = data[
                channel_indices[:, np.newaxis, np.newaxis],
                sample_indices[np.newaxis, :, :],
            ]
            projection = segments @ q_factor
            coefficients = np.linalg.solve(
                r_factor.T, projection.reshape(-1, n_parameters).T
            ).T.reshape(projection.shape)
            residuals = segments - coefficients @ basis.T

            for window_index, start in enumerate(batch_starts):
                stop = min(start + window_samples, n_times)
                width = stop - start
                clean[first_channel:last_channel, start:stop] += (
                    residuals[:, window_index, :width] * window[:width]
                )

    return clean / weights
