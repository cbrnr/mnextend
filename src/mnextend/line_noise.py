# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Subtraction of line-noise sinusoids with per-window frequency tracking."""

import numpy as np
from mne._fiff.pick import _picks_to_idx
from mne.annotations import _annotations_starts_stops
from mne.io import BaseRaw
from mne.utils import _check_preload, warn
from scipy.fft import fft, ifft, next_fast_len
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar

# Keep temporary window arrays bounded for long, high-density recordings. The output
# itself necessarily has the size of the selected data.
_MAX_WORKING_SET_BYTES = 64 * 1024**2
# minimum number of block sums per window used to estimate line frequencies
_MIN_BLOCKS = 64


def remove_line_noise(
    raw,
    line_freq,
    *,
    picks=None,
    include_harmonics=True,
    window_length=10.0,
    overlap=0.5,
    freq_tolerance=0.1,
    skip_by_annotation=("edge", "bad_acq_skip"),
):
    """Subtract fitted line-noise sinusoids from a Raw object in place.

    The line frequencies are fitted with sine and cosine components in overlapping time
    windows. The fitted components are subtracted with a weighted overlap-add
    reconstruction, allowing line amplitude and phase to vary over time. A constant
    offset is fitted jointly in each window (but not subtracted), so that large offsets
    (for example, in unreferenced or unfiltered data) do not distort the fitted line
    noise.

    The actual frequency of the power grid deviates slightly from its nominal value and
    drifts over time. Therefore, each fundamental frequency is estimated separately in
    every window within `freq_tolerance` of its nominal value, and its harmonics follow
    as multiples of the estimated frequency. The estimate maximizes the power at the
    fundamental and its harmonics, summed over all selected channels.

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
        Regression-window duration in seconds. Line frequency, amplitude, and phase are
        assumed to be constant within a window, so shorter windows follow faster
        changes, but they also remove more signal close to the line frequencies. The
        effective duration can differ because it is rounded to samples and is limited by
        shorter data segments.
    overlap : float
        Fractional overlap between successive regression windows. It must be in the
        interval [0, 1).
    freq_tolerance : float
        Maximum deviation in Hz of each fundamental frequency from its nominal value
        `line_freq`. The deviation is estimated separately in each window, and harmonics
        deviate by the corresponding multiple. If `0`, only the nominal frequencies are
        fitted.
    skip_by_annotation : str | list of str
        Annotations whose descriptions start with any of these strings (ignoring case)
        mark spans that are left unchanged, and the data on either side are cleaned
        independently. The default skips boundaries between concatenated recordings
        (`"edge"`) and acquisition gaps (`"bad_acq_skip"`), which may contain NaN. To
        clean the data in a single segment, use an empty list.

    Returns
    -------
    raw : instance of mne.io.BaseRaw
        The same instance, after modifying the selected channels in place.

    Notes
    -----
    NaN values in a window make the cleaned data of all overlapping windows NaN, so
    spans containing NaN should be excluded with `skip_by_annotation`. Segments between
    skipped spans that are too short to fit all frequencies are left unchanged (with a
    warning).
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw must be an instance of mne.io.BaseRaw.")
    _check_preload(raw, "remove_line_noise")
    pick_indices = _picks_to_idx(raw.info, picks, exclude=(), none="data_or_ica")
    sfreq = _validate_real_scalar(raw.info["sfreq"], "sfreq", positive=True)
    families = _prepare_families(line_freq, sfreq, include_harmonics)
    window_samples, overlap = _prepare_windows(window_length, overlap, sfreq)
    freq_tolerance = _prepare_tolerance(freq_tolerance, families)
    if _line_basis(window_samples, sfreq, _frequencies(families)) is None:
        raise ValueError(
            "The requested frequencies are not independently estimable with this "
            "window_length and sfreq."
        )

    onsets, ends = _annotations_starts_stops(raw, skip_by_annotation, invert=True)
    skipped = []
    for onset, end in zip(onsets, ends):
        if end == onset:
            continue
        clean = _clean_segment(
            raw._data[:, onset:end],
            pick_indices,
            sfreq,
            families,
            window_samples=window_samples,
            overlap=overlap,
            freq_tolerance=freq_tolerance,
        )
        if clean is None:
            skipped.append(end - onset)
        else:
            raw._data[pick_indices, onset:end] = clean
    if skipped:
        warn(
            f"Left {len(skipped)} data segment(s) unchanged ({sum(skipped)} samples in "
            "total), because they are too short to fit the requested frequencies."
        )
    return raw


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


def _prepare_families(line_freq, sfreq, include_harmonics):
    """Validate fundamental frequencies and return them with their harmonic numbers.

    Each frequency belongs to the lowest fundamental that generates it, so harmonics
    shared by several fundamentals are fitted only once.
    """
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

    families = []
    assigned = set()
    max_frequency = np.nextafter(nyquist, 0.0)
    for fundamental in np.unique(fundamentals):
        n_harmonics = int(np.floor(max_frequency / fundamental))
        multipliers = np.arange(1, n_harmonics + 1 if include_harmonics else 2)
        multipliers = multipliers[~np.isin(fundamental * multipliers, list(assigned))]
        if multipliers.size:
            assigned.update(fundamental * multipliers)
            families.append((fundamental, multipliers))
    return families


def _frequencies(families):
    """Return the nominal frequencies of all families in fitting order."""
    return np.concatenate(
        [fundamental * multipliers for fundamental, multipliers in families]
    )


def _prepare_windows(window_length, overlap, sfreq):
    """Validate window parameters and return the window length in samples."""
    window_length = _validate_real_scalar(window_length, "window_length", positive=True)
    window_samples = int(np.rint(window_length * sfreq))
    if window_samples < 2:
        raise ValueError(
            "window_length and sfreq must yield at least two samples per window."
        )

    overlap = _validate_real_scalar(overlap, "overlap")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be greater than or equal to 0 and less than 1.")
    if int(np.rint(window_samples * (1 - overlap))) < 1:
        raise ValueError("overlap leaves fewer than one sample between window starts.")
    return window_samples, overlap


def _prepare_tolerance(freq_tolerance, families):
    """Validate the frequency tolerance against the lowest fundamental frequency."""
    freq_tolerance = _validate_real_scalar(freq_tolerance, "freq_tolerance")
    if freq_tolerance < 0:
        raise ValueError("freq_tolerance must be greater than or equal to 0.")
    if freq_tolerance >= families[0][0]:
        raise ValueError("freq_tolerance must be less than the lowest line_freq.")
    return freq_tolerance


def _make_basis(window_samples, sfreq, frequencies):
    """Construct the sinusoid design matrix for one window shape."""
    times = np.arange(window_samples, dtype=float) / sfreq
    angles = 2 * np.pi * times[:, np.newaxis] * frequencies
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


def _line_basis(window_samples, sfreq, frequencies):
    """Return matrices to fit the sinusoids jointly with an offset, or `None`.

    The sinusoids are demeaned and then orthonormalized, which yields `q_factor`. For
    data `y`, the least-squares fit of the sinusoids in the joint model is
    `(y @ q_factor) @ sinusoids.T`, so the offset is fitted but not returned. `None`
    indicates that the sinusoids are not independently estimable.
    """
    basis = _make_basis(window_samples, sfreq, frequencies)
    demeaned = basis - basis.mean(axis=0)
    # Cholesky QR applied twice is much faster than Householder QR for these tall
    # matrices, and it is just as accurate if the basis is well-conditioned
    try:
        lower = np.linalg.cholesky(demeaned.T @ demeaned)
    except np.linalg.LinAlgError:
        lower = None
    if lower is None or np.linalg.cond(lower) > 1e4:
        q_factor, r_factor = np.linalg.qr(demeaned, mode="reduced")
        if np.linalg.matrix_rank(r_factor) != basis.shape[1]:
            return None
    else:
        q_factor = solve_triangular(lower, demeaned.T, lower=True, check_finite=False)
        lower = np.linalg.cholesky(q_factor @ q_factor.T)
        q_factor = solve_triangular(lower, q_factor, lower=True, check_finite=False).T
    # demeaned = q_factor @ r with triangular r = q_factor.T @ basis, so coefficients c
    # of q_factor correspond to coefficients inv(r) @ c of the sinusoids
    sinusoids = np.linalg.solve(basis.T @ q_factor, basis.T).T
    return q_factor, sinusoids


def _window_starts(n_times, window_samples, hop):
    """Return starts for complete windows."""
    if n_times <= window_samples:
        return np.array([0])

    starts = np.arange(0, n_times - window_samples + 1, hop)
    final_start = n_times - window_samples
    if starts[-1] != final_start:
        starts = np.append(starts, final_start)
    return starts


def _clean_segment(
    data, pick_indices, sfreq, families, *, window_samples, overlap, freq_tolerance
):
    """Return the cleaned picks of one contiguous segment, or `None` if it is short."""
    n_times = data.shape[1]
    window_samples = min(window_samples, n_times)
    if window_samples < 2:
        return None
    nominal = _line_basis(window_samples, sfreq, _frequencies(families))
    if nominal is None:
        return None

    hop = max(1, int(np.rint(window_samples * (1 - overlap))))
    starts = _window_starts(n_times, window_samples, hop)
    window = np.hanning(window_samples + 2)[1:-1]
    weights = np.zeros(n_times)
    for start in starts:
        weights[start : start + window_samples] += window

    # three arrays of this size are live at once: data, fitted data, and residuals
    batch_size = max(
        1, _MAX_WORKING_SET_BYTES // (3 * window_samples * data.dtype.itemsize)
    )
    batches = [
        pick_indices[first : first + batch_size]
        for first in range(0, len(pick_indices), batch_size)
    ]
    tracker = None
    if freq_tolerance > 0:
        tracker = _FrequencyTracker(window_samples, sfreq, families, freq_tolerance)

    clean = np.zeros((len(pick_indices), n_times), dtype=data.dtype)
    for start in starts:
        stop = start + window_samples
        q_factor, sinusoids = nominal
        if tracker is not None:
            frequencies = tracker.estimate(
                data[channels, start:stop] for channels in batches
            )
            q_factor, sinusoids = (
                _line_basis(window_samples, sfreq, frequencies) or nominal
            )
        first = 0
        for channels in batches:
            segments = data[channels, start:stop]
            residuals = segments - (segments @ q_factor) @ sinusoids.T
            clean[first : first + len(channels), start:stop] += residuals * window
            first += len(channels)
    return clean / weights


class _FrequencyTracker:
    """Estimate the line frequencies in windows of a fixed length.

    For each family of harmonics `k * f0`, the deviation `delta` maximizes the power at
    `k * (f0 + delta)` summed over the harmonics and channels. To evaluate this power
    cheaply for many values of `delta`, the data are demodulated at each nominal
    frequency and summed in blocks that are short enough that `k * delta` cannot rotate
    the phase much within a block. Each window is demeaned first, because the block sums
    do not suppress large offsets enough to prevent them from aliasing close to the line
    frequencies. The power is then a trigonometric polynomial in `delta` whose
    coefficients are the autocorrelations of the Hann-tapered block sums. It is
    evaluated on a grid fine enough to sample the main lobe of the highest harmonic, and
    the best grid point is refined with a bounded scalar optimization.
    """

    def __init__(self, window_samples, sfreq, families, tolerance):
        self.sfreq = sfreq
        self.tolerance = tolerance
        frequencies = _frequencies(families)
        max_multiplier = max(multipliers[-1] for _, multipliers in families)

        # keep the block rate at least eight times above the largest frequency offset,
        # and use enough blocks that the taper suppresses leakage from other components
        block = int(sfreq // (8 * max_multiplier * tolerance))
        self.block = max(1, min(block, window_samples // _MIN_BLOCKS))
        self.n_blocks = window_samples // self.block
        angles = 2 * np.pi * np.arange(self.block)[:, np.newaxis] * frequencies / sfreq
        self.demodulator = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
        block_times = np.arange(self.n_blocks) * self.block / sfreq
        taper = np.hanning(self.n_blocks + 2)[1:-1, np.newaxis]
        self.block_phases = taper * np.exp(
            -2j * np.pi * block_times[:, None] * frequencies
        )
        self.n_fft = next_fast_len(2 * self.n_blocks - 1)

        self.families = []
        lags = np.arange(1, self.n_blocks)
        first = 0
        for fundamental, multipliers in families:
            columns = slice(first, first + multipliers.size)
            first += multipliers.size
            # the grid spacing is a quarter of the main lobe width of the highest
            # harmonic, (sfreq / block) / n_grid = sfreq / (4 * n_blocks * block * k)
            n_grid = 4 * self.n_blocks * multipliers[-1]
            self.families.append(
                (fundamental, multipliers, columns, lags[:, None] * multipliers, n_grid)
            )

    def estimate(self, segments):
        """Return the estimated frequencies for one window of all picked channels."""
        n_frequencies = self.demodulator.shape[1] // 2
        power = np.zeros((self.n_fft, n_frequencies))
        used = self.n_blocks * self.block
        for segment in segments:
            segment = segment - segment.mean(axis=1, keepdims=True)
            blocks = segment[:, :used].reshape(len(segment), self.n_blocks, self.block)
            sums = blocks @ self.demodulator
            sums = (sums[..., :n_frequencies] - 1j * sums[..., n_frequencies:]) * (
                self.block_phases
            )
            power += np.sum(np.abs(fft(sums, n=self.n_fft, axis=1)) ** 2, axis=0)
        autocorrelation = ifft(power, axis=0)[: self.n_blocks]

        frequencies = np.empty(n_frequencies)
        for fundamental, multipliers, columns, lag_multiples, n_grid in self.families:
            delta = self._estimate_deviation(
                autocorrelation[:, columns], lag_multiples, n_grid
            )
            frequencies[columns] = multipliers * (fundamental + delta)
        return frequencies

    def _estimate_deviation(self, autocorrelation, lag_multiples, n_grid):
        """Return the deviation that maximizes the power of one family."""
        zero_lag = np.sum(autocorrelation[0].real)
        if not np.isfinite(zero_lag) or zero_lag <= 0:
            return 0.0
        coefficients = autocorrelation[1:]
        # rotation per block of harmonic k for a deviation of delta is k * delta * scale
        scale = 2 * np.pi * self.block / self.sfreq

        def power(delta):
            phases = np.exp(-1j * scale * delta * lag_multiples)
            return zero_lag + 2 * np.sum((coefficients * phases).real)

        # evaluate the power at the grid deviations j * step with a single FFT
        polynomial = np.zeros(n_grid, dtype=complex)
        np.add.at(polynomial, lag_multiples, coefficients)
        grid_power = 2 * fft(polynomial).real + zero_lag
        step = 2 * np.pi / (n_grid * scale)
        n_side = int(self.tolerance // step)
        grid = np.arange(-n_side, n_side + 1)
        best = grid[np.argmax(grid_power[grid % n_grid])] * step

        result = minimize_scalar(
            lambda delta: -power(delta),
            bounds=(
                max(-self.tolerance, best - step),
                min(self.tolerance, best + step),
            ),
            method="bounded",
            options={"xatol": 1e-4 * step},
        )
        return result.x
