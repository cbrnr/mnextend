# © MNEXTEND developers
#
# License: BSD (3-clause)

import struct
import warnings
import xml.etree.ElementTree as ETree
from collections import defaultdict
from datetime import UTC, datetime

import mne
import numpy as np
import scipy.signal
from mne.io import BaseRaw, get_channel_type_constants
from mne.io.constants import FIFF
from pyxdf import load_xdf
from pyxdf.pyxdf import _read_varlen_int, open_xdf

# maps base unit symbols (case-sensitive) to (FIFF unit code or None, accepts prefixes);
# degrees are not converted to radians and MNE has no unit for them, so they keep the
# default
_UNIT_SYMBOLS = {
    "V": (FIFF.FIFF_UNIT_V, True),
    "T": (FIFF.FIFF_UNIT_T, True),
    "m": (FIFF.FIFF_UNIT_M, True),
    "rad": (FIFF.FIFF_UNIT_RAD, True),
    "deg": (None, False),
    "°": (None, False),
    "S": (FIFF.FIFF_UNIT_S, True),
    "°C": (FIFF.FIFF_UNIT_CEL, True),
    "K": (FIFF.FIFF_UNIT_K, True),
    "s": (FIFF.FIFF_UNIT_SEC, True),
    "Hz": (FIFF.FIFF_UNIT_HZ, True),
    "px": (FIFF.FIFF_UNIT_PX, False),
}

# maps SI prefix symbols (case-sensitive) to their scale; the Greek mu is normalized to
# the micro sign before lookup and `u` is a common ASCII substitute
_PREFIXES = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "µ": 1e-6,
    "u": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "k": 1e3,
}

# maps unit words (lower case, matched case-insensitively) like the symbols above
_UNIT_WORDS = {
    **{
        word: (scale, FIFF.FIFF_UNIT_V)
        for words, scale in [
            (("volt", "volts"), 1.0),
            (("millivolt", "millivolts"), 1e-3),
            (("microvolt", "microvolts"), 1e-6),
            (("nanovolt", "nanovolts"), 1e-9),
        ]
        for word in words
    },
    **{
        word: (scale, FIFF.FIFF_UNIT_T)
        for words, scale in [
            (("tesla",), 1.0),
            (("picotesla",), 1e-12),
            (("femtotesla",), 1e-15),
        ]
        for word in words
    },
    **{
        word: (scale, FIFF.FIFF_UNIT_M)
        for words, scale in [
            (("meter", "meters", "metre", "metres"), 1.0),
            (("centimeter", "centimeters", "centimetre", "centimetres"), 1e-2),
            (("millimeter", "millimeters", "millimetre", "millimetres"), 1e-3),
        ]
        for word in words
    },
    "radian": (1.0, FIFF.FIFF_UNIT_RAD),
    "radians": (1.0, FIFF.FIFF_UNIT_RAD),
    "degree": (1.0, None),
    "degrees": (1.0, None),
    "siemens": (1.0, FIFF.FIFF_UNIT_S),
    "microsiemens": (1e-6, FIFF.FIFF_UNIT_S),
    "celsius": (1.0, FIFF.FIFF_UNIT_CEL),
    "degrees celsius": (1.0, FIFF.FIFF_UNIT_CEL),
    "kelvin": (1.0, FIFF.FIFF_UNIT_K),
    "second": (1.0, FIFF.FIFF_UNIT_SEC),
    "seconds": (1.0, FIFF.FIFF_UNIT_SEC),
    "millisecond": (1e-3, FIFF.FIFF_UNIT_SEC),
    "milliseconds": (1e-3, FIFF.FIFF_UNIT_SEC),
    "hertz": (1.0, FIFF.FIFF_UNIT_HZ),
    "pixel": (1.0, FIFF.FIFF_UNIT_PX),
    "pixels": (1.0, FIFF.FIFF_UNIT_PX),
}

# returned by `_parse_unit` for unit strings that are not recognized
_UNKNOWN_UNIT = object()

# stream types (lower case) that mark a regular-rate numeric stream as discrete
_EVENT_STREAM_TYPES = {
    "marker",
    "markers",
    "event",
    "events",
    "trigger",
    "triggers",
    "stim",
}

# accepted modes in `streams` (full names and abbreviations)
_MODES = {
    "continuous": "continuous",
    "c": "continuous",
    "discrete": "discrete",
    "d": "discrete",
    "annotations": "annotations",
    "a": "annotations",
}


class RawXDF(BaseRaw):
    """Raw data from .xdf file."""

    def __init__(
        self,
        fname,
        streams=None,
        *,
        fs_new=None,
        gap_threshold=0.0,
        prefix_markers=False,
        stream_ids=None,
        marker_ids=None,
        **kwargs,
    ):
        """Read raw data from .xdf file.

        Parameters
        ----------
        fname : str | Path
            File name to load.
        streams : int | list[int] | dict[int, str] | None
            Streams to load (all other streams are ignored). A list (or a single stream
            ID) loads each stream in its default mode. A dict maps stream IDs to modes,
            which can be "continuous" ("c"), "discrete" ("d"), or "annotations" ("a").
            If `None`, all streams containing samples are loaded in their default modes.
            At least one stream must be loaded as channels (continuous or discrete).
            Use `resolve_streams(fname)` to list available streams and
            `stream_modes()` to get the possible modes of a stream (see Notes).
        fs_new : float | None
            Target sampling frequency in Hz (required when loading multiple streams as
            channels or a stream with a nominal sampling frequency of 0 Hz). If only
            one stream is loaded as channels, this can be `None`, in which case the
            stream's original sampling rate is used.
        gap_threshold : float
            Detect gaps in timestamps larger than this value (in seconds) and mark those
            samples as NaN. Set to 0.0 to disable gap detection. If `gap_threshold > 0`,
            linear interpolation is used instead of resampling, and `fs_new` must be
            specified.
        prefix_markers : bool
            Whether to prefix annotations with the ID of their stream.
        stream_ids : int | list[int] | None
            Deprecated, use `streams` instead. IDs of streams to load as channels.
        marker_ids : list[int] | None
            Deprecated, use `streams` instead. IDs of string streams and numeric
            streams with 0 Hz to convert to annotations (`None` converts all such
            streams that are not listed in `stream_ids`).

        Notes
        -----
        Each stream is loaded in one of the following modes:
        - "continuous": channels containing signals (such as EEG), which are filtered
          and resampled as usual.
        - "discrete": channels containing discrete values (such as trigger codes),
          which are never filtered and are resampled by holding the previous value, so
          that they only contain values from the original data. A `RuntimeWarning` is
          issued if value changes are lost due to downsampling. Discrete channels of
          streams with a regular sampling frequency get the channel type "stim" unless
          the stream specifies a valid type.
        - "annotations": one annotation per non-empty string (string streams) or per
          sample (numeric streams, the value is the description).

        Which modes are possible depends on the stream (the first one is the default):
        - String streams: "annotations".
        - Numeric streams with a nominal sampling frequency of 0 Hz: "annotations",
          "discrete".
        - Numeric streams with a regular sampling frequency: "discrete", "continuous"
          if the stream type is one of "Marker(s)", "Event(s)", "Trigger(s)", or "Stim"
          (case-insensitive), otherwise "continuous", "discrete".

        Channels with the type "stim" are always discrete, even in continuous streams.
        "stim" channels contain 0 instead of NaN outside their stream's time range and
        in detected gaps.

        Resampling of continuous channels depends on whether gap detection is requested
        or not:
        - If `gap_threshold > 0`, uses linear interpolation to resample to the new
          sampling frequency `fs_new`. This method will detect gaps in the original
          timestamps and mark those samples as NaN.
        - If `gap_threshold == 0`, uses Fourier-based resampling if `fs_new` is provided
          or does not resample at all if `fs_new` is `None`. This method assumes that
          the original timestamps are regular and does not account for any gaps.
        By default, gap detection is disabled.

        Data are converted to SI units (for example, microvolts to volts and millimeters
        to meters) based on the unit of each channel, and the corresponding MNE unit is
        stored in `info["chs"][i]["unit"]`. Channels without a unit (empty, `"NA"` or
        missing) are left unchanged and keep MNE's default unit for their channel type.
        Angles in degrees are recognized, but neither converted to radians nor assigned
        a unit because MNE has no unit for degrees. A `RuntimeWarning` lists all unit
        strings that are not recognized (the corresponding channels are not scaled).
        """
        if gap_threshold < 0:
            raise ValueError(
                f"Argument `gap_threshold` must be non-negative, got {gap_threshold}."
            )

        if gap_threshold > 0 and fs_new is None:
            raise ValueError(
                "Argument `fs_new` is required when `gap_threshold > 0`. "
                "Gap detection requires resampling to a regular time grid."
            )

        xdf_streams, header = load_xdf(fname)
        xdf_streams = {stream["info"]["stream_id"]: stream for stream in xdf_streams}
        infos = {i: _stream_info(stream) for i, stream in xdf_streams.items()}

        if stream_ids is not None or marker_ids is not None:
            if streams is not None:
                raise ValueError(
                    "Arguments `stream_ids` and `marker_ids` cannot be combined with "
                    "`streams`."
                )
            streams = _streams_from_ids(stream_ids, marker_ids, infos)
            warnings.warn(
                "Arguments `stream_ids` and `marker_ids` are deprecated and will be "
                "removed in MNEXTEND 0.6.0, use `streams` instead.",
                FutureWarning,
                stacklevel=2,
            )

        select_all = streams is None
        if select_all:  # empty streams would only raise errors
            streams = [i for i, s in xdf_streams.items() if len(s["time_stamps"])]
        modes = _parse_streams(streams, infos)
        channel_ids = [i for i, mode in modes.items() if mode != "annotations"]
        annotation_ids = [i for i, mode in modes.items() if mode == "annotations"]
        streams = xdf_streams  # the selection is now stored in `modes`

        if not channel_ids:
            raise ValueError(
                "At least one stream must be loaded as channels (mode 'continuous' or "
                "'discrete')."
            )

        if len(channel_ids) > 1 and fs_new is None:
            msg = (
                "Argument `fs_new` is required when loading multiple streams as "
                "channels."
            )
            if select_all:
                msg += (
                    " All streams are loaded if `streams` is not specified, so specify "
                    "`fs_new` or select streams with `streams`. Available streams and "
                    "their possible modes (default first):\n"
                    f"{_describe_streams(infos)}"
                )
            raise ValueError(msg)

        if fs_new is None and _stream_srate(streams[channel_ids[0]]) == 0:
            raise ValueError(
                "Argument `fs_new` is required when loading a stream with a nominal "
                "sampling frequency of 0 Hz as channels."
            )

        labels_all, types_all, units_all, discrete_all = [], [], [], []
        channel_types = get_channel_type_constants(True)
        for stream_id in channel_ids:
            stream = streams[stream_id]

            n_chans = int(stream["info"]["channel_count"][0])
            labels, types, units = [], [], []
            try:
                for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
                    labels.append(str(ch["label"][0]))
                    if ch["type"] and ch["type"][0].lower() in channel_types:
                        types.append(ch["type"][0].lower())
                    else:
                        types.append(None)
                    units.append(ch["unit"][0] if ch["unit"] else "NA")
            except (TypeError, IndexError):  # no channel labels found
                pass
            if not labels:
                labels = [f"{stream['info']['name'][0]}_{n}" for n in range(n_chans)]
            if not units:
                units = ["NA" for _ in range(n_chans)]
            if not types:
                types = [None for _ in range(n_chans)]
            discrete = [modes[stream_id] == "discrete" or t == "stim" for t in types]
            # discrete channels of regular streams default to stim, irregular streams
            # are often sensor values (like heart rate) and keep the misc default
            regular = _stream_srate(stream) > 0
            types = [
                t if t is not None else "stim" if d and regular else "misc"
                for t, d in zip(types, discrete)
            ]
            labels_all.extend(labels)
            types_all.extend(types)
            units_all.extend(units)
            discrete_all.extend(discrete)

        # interpolate if gap detection is requested, otherwise resample
        use_interpolation = gap_threshold > 0

        if fs_new is not None:
            data, first_time = _resample_streams(
                streams, channel_ids, fs_new, use_interpolation, np.array(discrete_all)
            )
            fs = fs_new

            if gap_threshold > 0:  # mark gaps if requested
                timestamps = first_time + np.arange(len(data)) / fs
                col_start = 0
                for stream_id in channel_ids:
                    n_chans = int(streams[stream_id]["info"]["channel_count"][0])
                    # irregular streams have no nominal sample spacing, so no gaps
                    if _stream_srate(streams[stream_id]) > 0:
                        _mark_gaps(
                            data,
                            timestamps,
                            streams[stream_id]["time_stamps"],
                            gap_threshold,
                            slice(col_start, col_start + n_chans),
                        )
                    col_start += n_chans
        else:  # only possible if a single stream was selected
            if len(streams[channel_ids[0]]["time_stamps"]) == 0:
                raise ValueError(f"Stream {channel_ids[0]} contains no samples.")
            data = np.array(streams[channel_ids[0]]["time_series"], dtype=float)
            first_time = streams[channel_ids[0]]["time_stamps"][0]
            fs = float(
                np.array(streams[channel_ids[0]]["info"]["effective_srate"]).item()
            )
            if fs == 0:  # fall back to nominal rate (e.g. when only one sample exists)
                fs = float(streams[channel_ids[0]]["info"]["nominal_srate"][0])

        # NaN in stim channels breaks `mne.find_events`, and 0 means "no event" anyway
        stim = np.array(types_all) == "stim"
        data[:, stim] = np.where(np.isnan(data[:, stim]), 0.0, data[:, stim])

        info = mne.create_info(ch_names=labels_all, sfreq=fs, ch_types=types_all)

        parsed = [_parse_unit(unit) for unit in units_all]
        unknown = sorted(
            {
                str(unit).strip()
                for unit, p in zip(units_all, parsed)
                if p is _UNKNOWN_UNIT
            }
        )
        if unknown:
            warnings.warn(
                f"Unrecognized unit(s) {', '.join(map(repr, unknown))}, data of the "
                "corresponding channels are not scaled.",
                RuntimeWarning,
                stacklevel=2,
            )
        known = [None if p is None or p is _UNKNOWN_UNIT else p for p in parsed]
        scale = np.array([1.0 if p is None else p[0] for p in known])
        data = (data * scale).T
        super().__init__(preload=data, info=info, filenames=[fname], **kwargs)

        # data are already in SI, so `unit_mul` stays at 0
        for ch, p in zip(self.info["chs"], known):
            if p is not None and p[1] is not None:
                ch["unit"] = p[1]

        for stream_id in annotation_ids:
            stream = streams[stream_id]
            prefix = f"{stream_id}-" if prefix_markers else ""
            onsets_list, descriptions_list = [], []
            for ts, sub in zip(stream["time_stamps"], stream["time_series"]):
                if _is_string_stream(stream):
                    items = [item for item in sub if item]  # skip empty strings
                else:  # one annotation per sample with all (non-NaN) channel values
                    values = [str(value) for value in sub if not np.isnan(value)]
                    items = [",".join(values)] if values else []
                for item in items:
                    onsets_list.append(ts - first_time)
                    descriptions_list.append(f"{prefix}{item}")
            if onsets_list:
                self.annotations.append(
                    onsets_list, [0] * len(onsets_list), descriptions_list
                )

        recording_datetime = header["info"].get("datetime", [None])[0]
        if recording_datetime is not None:
            try:
                meas_date = datetime.fromisoformat(recording_datetime)
            except ValueError:
                # LabRecorder emits timezone offsets as +HHMM without the colon
                recording_datetime = (
                    recording_datetime[:-2] + ":" + recording_datetime[-2:]
                )
                meas_date = datetime.fromisoformat(recording_datetime)
            self.set_meas_date(meas_date.astimezone(UTC))


def _parse_unit(unit):
    """Parse an XDF channel unit string.

    Parameters
    ----------
    unit : str | None
        Unit string as found in the stream header.

    Returns
    -------
    tuple[float, int | None] | None | object
        `(scale, fiff_unit)` for a recognized unit, where `scale` converts to SI and
        `fiff_unit` is the MNE unit code (or `None` if MNE has no matching unit).
        `None` if the channel has no unit (`None`, empty or `"NA"`), and the
        `_UNKNOWN_UNIT` sentinel if the unit is not recognized.
    """
    if unit is None:
        return None
    unit = str(unit).strip()
    if unit in ("", "NA"):
        return None
    unit = unit.replace("\u03bc", "\u00b5")  # Greek mu -> micro sign
    # symbols are case-sensitive; try the whole string first so that `m` is a metre
    # and not a prefix, and `mm` is a millimetre
    if unit in _UNIT_SYMBOLS:
        return 1.0, _UNIT_SYMBOLS[unit][0]
    if unit[0] in _PREFIXES and unit[1:] in _UNIT_SYMBOLS:
        fiff_unit, prefixable = _UNIT_SYMBOLS[unit[1:]]
        if prefixable:
            return _PREFIXES[unit[0]], fiff_unit
    return _UNIT_WORDS.get(unit.lower(), _UNKNOWN_UNIT)


def _mark_gaps(data, timestamps, original_timestamps, gap_threshold, cols):
    """Mark gaps in data with NaN based on gaps in original timestamps.

    This function modifies the data array in-place.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_samples, n_channels). Modified in-place.
    timestamps : np.ndarray
        Timestamps corresponding to data (interpolated/resampled uniform grid).
    original_timestamps : np.ndarray
        Original timestamps from the stream.
    gap_threshold : float
        Gap threshold in seconds.
    cols : slice
        Column slice indicating which columns belong to this stream.
    """
    # find gaps in original timestamps
    gaps = np.diff(original_timestamps) > gap_threshold
    gap_indices = np.where(gaps)[0]

    if len(gap_indices) == 0:
        return

    # for each gap, find the time range and mark it in the data
    for idx in gap_indices:
        gap_start_time = original_timestamps[idx]
        gap_end_time = original_timestamps[idx + 1]

        # find corresponding indices in the uniform time grid
        start_idx = np.searchsorted(timestamps, gap_start_time, side="right")
        end_idx = np.searchsorted(timestamps, gap_end_time, side="left")

        # mark the gap region as NaN (only for this stream's columns)
        if start_idx < len(data) and end_idx <= len(data):
            data[start_idx:end_idx, cols] = np.nan


def _resample_streams(
    streams, stream_ids, fs_new, use_interpolation=False, discrete=None
):
    """Resample XDF stream(s) to a common sampling rate.

    Parameters
    ----------
    streams : dict
        A dictionary mapping stream IDs to XDF streams.
    stream_ids : list[int]
        The IDs of the desired streams.
    fs_new : float
        Target sampling frequency in Hz.
    use_interpolation : bool
        If True, use linear interpolation. If False, use Fourier-based resampling. This
        only affects continuous channels.
    discrete : np.ndarray | None
        Boolean mask of shape (n_channels,) indicating discrete channels, which are not
        filtered and are resampled by holding the previous value. If `None`, all
        channels are continuous.

    Returns
    -------
    all_time_series : np.ndarray
        Array of shape (n_samples, n_channels) containing raw data. Time intervals where
        a stream has no data contain `np.nan`.
    first_time : float
        Time of the very first sample in seconds.
    """
    from scipy.interpolate import interp1d
    from scipy.signal import butter, sosfiltfilt

    start_times = []
    end_times = []
    n_total_chans = 0
    for stream_id in stream_ids:
        if len(streams[stream_id]["time_stamps"]) == 0:
            raise ValueError(f"Stream {stream_id} contains no samples.")
        start_times.append(streams[stream_id]["time_stamps"][0])
        end_times.append(streams[stream_id]["time_stamps"][-1])
        n_total_chans += int(streams[stream_id]["info"]["channel_count"][0])
    first_time = min(start_times)
    last_time = max(end_times)

    n_samples = int(np.ceil((last_time - first_time) * fs_new))
    all_time_series = np.full((n_samples, n_total_chans), np.nan)
    time_grid = first_time + np.arange(n_samples) / fs_new

    col_start = 0
    for stream_id in stream_ids:
        timestamps = streams[stream_id]["time_stamps"]
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        timestamps, unique_idx = np.unique(timestamps, return_index=True)

        if not sort_indices.shape == unique_idx.shape:
            from warnings import warn

            warn(
                f"Non-unique timestamps found in stream {stream_id}: "
                f"{sort_indices.shape[0]} timestamps, {unique_idx.shape[0]} unique.",
                RuntimeWarning,
            )

        start_time = timestamps[0]
        end_time = timestamps[-1]
        time_series = np.asarray(streams[stream_id]["time_series"], dtype=float)
        x_old = time_series[sort_indices[unique_idx], :]
        col_end = col_start + x_old.shape[1]
        if discrete is None:
            is_discrete = np.zeros(x_old.shape[1], dtype=bool)
        else:
            is_discrete = discrete[col_start:col_end]

        # find valid time range in output grid
        row_start = int(np.floor((start_time - first_time) * fs_new))
        row_end = int(np.ceil((end_time - first_time) * fs_new))
        time_new = time_grid[row_start:row_end]
        x_new = np.full((len(time_new), x_old.shape[1]), np.nan)

        if is_discrete.any():
            x_new[:, is_discrete] = _hold_previous(
                stream_id, timestamps, x_old[:, is_discrete], time_new, fs_new
            )

        if not is_discrete.all():
            x_cont = x_old[:, ~is_discrete]

            # apply anti-aliasing filter if downsampling
            fs_original = float(
                np.array(streams[stream_id]["info"]["effective_srate"]).item()
            )
            if fs_new < fs_original:
                nyquist = fs_new / 2
                sos = butter(
                    8, 0.95 * nyquist, btype="low", fs=fs_original, output="sos"
                )
                x_cont = sosfiltfilt(sos, x_cont, axis=0)

            if use_interpolation:  # linear interpolation
                interpolator = interp1d(
                    timestamps,
                    x_cont,
                    axis=0,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                x_new[:, ~is_discrete] = interpolator(time_new)
            else:  # Fourier-based resampling
                len_new = len(time_new)
                x_new[:, ~is_discrete] = scipy.signal.resample(x_cont, len_new, axis=0)

        all_time_series[row_start:row_end, col_start:col_end] = x_new

        col_start = col_end

    return all_time_series, first_time


def _hold_previous(stream_id, timestamps, x, time_new, fs_new):
    """Resample discrete data by holding the previous value.

    Parameters
    ----------
    stream_id : int
        ID of the stream (only used in the warning message).
    timestamps : np.ndarray
        Sorted unique original timestamps.
    x : np.ndarray
        Original data of shape (n_samples, n_channels).
    time_new : np.ndarray
        Timestamps of the new grid.
    fs_new : float
        Target sampling frequency in Hz (only used in the warning message).

    Returns
    -------
    np.ndarray
        Resampled data of shape (len(time_new), n_channels). Samples before the first
        original timestamp are NaN.
    """
    idx = np.searchsorted(timestamps, time_new, side="right") - 1
    x_new = x[np.clip(idx, 0, None)]
    x_new[idx < 0] = np.nan

    # only count changes in original samples that are covered by the new grid
    n_covered = idx[-1] + 1 if len(idx) else 0
    lost = _count_changes(x[:n_covered]) - _count_changes(x_new)
    if lost > 0:
        warnings.warn(
            f"Resampling stream {stream_id} to {fs_new} Hz lost {lost} value change(s) "
            "in discrete channels (events shorter than one sample at the new sampling "
            "frequency).",
            RuntimeWarning,
            stacklevel=4,
        )
    return x_new


def _count_changes(x):
    """Count value changes along the first axis, ignoring changes from or to NaN."""
    diff = np.diff(x, axis=0)
    return np.count_nonzero(diff[~np.isnan(diff)])


def read_raw_xdf(
    fname,
    streams=None,
    *,
    fs_new=None,
    gap_threshold=0.0,
    prefix_markers=False,
    stream_ids=None,
    marker_ids=None,
    **kwargs,
):
    """Read XDF file.

    Parameters
    ----------
    fname : str
        File name to load.
    streams : int | list[int] | dict[int, str] | None
        Streams to load (all other streams are ignored). A list (or a single stream ID)
        loads each stream in its default mode. A dict maps stream IDs to modes, which
        can be "continuous" ("c"), "discrete" ("d"), or "annotations" ("a"). If `None`,
        all streams containing samples are loaded in their default modes. See `RawXDF`
        for details on how streams are loaded.
    fs_new : float | None
        Target sampling frequency in Hz (required when loading multiple streams as
        channels or a stream with a nominal sampling frequency of 0 Hz). If only one
        stream is loaded as channels, this can be `None`, in which case the stream's
        original sampling rate is used.
    gap_threshold : float
        Detect gaps in timestamps larger than this value (in seconds) and mark those
        samples as NaN. Set to 0.0 to disable gap detection. If `gap_threshold > 0`,
        linear interpolation is used instead of resampling, and `fs_new` must be
        specified.
    prefix_markers : bool
        Whether to prefix annotations with the ID of their stream.
    stream_ids : int | list[int] | None
        Deprecated, use `streams` instead. IDs of streams to load as channels.
    marker_ids : list[int] | None
        Deprecated, use `streams` instead. IDs of string streams and numeric streams
        with 0 Hz to convert to annotations (`None` converts all such streams that are
        not listed in `stream_ids`).

    Returns
    -------
    RawXDF
        The raw data.
    """
    return RawXDF(
        fname,
        streams,
        fs_new=fs_new,
        gap_threshold=gap_threshold,
        prefix_markers=prefix_markers,
        stream_ids=stream_ids,
        marker_ids=marker_ids,
    )


def stream_modes(stream):
    """Return the modes in which a stream can be loaded.

    Parameters
    ----------
    stream : dict
        Stream information as returned by `resolve_streams(fname)`.

    Returns
    -------
    list[str]
        The possible modes ("continuous", "discrete", or "annotations"), starting with
        the default mode that is used if the stream is passed to `streams` without a
        mode. String streams can only be converted to annotations. Numeric streams with
        a nominal sampling frequency of 0 Hz are converted to annotations by default,
        but can also be loaded as discrete channels. Numeric streams with a regular
        sampling frequency are loaded as continuous or discrete channels, depending on
        their stream type (discrete for "Marker(s)", "Event(s)", "Trigger(s)", and
        "Stim").
    """
    if stream["channel_format"] == "string":
        return ["annotations"]
    if float(stream["nominal_srate"]) == 0:
        return ["annotations", "discrete"]
    if str(stream["type"] or "").strip().lower() in _EVENT_STREAM_TYPES:
        return ["discrete", "continuous"]
    return ["continuous", "discrete"]


def _describe_streams(infos):
    """Return one line per stream with its ID, name, and possible modes."""
    return "\n".join(
        f"  {i}: {info['name']} ({', '.join(stream_modes(info))})"
        for i, info in infos.items()
    )


def _parse_streams(streams, infos):
    """Return a dict that maps each stream ID in `streams` to its mode.

    Parameters
    ----------
    streams : int | list[int] | dict[int, str | None]
        The `streams` argument (`None` as a mode selects the default mode).
    infos : dict[int, dict]
        Stream information for all streams in the file (keys are stream IDs).

    Returns
    -------
    dict[int, str]
        The full mode name for each stream.
    """
    if isinstance(streams, int):
        streams = [streams]
    if not isinstance(streams, dict):
        streams = dict.fromkeys(streams)
    unknown_ids = [i for i in streams if i not in infos]
    if unknown_ids:
        raise ValueError(
            f"Stream(s) {', '.join(map(str, unknown_ids))} not found (available stream "
            f"IDs: {', '.join(map(str, infos))})."
        )
    modes = {}
    for stream_id, mode in streams.items():
        possible = stream_modes(infos[stream_id])
        if mode is None:
            modes[stream_id] = possible[0]
            continue
        full = _MODES.get(str(mode).lower())
        if full is None:
            raise ValueError(
                f"Invalid mode {mode!r} for stream {stream_id} (use 'continuous', "
                "'discrete', 'annotations', or their abbreviations 'c', 'd', 'a')."
            )
        if full not in possible:
            raise ValueError(
                f"Stream {stream_id} cannot be loaded as {full} (possible modes: "
                f"{', '.join(possible)})."
            )
        modes[stream_id] = full
    return modes


def _streams_from_ids(stream_ids, marker_ids, infos):
    """Convert the deprecated `stream_ids` and `marker_ids` to `streams`."""
    if isinstance(stream_ids, int):
        stream_ids = [stream_ids]
    stream_ids = list(stream_ids or [])
    both_ids = sorted(set(stream_ids) & set(marker_ids or []))
    if both_ids:
        raise ValueError(
            f"Stream(s) {', '.join(map(str, both_ids))} must not be listed in both "
            "`stream_ids` and `marker_ids`."
        )
    streams = {}
    for stream_id in stream_ids:
        # irregular numeric streams can only be loaded as discrete channels
        possible = stream_modes(infos[stream_id]) if stream_id in infos else [None]
        streams[stream_id] = "discrete" if possible[0] == "annotations" else None
    if marker_ids is None:
        marker_ids = [
            i
            for i, info in infos.items()
            if i not in stream_ids and stream_modes(info)[0] == "annotations"
        ]
    streams.update(dict.fromkeys(marker_ids, "annotations"))
    return streams


def _is_string_stream(stream):
    return stream["info"]["channel_format"][0] == "string"


def _stream_info(stream):
    """Return the information of a stream from `load_xdf()` like `resolve_streams()`."""
    info = stream["info"]
    return {
        "stream_id": info["stream_id"],
        "name": (info.get("name") or [None])[0],
        "type": (info.get("type") or [None])[0],
        "channel_format": info["channel_format"][0],
        "nominal_srate": float(info["nominal_srate"][0]),
    }


def _stream_srate(stream):
    return float(stream["info"]["nominal_srate"][0])


def get_xml(fname):
    """Get XML stream headers and footers from all streams.

    Parameters
    ----------
    fname : str
        Name of the XDF file.

    Returns
    -------
    xml : dict
        XML stream headers and footers.
    """
    with open_xdf(fname) as f:
        xml = defaultdict(dict)
        while True:
            try:
                nbytes = _read_varlen_int(f)
            except EOFError:
                return xml
            tag = struct.unpack("<H", f.read(2))[0]
            if tag in [2, 3, 4, 6]:
                stream_id = struct.unpack("<I", f.read(4))[0]
                if tag in [2, 6]:  # parse StreamHeader/StreamFooter chunk
                    string = f.read(nbytes - 6).decode()
                    xml[stream_id][tag] = ETree.fromstring(string)
                else:  # skip remaining chunk contents
                    f.seek(nbytes - 6, 1)
            else:
                f.seek(nbytes - 2, 1)  # skip remaining chunk contents


def list_chunks(fname):
    """List all chunks contained in an XDF file.

    Listing chunks summarizes the content of the XDF file. Because this function does
    not attempt to parse the data, this also works for corrupted files.

    Parameters
    ----------
    fname : str
        Name of the XDF file.

    Returns
    -------
    chunks : list
        List of dicts containing a short summary for each chunk.
    """
    with open_xdf(fname) as f:
        chunks = []
        while True:
            try:
                nbytes = _read_varlen_int(f)
            except EOFError:
                return chunks
            chunk = {"nbytes": nbytes}
            tag = struct.unpack("<H", f.read(2))[0]
            chunk["tag"] = tag
            if tag == 1:
                chunk["content"] = f.read(nbytes - 2).decode()
            elif tag == 5:
                chunk["content"] = (
                    "0x43 0xA5 0x46 0xDC 0xCB 0xF5 0x41 0x0F "
                    "0xB3 0x0E 0xD5 0x46 0x73 0x83 0xCB 0xE4"
                )
                f.seek(chunk["nbytes"] - 2, 1)  # skip remaining chunk contents
            elif tag in [2, 6]:  # XML
                chunk["stream_id"] = struct.unpack("<I", f.read(4))[0]
                chunk["content"] = (
                    f.read(chunk["nbytes"] - 6).decode().replace("\t", "  ")
                )
            elif tag == 4:
                chunk["stream_id"] = struct.unpack("<I", f.read(4))[0]
                collection_time = struct.unpack("<d", f.read(8))[0]
                offset_value = struct.unpack("<d", f.read(8))[0]
                chunk["content"] = (
                    f"Collection time: {collection_time}\nOffset value: {offset_value}"
                )
            elif tag == 3:
                chunk["stream_id"] = struct.unpack("<I", f.read(4))[0]
                remainder = chunk["nbytes"] - 6
                chunk["content"] = f"<BINARY DATA ({remainder} Bytes)>"
                f.seek(remainder, 1)  # skip remaining chunk contents
            else:
                f.seek(chunk["nbytes"] - 2, 1)  # skip remaining chunk contents
            chunks.append(chunk)
