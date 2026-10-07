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
from pyxdf import load_xdf, resolve_streams
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


class RawXDF(BaseRaw):
    """Raw data from .xdf file."""

    def __init__(
        self,
        fname,
        stream_ids,
        marker_ids=None,
        prefix_markers=False,
        fs_new=None,
        gap_threshold=0.0,
        *args,
        **kwargs,
    ):
        """Read raw data from .xdf file.

        Parameters
        ----------
        fname : str | Path
            File name to load.
        stream_ids : int | list[int]
            ID(s) of streams to load. Use `pyxdf.resolve_streams(fname)` to list
            available streams.
        marker_ids : list[int] | None
            IDs of marker streams to load. If `None`, load all marker streams. A marker
            stream is a stream with a nominal sampling frequency of 0 Hz.
        prefix_markers : bool
            Whether to prefix marker streams with their corresponding stream ID.
        fs_new : float | None
            Target sampling frequency in Hz (required when reading multiple streams). If
            only one stream is provided, this can be `None`, in which case the stream's
            original sampling rate is used.
        gap_threshold : float
            Detect gaps in timestamps larger than this value (in seconds) and mark those
            samples as NaN. Set to 0.0 to disable gap detection. If `gap_threshold > 0`,
            linear interpolation is used instead of resampling, and `fs_new` must be
            specified.

        Notes
        -----
        Resampling depends on whether gap detection is requested or not:
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
        if len(stream_ids) == 0:
            raise ValueError("Argument `stream_ids` must not be empty.")

        if len(stream_ids) > 1 and fs_new is None:
            raise ValueError(
                "Argument `fs_new` is required when reading multiple streams."
            )

        if gap_threshold < 0:
            raise ValueError(
                f"Argument `gap_threshold` must be non-negative, got {gap_threshold}."
            )

        if gap_threshold > 0 and fs_new is None:
            raise ValueError(
                "Argument `fs_new` is required when `gap_threshold > 0`. "
                "Gap detection requires resampling to a regular time grid."
            )

        streams, header = load_xdf(fname)
        streams = {stream["info"]["stream_id"]: stream for stream in streams}

        if all(_is_stringstream(streams[stream_id]) for stream_id in stream_ids):
            raise RuntimeError(
                "Loading only marker streams is not supported, at least one stream must"
                " be a regular stream."
            )

        labels_all, types_all, units_all = [], [], []
        channel_types = get_channel_type_constants(True)
        for stream_id in stream_ids:
            stream = streams[stream_id]

            n_chans = int(stream["info"]["channel_count"][0])
            labels, types, units = [], [], []
            try:
                for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
                    labels.append(str(ch["label"][0]))
                    if ch["type"] and ch["type"][0].lower() in channel_types:
                        types.append(ch["type"][0].lower())
                    else:
                        types.append("misc")
                    units.append(ch["unit"][0] if ch["unit"] else "NA")
            except (TypeError, IndexError):  # no channel labels found
                pass
            if not labels:
                labels = [f"{stream['info']['name'][0]}_{n}" for n in range(n_chans)]
            if not units:
                units = ["NA" for _ in range(n_chans)]
            if not types:
                types = ["misc" for _ in range(n_chans)]
            labels_all.extend(labels)
            types_all.extend(types)
            units_all.extend(units)

        # interpolate if gap detection is requested, otherwise resample
        use_interpolation = gap_threshold > 0

        if fs_new is not None:
            data, first_time = _resample_streams(
                streams, stream_ids, fs_new, use_interpolation
            )
            fs = fs_new

            if gap_threshold > 0:  # mark gaps if requested
                timestamps = first_time + np.arange(len(data)) / fs
                col_start = 0
                for stream_id in stream_ids:
                    n_chans = int(streams[stream_id]["info"]["channel_count"][0])
                    _mark_gaps(
                        data,
                        timestamps,
                        streams[stream_id]["time_stamps"],
                        gap_threshold,
                        slice(col_start, col_start + n_chans),
                    )
                    col_start += n_chans
        else:  # only possible if a single stream was selected
            if len(streams[stream_ids[0]]["time_stamps"]) == 0:
                raise ValueError(f"Stream {stream_ids[0]} contains no samples.")
            data = streams[stream_ids[0]]["time_series"]
            first_time = streams[stream_ids[0]]["time_stamps"][0]
            fs = float(
                np.array(streams[stream_ids[0]]["info"]["effective_srate"]).item()
            )
            if fs == 0:  # fall back to nominal rate (e.g. when only one sample exists)
                fs = float(streams[stream_ids[0]]["info"]["nominal_srate"][0])

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
        super().__init__(preload=data, info=info, filenames=[fname], *args, **kwargs)

        # data are already in SI, so `unit_mul` stays at 0
        for ch, p in zip(self.info["chs"], known):
            if p is not None and p[1] is not None:
                ch["unit"] = p[1]

        # convert string streams to annotations
        for stream_id, stream in streams.items():
            if not _is_stringstream(stream):
                continue
            srate = float(stream["info"]["nominal_srate"][0])
            # classic marker streams (srate=0) respect the user's marker selection;
            # regular-rate string streams are always converted automatically
            if srate == 0 and marker_ids is not None and stream_id not in marker_ids:
                continue
            prefix = f"{stream_id}-" if prefix_markers else ""
            onsets_list, descriptions_list = [], []
            for ts, sub in zip(stream["time_stamps"], stream["time_series"]):
                for item in sub:
                    if item:  # skip empty strings
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


def _resample_streams(streams, stream_ids, fs_new, use_interpolation=False):
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
        If True, use linear interpolation. If False, use Fourier-based resampling.

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
        x_old = streams[stream_id]["time_series"][sort_indices[unique_idx], :]

        # apply anti-aliasing filter if downsampling
        fs_original = float(
            np.array(streams[stream_id]["info"]["effective_srate"]).item()
        )
        if fs_new < fs_original:
            nyquist = fs_new / 2
            sos = butter(8, 0.95 * nyquist, btype="low", fs=fs_original, output="sos")
            x_old = sosfiltfilt(sos, x_old, axis=0)

        # find valid time range in output grid
        row_start = int(np.floor((start_time - first_time) * fs_new))
        row_end = int(np.ceil((end_time - first_time) * fs_new))
        time_new = time_grid[row_start:row_end]

        if use_interpolation:  # linear interpolation
            interpolator = interp1d(
                timestamps,
                x_old,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            x_new = interpolator(time_new)
        else:  # Fourier-based resampling
            len_new = len(time_new)
            x_new = scipy.signal.resample(x_old, len_new, axis=0)

        col_end = col_start + x_new.shape[1]
        all_time_series[row_start:row_end, col_start:col_end] = x_new

        col_start += x_new.shape[1]

    return all_time_series, first_time


def read_raw_xdf(
    fname,
    stream_ids=None,
    marker_ids=None,
    prefix_markers=False,
    fs_new=None,
    gap_threshold=0.0,
    **kwargs,
):
    """Read XDF file.

    Parameters
    ----------
    fname : str
        File name to load.
    stream_ids : int | list[int] | None
        ID(s) of streams to load. If `None`, raises a `ValueError` listing the available
        regular stream IDs. Use `pyxdf.resolve_streams(fname)` to list available
        streams.
    marker_ids : list[int] | None
        IDs of marker streams to load. If `None`, load all marker streams. A marker
        stream is a stream with a nominal sampling frequency of 0 Hz.
    prefix_markers : bool
        Whether to prefix marker streams with their corresponding stream ID.
    fs_new : float | None
        Target sampling frequency in Hz (required when reading multiple streams). If
        only one stream is provided, this can be `None`, in which case the stream's
        original sampling rate is used.
    gap_threshold : float
        Detect gaps in timestamps larger than this value (in seconds) and mark those
        samples as NaN. Set to 0.0 to disable gap detection. If `gap_threshold > 0`,
        linear interpolation is used instead of resampling, and `fs_new` must be
        specified.

    Returns
    -------
    RawXDF
        The raw data.
    """
    if isinstance(stream_ids, int):
        stream_ids = [stream_ids]
    if stream_ids is None:
        streams = resolve_streams(fname)
        ids = [s["stream_id"] for s in streams if s["channel_format"] != "string"]
        msg = (
            "Argument `stream_ids` is required (available regular stream IDs: "
            f"{', '.join(map(str, ids))})."
        )
        raise ValueError(msg)
    return RawXDF(
        fname,
        stream_ids,
        marker_ids,
        prefix_markers,
        fs_new,
        gap_threshold,
    )


def _is_stringstream(stream):
    return stream["info"]["channel_format"][0] == "string"


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
