# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Tests for mnextend.io.xdf."""

import mne
import numpy as np
import pytest
from mne.io.constants import FIFF

import mnextend.io.xdf
from mnextend.io.xdf import (
    _UNKNOWN_UNIT,
    RawXDF,
    _discrete_channels,
    _parse_unit,
    is_discrete_stream,
    is_marker_stream,
)


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("V", (1.0, FIFF.FIFF_UNIT_V)),
        ("mV", (1e-3, FIFF.FIFF_UNIT_V)),
        ("uV", (1e-6, FIFF.FIFF_UNIT_V)),
        ("µV", (1e-6, FIFF.FIFF_UNIT_V)),  # micro sign
        ("μV", (1e-6, FIFF.FIFF_UNIT_V)),  # Greek small mu
        ("nV", (1e-9, FIFF.FIFF_UNIT_V)),
        ("fT", (1e-15, FIFF.FIFF_UNIT_T)),
        ("nT", (1e-9, FIFF.FIFF_UNIT_T)),
        ("µm", (1e-6, FIFF.FIFF_UNIT_M)),
        ("kHz", (1e3, FIFF.FIFF_UNIT_HZ)),
        ("mK", (1e-3, FIFF.FIFF_UNIT_K)),
        ("mrad", (1e-3, FIFF.FIFF_UNIT_RAD)),
        ("mm", (1e-3, FIFF.FIFF_UNIT_M)),
        ("cm", (1e-2, FIFF.FIFF_UNIT_M)),
        ("rad", (1.0, FIFF.FIFF_UNIT_RAD)),
        ("μS", (1e-6, FIFF.FIFF_UNIT_S)),
        ("°C", (1.0, FIFF.FIFF_UNIT_CEL)),
        ("K", (1.0, FIFF.FIFF_UNIT_K)),
        ("ms", (1e-3, FIFF.FIFF_UNIT_SEC)),
        ("Hz", (1.0, FIFF.FIFF_UNIT_HZ)),
        ("px", (1.0, FIFF.FIFF_UNIT_PX)),
        ("microvolts", (1e-6, FIFF.FIFF_UNIT_V)),
        ("Microvolts", (1e-6, FIFF.FIFF_UNIT_V)),
        ("MILLIMETRES", (1e-3, FIFF.FIFF_UNIT_M)),
        ("Degrees Celsius", (1.0, FIFF.FIFF_UNIT_CEL)),
        ("Hertz", (1.0, FIFF.FIFF_UNIT_HZ)),
        (" mV ", (1e-3, FIFF.FIFF_UNIT_V)),  # surrounding whitespace
        ("deg", (1.0, None)),
        ("°", (1.0, None)),
        ("Degrees", (1.0, None)),
        ("NA", None),
        ("", None),
        ("  ", None),
        (None, None),
        ("MV", _UNKNOWN_UNIT),  # symbols are case-sensitive, no mega prefix
        ("kpx", _UNKNOWN_UNIT),  # pixels and degrees do not accept prefixes
        ("mdeg", _UNKNOWN_UNIT),
        ("kk", _UNKNOWN_UNIT),
        ("Vm", _UNKNOWN_UNIT),
        ("uv", _UNKNOWN_UNIT),
        ("foo", _UNKNOWN_UNIT),
    ],
)
def test_parse_unit(unit, expected):
    """Test parsing of unit strings."""
    assert _parse_unit(unit) == expected


def _make_stream(
    units,
    data,
    stream_id=1,
    srate=100,
    stream_type=None,
    types=None,
    time_stamps=None,
):
    """Create a synthetic numeric XDF stream with one channel per unit."""
    n_samples, n_chans = data.shape
    if types is None:
        types = ["misc"] * n_chans
    channels = [
        {
            "label": [f"s{stream_id}ch{i}"],
            "type": [ch_type] if ch_type is not None else [],
            "unit": [unit] if unit is not None else [],
        }
        for i, (unit, ch_type) in enumerate(zip(units, types))
    ]
    info = {
        "stream_id": stream_id,
        "name": ["test"],
        "type": [stream_type],
        "channel_count": [str(n_chans)],
        "channel_format": [data.dtype.name],
        "nominal_srate": [str(srate)],
        "effective_srate": float(srate),
        "desc": [{"channels": [{"channel": channels}]}],
    }
    if time_stamps is None:
        time_stamps = np.arange(n_samples) / (srate or 1)
    return {"info": info, "time_series": data, "time_stamps": time_stamps}


def _make_string_stream(stream_id, markers, time_stamps, srate=0):
    """Create a synthetic string XDF stream."""
    info = {
        "stream_id": stream_id,
        "name": ["markers"],
        "type": ["Markers"],
        "channel_count": ["1"],
        "channel_format": ["string"],
        "nominal_srate": [str(srate)],
        "effective_srate": float(srate),
        "desc": [None],
    }
    return {
        "info": info,
        "time_series": [[marker] for marker in markers],
        "time_stamps": np.asarray(time_stamps, dtype=float),
    }


def _read_streams(monkeypatch, tmp_path, streams, stream_ids, **kwargs):
    fname = tmp_path / "test.xdf"
    fname.touch()  # MNE checks that the file exists
    monkeypatch.setattr(
        mnextend.io.xdf, "load_xdf", lambda fname: (streams, {"info": {}})
    )
    return RawXDF(fname, stream_ids, **kwargs)


def _read(monkeypatch, tmp_path, units, data):
    return _read_streams(monkeypatch, tmp_path, [_make_stream(units, data)], [1])


def _trigger_data(n_samples, pulses, dtype=np.int32):
    """Create a single-channel trigger signal from `{start: (length, code)}`."""
    data = np.zeros((n_samples, 1), dtype=dtype)
    for start, (length, code) in pulses.items():
        data[start : start + length] = code
    return data


def test_units_scaling(monkeypatch, tmp_path):
    """Test unit conversion, unit codes, and warning for unknown units."""
    units = ["uV", "mV", "mm", "deg", "NA", "foo"]
    data = np.ones((10, len(units)))
    with pytest.warns(RuntimeWarning, match="foo") as record:
        raw = _read(monkeypatch, tmp_path, units, data)
    assert len(record) == 1

    expected = [1e-6, 1e-3, 1e-3, 1, 1, 1]
    np.testing.assert_allclose(raw.get_data(), np.tile(expected, (10, 1)).T)

    default = FIFF.FIFF_UNIT_NONE  # default for misc channels
    codes = [ch["unit"] for ch in raw.info["chs"]]
    assert codes == [
        FIFF.FIFF_UNIT_V,
        FIFF.FIFF_UNIT_V,
        FIFF.FIFF_UNIT_M,
        default,
        default,
        default,
    ]
    assert all(ch["unit_mul"] == 0 for ch in raw.info["chs"])


def test_units_known_only(monkeypatch, tmp_path):
    """Test that known units (and missing units) do not warn."""
    units = ["Microvolts", "°C", "NA", "", None]
    data = np.ones((10, len(units)))
    raw = _read(monkeypatch, tmp_path, units, data)  # warnings are errors in CI
    np.testing.assert_allclose(raw.get_data()[0], 1e-6)
    np.testing.assert_allclose(raw.get_data()[1:], 1)
    assert raw.info["chs"][1]["unit"] == FIFF.FIFF_UNIT_CEL


def test_units_warning_deduplicated(monkeypatch, tmp_path):
    """Test that unknown units are reported once, sorted and de-duplicated."""
    units = ["foo", "bar", "foo"]
    with pytest.warns(RuntimeWarning, match="'bar', 'foo'") as record:
        _read(monkeypatch, tmp_path, units, np.ones((10, 3)))
    assert len(record) == 1


@pytest.mark.parametrize(
    "dtype, srate, stream_type, types, discrete_ids, expected",
    [
        (np.int16, 100, "EEG", ["eeg"], None, [False]),  # raw ADC counts
        (np.int32, 100, "Trigger", [None], None, [True]),
        (np.float32, 100, " Markers ", [None], None, [True]),
        (np.float32, 100, "EEG", ["eeg", "stim"], None, [False, True]),
        (np.int32, 100, "TTL", [None], [1], [True]),
        (np.int32, 100, "TTL", [None], [2], [False]),
        (np.float32, 100, "Trigger", ["eeg"], [], [False]),
        (np.float32, 100, "Trigger", ["stim"], [], [True]),  # stim is always discrete
        (np.float32, 100, "EEG", ["eeg", "stim"], [2], [False, True]),
        (np.float32, 0, "HeartRate", [None], [], [True]),
    ],
)
def test_discrete_channels(dtype, srate, stream_type, types, discrete_ids, expected):
    """Test detection of discrete channels."""
    data = np.zeros((10, len(types)), dtype=dtype)
    stream = _make_stream(
        ["NA"] * len(types), data, srate=srate, stream_type=stream_type, types=types
    )
    assert _discrete_channels(stream, types, discrete_ids) == expected


@pytest.mark.parametrize(
    "channel_format, srate, stream_type, marker, discrete",
    [
        ("float32", 100.0, "EEG", False, False),
        ("int16", 100.0, None, False, False),  # raw ADC counts
        ("int32", 1000.0, " Trigger ", False, True),
        ("float32", 100.0, "markers", False, True),
        ("float32", 0.0, "HeartRate", True, True),
        ("string", 0.0, "Markers", True, False),
        ("string", 5000.0, "sampledMarkers", True, False),
    ],
)
def test_stream_classification(channel_format, srate, stream_type, marker, discrete):
    """Test classification of streams from `resolve_streams()` information."""
    stream = {
        "stream_id": 1,
        "type": stream_type,
        "channel_count": 1,
        "channel_format": channel_format,
        "nominal_srate": srate,
    }
    assert is_marker_stream(stream) == marker
    assert is_discrete_stream(stream) == discrete


def test_int_trigger_stream(monkeypatch, tmp_path):
    """Test that an integer trigger stream is loaded unchanged as a stim channel."""
    data = _trigger_data(200, {50: (3, 7), 120: (3, 255)})
    stream = _make_stream(["NA"], data, stream_type="Trigger", types=[None])
    raw = _read_streams(monkeypatch, tmp_path, [stream], [1])
    assert raw.get_channel_types() == ["stim"]
    np.testing.assert_array_equal(raw.get_data()[0], data[:, 0])
    events = mne.find_events(raw, verbose=False)
    np.testing.assert_array_equal(events[:, [0, 2]], [[50, 7], [120, 255]])


@pytest.mark.parametrize("fs_new", [50, 200])
def test_int_trigger_stream_resampling(monkeypatch, tmp_path, fs_new):
    """Test that resampling a trigger stream only produces original codes."""
    data = _trigger_data(200, {50: (4, 7), 120: (4, 255)})
    stream = _make_stream(["NA"], data, stream_type="Trigger", types=[None])
    raw = _read_streams(monkeypatch, tmp_path, [stream], [1], fs_new=fs_new)
    assert set(np.unique(raw.get_data())) == {0, 7, 255}
    events = mne.find_events(raw, verbose=False)
    np.testing.assert_allclose(events[:, 0] / fs_new, [0.5, 1.2])
    np.testing.assert_array_equal(events[:, 2], [7, 255])


def test_int_trigger_stream_lost_events(monkeypatch, tmp_path):
    """Test that losing short events when downsampling warns."""
    data = _trigger_data(200, {51: (1, 7), 120: (4, 255)})
    stream = _make_stream(["NA"], data, stream_type="Trigger", types=[None])
    with pytest.warns(RuntimeWarning, match="lost 2 value change"):
        raw = _read_streams(monkeypatch, tmp_path, [stream], [1], fs_new=50)
    assert set(np.unique(raw.get_data())) == {0, 255}


def test_stim_channel_padding(monkeypatch, tmp_path):
    """Test that stim channels contain 0 instead of NaN outside data and in gaps."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    data = _trigger_data(100, {10: (3, 7), 70: (3, 255)})
    time_stamps = 0.5 + np.arange(100) / 100
    keep = (time_stamps < 0.8) | (time_stamps >= 1.0)  # gap from 0.8 to 1.0 s
    trigger = _make_stream(
        ["NA"],
        data[keep],
        stream_id=2,
        stream_type="Trigger",
        types=[None],
        time_stamps=time_stamps[keep],
    )
    raw = _read_streams(
        monkeypatch, tmp_path, [eeg, trigger], [1, 2], fs_new=100, gap_threshold=0.05
    )
    assert raw.get_channel_types() == ["eeg", "stim"]
    stim = raw.get_data(picks="stim")[0]
    assert not np.isnan(stim).any()
    np.testing.assert_array_equal(stim[:50], 0)  # before the trigger stream starts
    np.testing.assert_array_equal(stim[80:100], 0)  # gap
    np.testing.assert_array_equal(stim[150:], 0)  # after the trigger stream ends
    events = mne.find_events(raw, verbose=False)  # warnings are errors in CI
    np.testing.assert_array_equal(events[:, [0, 2]], [[60, 7], [120, 255]])


def test_numeric_marker_stream_annotations(monkeypatch, tmp_path):
    """Test that numeric streams with 0 Hz are converted to annotations."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    codes = _make_stream(
        ["NA"],
        np.array([[3], [5]], dtype=np.int32),
        stream_id=2,
        srate=0,
        time_stamps=np.array([0.5, 1.2]),
    )
    values = _make_stream(
        ["NA", "NA"],
        np.array([[1.5, np.nan], [np.nan, np.nan]]),
        stream_id=3,
        srate=0,
        time_stamps=np.array([0.7, 0.9]),
    )
    streams = [eeg, codes, values]
    raw = _read_streams(monkeypatch, tmp_path, streams, [1], prefix_markers=True)
    np.testing.assert_allclose(raw.annotations.onset, [0.5, 0.7, 1.2])
    assert list(raw.annotations.description) == ["2-3", "3-1.5", "2-5"]

    raw = _read_streams(monkeypatch, tmp_path, streams, [1], marker_ids=[3])
    assert list(raw.annotations.description) == ["1.5"]


def test_numeric_marker_stream_as_channel(monkeypatch, tmp_path):
    """Test that numeric streams with 0 Hz can be loaded as held channels."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    heart_rate = _make_stream(
        ["NA"],
        np.array([[60.0], [65.0], [70.0]]),
        stream_id=2,
        srate=0,
        stream_type="HeartRate",
        types=[None],
        time_stamps=np.array([0.5, 0.8, 1.0]),
    )
    raw = _read_streams(monkeypatch, tmp_path, [eeg, heart_rate], [1, 2], fs_new=100)
    assert raw.get_channel_types() == ["eeg", "misc"]
    assert len(raw.annotations) == 0
    data = raw.get_data(picks="misc")[0]
    assert np.isnan(data[:50]).all()
    np.testing.assert_array_equal(data[50:80], 60)
    np.testing.assert_array_equal(data[80:100], 65)

    with pytest.raises(ValueError, match="`fs_new` is required"):
        _read_streams(monkeypatch, tmp_path, [heart_rate], [2])
    with pytest.raises(ValueError, match="both `stream_ids` and `marker_ids`"):
        _read_streams(
            monkeypatch, tmp_path, [eeg, heart_rate], [1, 2], fs_new=100, marker_ids=[2]
        )


def test_string_stream_as_channel(monkeypatch, tmp_path):
    """Test that string streams cannot be loaded as channels."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    markers = _make_string_stream(2, ["start", "", "stop"], [0.5, 0.7, 1.5])
    with pytest.raises(ValueError, match=r"String stream\(s\) 2 cannot be loaded"):
        _read_streams(monkeypatch, tmp_path, [eeg, markers], [1, 2], fs_new=100)

    raw = _read_streams(monkeypatch, tmp_path, [eeg, markers], [1])
    np.testing.assert_allclose(raw.annotations.onset, [0.5, 1.5])
    assert list(raw.annotations.description) == ["start", "stop"]


def test_marker_ids(monkeypatch, tmp_path):
    """Test that `marker_ids` selects all streams converted to annotations."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    markers = _make_string_stream(2, ["start", "stop"], [0.5, 1.5])
    sampled = _make_string_stream(3, ["", "a", "", "b"], [0, 0.2, 0.4, 0.6], srate=5)
    streams = [eeg, markers, sampled]
    raw = _read_streams(monkeypatch, tmp_path, streams, [1])
    assert list(raw.annotations.description) == ["a", "start", "b", "stop"]

    raw = _read_streams(monkeypatch, tmp_path, streams, [1], marker_ids=[2])
    assert list(raw.annotations.description) == ["start", "stop"]

    raw = _read_streams(monkeypatch, tmp_path, streams, [1], marker_ids=[])
    assert len(raw.annotations) == 0

    trigger = _make_stream(["NA"], np.zeros((200, 1)), stream_id=4, types=["stim"])
    with pytest.raises(ValueError, match=r"Stream\(s\) 4 cannot be loaded as anno"):
        _read_streams(monkeypatch, tmp_path, [eeg, trigger], [1], marker_ids=[4])
