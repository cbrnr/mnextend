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
    _parse_streams,
    _parse_unit,
    read_raw_xdf,
    stream_modes,
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


def _read_streams(monkeypatch, tmp_path, xdf_streams, streams=None, **kwargs):
    fname = tmp_path / "test.xdf"
    fname.touch()  # MNE checks that the file exists
    monkeypatch.setattr(
        mnextend.io.xdf, "load_xdf", lambda fname: (xdf_streams, {"info": {}})
    )
    return RawXDF(fname, streams, **kwargs)


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


def _info(channel_format, srate, stream_type):
    """Create stream information like `resolve_streams()`."""
    return {
        "stream_id": 1,
        "name": "test",
        "type": stream_type,
        "channel_count": 1,
        "channel_format": channel_format,
        "nominal_srate": srate,
    }


@pytest.mark.parametrize(
    "channel_format, srate, stream_type, expected",
    [
        ("float32", 100.0, "EEG", ["continuous", "discrete"]),
        ("int16", 100.0, None, ["continuous", "discrete"]),  # raw ADC counts
        ("int32", 1000.0, " Trigger ", ["discrete", "continuous"]),
        ("float32", 100.0, "markers", ["discrete", "continuous"]),
        ("float32", 0.0, "HeartRate", ["annotations", "discrete"]),
        ("string", 0.0, "Markers", ["annotations"]),
        ("string", 5000.0, "sampledMarkers", ["annotations"]),
    ],
)
def test_stream_modes(channel_format, srate, stream_type, expected):
    """Test possible modes of streams from `resolve_streams()` information."""
    assert stream_modes(_info(channel_format, srate, stream_type)) == expected


def test_parse_streams():
    """Test parsing of the `streams` argument."""
    infos = {
        1: _info("float32", 500.0, "EEG"),
        2: _info("int32", 1000.0, "Trigger"),
        3: _info("string", 0.0, "Markers"),
        4: _info("float32", 0.0, "HeartRate"),
    }
    defaults = {1: "continuous", 2: "discrete", 3: "annotations", 4: "annotations"}
    assert _parse_streams([1, 2, 3, 4], infos) == defaults
    assert _parse_streams(2, infos) == {2: "discrete"}
    assert _parse_streams({2: "c", 4: "D", 3: "annotations"}, infos) == {
        2: "continuous",
        4: "discrete",
        3: "annotations",
    }
    with pytest.raises(ValueError, match=r"Stream\(s\) 5 not found.*: 1, 2, 3, 4\)"):
        _parse_streams([1, 5], infos)
    with pytest.raises(ValueError, match="Invalid mode 'x' for stream 1"):
        _parse_streams({1: "x"}, infos)
    with pytest.raises(ValueError, match=r"3 cannot be loaded as continuous \(possi"):
        _parse_streams({3: "c"}, infos)
    with pytest.raises(ValueError, match="4 cannot be loaded as continuous"):
        _parse_streams({4: "continuous"}, infos)


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
    raw = _read_streams(monkeypatch, tmp_path, streams, [1, 2, 3], prefix_markers=True)
    np.testing.assert_allclose(raw.annotations.onset, [0.5, 0.7, 1.2])
    assert list(raw.annotations.description) == ["2-3", "3-1.5", "2-5"]

    raw = _read_streams(monkeypatch, tmp_path, streams, [1, 3])
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
    raw = _read_streams(
        monkeypatch, tmp_path, [eeg, heart_rate], {1: "c", 2: "d"}, fs_new=100
    )
    assert raw.get_channel_types() == ["eeg", "misc"]
    assert len(raw.annotations) == 0
    data = raw.get_data(picks="misc")[0]
    assert np.isnan(data[:50]).all()
    np.testing.assert_array_equal(data[50:80], 60)
    np.testing.assert_array_equal(data[80:100], 65)

    with pytest.raises(ValueError, match="`fs_new` is required"):
        _read_streams(monkeypatch, tmp_path, [heart_rate], {2: "d"})


def test_string_stream_modes(monkeypatch, tmp_path):
    """Test that string streams can only be converted to annotations."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    markers = _make_string_stream(2, ["start", "", "stop"], [0.5, 0.7, 1.5])
    with pytest.raises(ValueError, match="2 cannot be loaded as discrete"):
        _read_streams(monkeypatch, tmp_path, [eeg, markers], {1: "c", 2: "d"})

    raw = _read_streams(monkeypatch, tmp_path, [eeg, markers], [1, 2])
    np.testing.assert_allclose(raw.annotations.onset, [0.5, 1.5])
    assert list(raw.annotations.description) == ["start", "stop"]


def test_only_listed_streams(monkeypatch, tmp_path):
    """Test that only streams listed in `streams` are loaded."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    markers = _make_string_stream(2, ["start", "stop"], [0.5, 1.5])
    sampled = _make_string_stream(3, ["", "a", "", "b"], [0, 0.2, 0.4, 0.6], srate=5)
    streams = [eeg, markers, sampled]
    raw = _read_streams(monkeypatch, tmp_path, streams, [1, 2, 3])
    assert list(raw.annotations.description) == ["a", "start", "b", "stop"]

    raw = _read_streams(monkeypatch, tmp_path, streams, [1, 2])
    assert list(raw.annotations.description) == ["start", "stop"]

    raw = _read_streams(monkeypatch, tmp_path, streams, [1])
    assert len(raw.annotations) == 0

    with pytest.raises(ValueError, match="At least one stream must be loaded as chan"):
        _read_streams(monkeypatch, tmp_path, streams, [2, 3])


@pytest.mark.parametrize(
    "stream_type, mode, ch_type, values",
    [
        ("TTL", "d", "stim", {0, 7, 255}),  # unrecognized type, explicitly discrete
        ("Trigger", None, "stim", {0, 7, 255}),  # recognized type
        ("Trigger", "c", "misc", None),  # analog signal with a misleading type
    ],
)
def test_explicit_modes(monkeypatch, tmp_path, stream_type, mode, ch_type, values):
    """Test that explicit modes override the detection based on the stream type."""
    data = _trigger_data(200, {50: (4, 7), 120: (4, 255)})
    stream = _make_stream(["NA"], data, stream_type=stream_type, types=[None])
    raw = _read_streams(monkeypatch, tmp_path, [stream], {1: mode}, fs_new=50)
    assert raw.get_channel_types() == [ch_type]
    if values is None:  # filtered and resampled, so the codes are not preserved
        assert not set(np.unique(raw.get_data())) <= {0, 7, 255}
    else:
        assert set(np.unique(raw.get_data())) == values


def test_stim_channel_in_continuous_stream(monkeypatch, tmp_path):
    """Test that stim channels are discrete even in continuous streams."""
    data = np.zeros((200, 2))
    data[:, 0] = np.sin(np.arange(200) / 10)
    data[:, 1] = _trigger_data(200, {50: (4, 7), 120: (4, 255)})[:, 0]
    stream = _make_stream(["NA", "NA"], data, types=["eeg", "stim"])
    raw = _read_streams(monkeypatch, tmp_path, [stream], {1: "c"}, fs_new=50)
    assert raw.get_channel_types() == ["eeg", "stim"]
    assert set(np.unique(raw.get_data(picks="stim"))) == {0, 7, 255}


def test_all_streams(monkeypatch, tmp_path):
    """Test that all streams with samples are loaded if `streams` is not specified."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    markers = _make_string_stream(2, ["start", "stop"], [0.5, 1.5])
    empty = _make_stream(["NA"], np.zeros((0, 1)), stream_id=3, types=["eeg"])
    raw = _read_streams(monkeypatch, tmp_path, [eeg, markers, empty])
    assert raw.get_channel_types() == ["eeg"]
    assert list(raw.annotations.description) == ["start", "stop"]
    raw = read_raw_xdf(tmp_path / "test.xdf")  # uses the same patched `load_xdf`
    assert raw.get_channel_types() == ["eeg"]

    with pytest.raises(ValueError, match="Stream 3 contains no samples"):
        _read_streams(monkeypatch, tmp_path, [eeg, markers, empty], [1, 3], fs_new=100)

    data = _trigger_data(200, {50: (3, 7)})
    trigger = _make_stream(
        ["NA"], data, stream_id=4, stream_type="Trigger", types=[None]
    )
    streams = [eeg, markers, empty, trigger]
    with pytest.raises(ValueError, match=r"All streams.*\n  1: test \(continuous, di"):
        _read_streams(monkeypatch, tmp_path, streams)
    raw = _read_streams(monkeypatch, tmp_path, streams, fs_new=100)
    assert raw.get_channel_types() == ["eeg", "stim"]
    assert list(raw.annotations.description) == ["start", "stop"]


def test_deprecated_ids(monkeypatch, tmp_path):
    """Test the deprecated `stream_ids` and `marker_ids` arguments."""
    eeg = _make_stream(["NA"], np.ones((200, 1)), types=["eeg"])
    heart_rate = _make_stream(
        ["NA"],
        np.array([[60.0], [65.0]]),
        stream_id=2,
        srate=0,
        types=[None],
        time_stamps=np.array([0.5, 1.0]),
    )
    markers = _make_string_stream(3, ["start", "stop"], [0.2, 1.5])
    streams = [eeg, heart_rate, markers]
    with pytest.warns(FutureWarning, match="`stream_ids` and `marker_ids` are depre"):
        raw = _read_streams(monkeypatch, tmp_path, streams, stream_ids=[1])
    assert raw.get_channel_types() == ["eeg"]
    assert list(raw.annotations.description) == ["start", "60.0", "65.0", "stop"]

    with pytest.warns(FutureWarning):
        raw = _read_streams(
            monkeypatch, tmp_path, streams, stream_ids=1, marker_ids=[3]
        )
    assert list(raw.annotations.description) == ["start", "stop"]

    # irregular numeric streams in `stream_ids` are loaded as discrete channels
    with pytest.warns(FutureWarning):
        raw = _read_streams(
            monkeypatch, tmp_path, streams, stream_ids=[1, 2], fs_new=100
        )
    assert raw.get_channel_types() == ["eeg", "misc"]
    assert list(raw.annotations.description) == ["start", "stop"]

    with pytest.raises(ValueError, match="both `stream_ids` and `marker_ids`"):
        _read_streams(monkeypatch, tmp_path, streams, stream_ids=[1, 3], marker_ids=[3])
    with pytest.raises(ValueError, match="cannot be combined with `streams`"):
        _read_streams(monkeypatch, tmp_path, streams, [1], stream_ids=[1])
