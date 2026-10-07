# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Tests for mnextend.io.xdf."""

import numpy as np
import pytest
from mne.io.constants import FIFF

import mnextend.io.xdf
from mnextend.io.xdf import _UNKNOWN_UNIT, RawXDF, _parse_unit


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


def _make_stream(units, data):
    """Create a synthetic numeric XDF stream with one channel per unit."""
    n_samples, n_chans = data.shape
    channels = [
        {
            "label": [f"ch{i}"],
            "type": ["misc"],
            "unit": [unit] if unit is not None else [],
        }
        for i, unit in enumerate(units)
    ]
    info = {
        "stream_id": 1,
        "name": ["test"],
        "channel_count": [str(n_chans)],
        "channel_format": ["float32"],
        "nominal_srate": ["100"],
        "effective_srate": 100.0,
        "desc": [{"channels": [{"channel": channels}]}],
    }
    return {
        "info": info,
        "time_series": data,
        "time_stamps": np.arange(n_samples) / 100,
    }


def _read(monkeypatch, tmp_path, units, data):
    fname = tmp_path / "test.xdf"
    fname.touch()  # MNE checks that the file exists
    stream = _make_stream(units, data)
    monkeypatch.setattr(
        mnextend.io.xdf, "load_xdf", lambda fname: ([stream], {"info": {}})
    )
    return RawXDF(fname, [1])


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
