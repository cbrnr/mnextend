# © MNEXTEND developers
#
# License: BSD (3-clause)

import pytest

from mnextend.io.readers import split_name_ext, _raw_supported


@pytest.mark.parametrize("ext", _raw_supported.keys())
def test_split_name_ext(ext):
    fname = f"test{ext}"
    assert split_name_ext(fname) == ("test", ext)


def test_split_name_ext_unsupported():
    assert split_name_ext("test.xxx") == ("test.xxx", None)
