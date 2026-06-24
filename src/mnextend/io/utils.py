# © MNEXTEND developers
#
# License: BSD (3-clause)

from pathlib import Path


def split_name_ext(fname, readers):
    """Return name and supported file extension."""
    maxsuffixes = max(ext.count(".") for ext in readers)
    suffixes = Path(fname).suffixes
    for n in range(maxsuffixes, 0, -1):
        ext = "".join(suffixes[-n:]).lower()
        if ext in readers:
            return Path(fname).name[: -len(ext)], ext
    return Path(fname).name, None
