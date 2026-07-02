# © MNEXTEND developers
#
# License: BSD (3-clause)

from importlib.metadata import version

from mnextend.iclabel import IC_LABELS, plot_ica_components, run_iclabel
from mnextend.io.readers import read_epochs, read_raw
from mnextend.io.utils import split_name_ext
from mnextend.io.writers import write_epochs, write_raw

__version__ = version("mnextend")

__all__ = [
    "IC_LABELS",
    "plot_ica_components",
    "read_epochs",
    "read_raw",
    "run_iclabel",
    "split_name_ext",
    "write_epochs",
    "write_raw",
]
