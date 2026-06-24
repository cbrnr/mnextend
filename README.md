# MNEXTEND

This package provides additional functionality for working with [MNE-Python](https://mne.tools/), the most popular Python package for processing electrophysiological data (EEG, MEG, ...).

## Features

### Reading additional file formats

MNEXTEND provides readers for the following file formats that are not natively supported by MNE-Python:

- [XDF](https://github.com/sccn/xdf/wiki/Specifications) (Extensible Data Format)
- [MAT](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html) (MATLAB)
- [NPY](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) (NumPy)

In addition, MNEXTEND adds readers from third-party packages to provide a unified interface via `read_raw()` and `read_epochs()`:

- [BVRF](https://www.brainproducts.com/support-resources/brainvision-recording-format/) (via [PyBVRF](https://github.com/cbrnr/pybvrf))

Together with the native MNE-Python readers, `read_raw()` and `read_epochs()` provide a unified interface for reading electrophysiological data from a wide range of file formats, so all you have to do is:

```python
from mnextend import read_raw, read_epochs

raw = read_raw("my_data-raw.xdf", stream_ids=[1, 2, 3])
epochs = read_epochs("my_data-epochs.fif.gz")
```

### Writing raw data

Writing raw data is supported via `write_raw()`, which does not implement any new file formats, but provides a unified interface for writing raw data to the file formats that are natively supported by MNE-Python:

```python
from mnextend import write_raw

write_raw("my_data-raw.fif.gz", raw)
```
