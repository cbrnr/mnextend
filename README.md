# MNExtend

This package provides additional functionality for working with [MNE-Python](https://mne.tools/), the most popular Python package for processing electrophysiological data (EEG, MEG, ...).

## Features

### Reading additional file formats

MNExtend provides readers for the following file formats that are not natively supported by MNE-Python:

- [XDF](https://github.com/sccn/xdf/wiki/Specifications) (Extensible Data Format)
- [MAT](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html) (MATLAB)
- [NPY](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) (NumPy)

In addition, MNExtend adds the following readers from third-party packages:

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

### ICLabel classification

MNExtend includes [ICLabel](https://labeling.ucsd.edu/tutorial/overview), a pre-trained classifier that labels ICA components as one of seven types: brain, muscle, eye, heart, line noise, channel noise, or other. In contrast to [MNE-ICALabel](https://mne.tools/mne-icalabel/stable/index.html), the classifier is implemented in pure NumPy and does not depend on [ONNX Runtime](https://onnxruntime.ai).

`run_iclabel()` takes a fitted `ICA` object and the corresponding `Raw` or `Epochs` instance (which must have a montage set), and returns an array of class probabilities:

```python
from mnextend import plot_ica_components, run_iclabel

probs = run_iclabel(raw, ica)
figs = plot_ica_components(raw, ica, probs)
```
