# MNEXTEND

This package provides additional functionality for working with [MNE-Python](https://mne.tools/), the most popular Python package for processing electrophysiological data (EEG, MEG, ...).

## Features

### Reading additional file formats

MNEXTEND provides readers for the following file formats that are not natively supported by MNE-Python:

- [XDF](https://github.com/sccn/xdf/wiki/Specifications) (Extensible Data Format) – see below for more details
- [MAT](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html) (MATLAB)
- [NPY](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) (NumPy)

In addition, MNEXTEND adds the following readers from third-party packages:

- [BVRF](https://www.brainproducts.com/support-resources/brainvision-recording-format/) (via [PyBVRF](https://github.com/cbrnr/pybvrf))

Together with the native MNE-Python readers, `read_raw()` and `read_epochs()` provide a unified interface for reading electrophysiological data from a wide range of file formats, so all you have to do is:

```python
from mnextend import read_raw, read_epochs

raw = read_raw("my_data.xdf", streams=[1, 2], fs_new=500)
epochs = read_epochs("my_data-epochs.fif.gz")
```

### Inspecting files before reading

Some formats require inspecting file contents before calling `read_raw()`, e.g. to let a user pick which stream(s) or participant(s) to load:

```python
from mnextend.io.bvrf import read_bvrf_header
from mnextend.io.xdf import resolve_streams

streams = resolve_streams("my_data.xdf")
header = read_bvrf_header("my_data.bvrh")
```

### Reading XDF files

XDF files usually contain several streams with different sampling rates and data types, for example EEG from an amplifier, trigger codes from a trigger box, and text markers from the experiment software. The `streams` argument in the example above selects the streams to load (all other streams are ignored), and `fs_new` resamples them to a common sampling frequency.

Let's consider a file with the following streams (as listed by `resolve_streams()`):

| ID | Type | Format | Sampling rate | Content | Possible modes (default first) |
|---|---|---|---|---|---|
| 1 | EEG | float32 | 500 Hz | EEG signals | continuous, discrete |
| 2 | Trigger | int32 | 1000 Hz | trigger codes (0, 7, 255) | discrete, continuous |
| 3 | Markers | string | 0 Hz | text markers ("start", "stop") | annotations |
| 4 | Markers | int32 | 0 Hz | event codes | annotations, discrete |
| 5 | HeartRate | float32 | 0 Hz | heart rate updates | annotations, discrete |
| 6 | TTL | int16 | 1000 Hz | TTL pulses (0, 1) | continuous, discrete |

Each stream is loaded in one of three modes:

- **Continuous** (`"continuous"` or `"c"`): channels with signals such as EEG, which are filtered and resampled like any signal.
- **Discrete** (`"discrete"` or `"d"`): channels with values such as trigger codes, which must not be changed. They are resampled without filtering by holding the previous value, so that they only contain values from the original data (a warning is issued if short pulses are lost due to downsampling). Discrete channels of streams with a regular sampling frequency get the channel type "stim" (unless the stream specifies a different type), so that `mne.find_events()` works out of the box.
- **Annotations** (`"annotations"` or `"a"`): one annotation per non-empty string (string streams) or per sample with the value as its description (numeric streams).

If `streams` is not specified, all streams containing samples are loaded in their default modes. If `streams` is a list of stream IDs, only these streams are loaded in their default modes. A dict maps stream IDs to modes instead. Channels are always ordered by stream ID. Which modes are possible depends on the stream:

- String streams (such as stream 3) can only be converted to annotations.
- Numeric streams with a nominal sampling frequency of 0 Hz (streams 4 and 5) are converted to annotations by default. They can also be loaded as discrete channels, which makes sense for irregular measurements like heart rate (each value is held until the next one arrives).
- Numeric streams with a regular sampling frequency (streams 1, 2, and 6) are loaded as continuous or discrete channels. The default is discrete if the stream type is "Marker(s)", "Event(s)", "Trigger(s)", or "Stim" (case-insensitive), such as stream 2, and continuous otherwise.

The stream type is just a name chosen by the software that created the stream, so the default can be wrong in both directions. Stream 6 contains discrete pulses, but its type "TTL" is not recognized. Conversely, an analog photodiode signal could be streamed with the type "Trigger". In such cases, specify the mode explicitly. The data type plays no role, because many programs stream trigger codes as floats, and amplifiers often stream continuous signals as integers. Channels with the channel type "stim" (for example, a trigger channel in an EEG stream) are always discrete, even in continuous streams.

Here are some examples for the file above:

```python
from mnextend import read_raw

# all streams in their default modes (`fs_new` is required for multiple channel streams)
# (1, 6: continuous; 2: discrete; 3, 4, 5: annotations)
raw = read_raw("my_data.xdf", fs_new=500)

# all streams except the TTL pulses in their default modes
# (1: continuous; 2: discrete; 3, 4, 5: annotations; 6: not loaded)
raw = read_raw("my_data.xdf", streams=[1, 2, 3, 4, 5], fs_new=500)

# EEG, triggers, and only the text markers
# (1: continuous; 2: discrete; 3: annotations; 4, 5, 6: not loaded)
raw = read_raw("my_data.xdf", streams=[1, 2, 3], fs_new=500)

# heart rate as a channel instead of annotations
# (1: continuous; 5: discrete; 3, 4: annotations; 2, 6: not loaded)
raw = read_raw("my_data.xdf", streams={1: "c", 3: "a", 4: "a", 5: "d"}, fs_new=500)

# TTL pulses as a discrete channel
# (1: continuous; 2, 6: discrete; 3: annotations; 4, 5: not loaded)
raw = read_raw("my_data.xdf", streams={1: "c", 2: "d", 3: "a", 6: "d"}, fs_new=500)
```

If `streams` is not specified and several streams would be loaded as channels without `fs_new`, the error message lists all streams with their possible modes. To get this information programmatically, use `stream_modes()`:

```python
from mnextend.io.xdf import resolve_streams, stream_modes

for stream in resolve_streams("my_data.xdf"):
    print(stream["stream_id"], stream["name"], stream_modes(stream))
```

### Writing raw data

Writing raw data is supported via `write_raw()`, which does not implement any new file formats, but provides a unified interface for writing raw data to the file formats that are natively supported by MNE-Python:

```python
from mnextend import write_raw

write_raw("my_data-raw.fif.gz", raw)
```

### ICLabel classification

MNEXTEND includes [ICLabel](https://labeling.ucsd.edu/tutorial/overview), a pre-trained classifier that labels ICA components as one of seven types: brain, muscle, eye, heart, line noise, channel noise, or other. In contrast to [MNE-ICALabel](https://mne.tools/mne-icalabel/stable/index.html), the classifier is implemented in pure NumPy and does not depend on [ONNX Runtime](https://onnxruntime.ai).

`run_iclabel()` takes a fitted `ICA` object and the corresponding `Raw` or `Epochs` instance (which must have a montage set), and returns an array of class probabilities:

```python
from mnextend import plot_ica_components, run_iclabel

probs = run_iclabel(raw, ica)
figs = plot_ica_components(raw, ica, probs)
```

### Adaptive line-noise removal

Inspired by `Raw.notch_filter(method="spectrum_fit")` in MNE-Python, this implementation removes known line frequencies from a preloaded `Raw` object much faster (roughly 50× in our tests). It fits sinusoids in overlapping windows, accommodating gradual amplitude and phase changes without suppressing nearby frequencies.

```python
from mnextend import remove_line_noise

raw.load_data()
remove_line_noise(
    raw,
    line_freq=50,
    picks="eeg",
    include_harmonics=True,
    window_length=10,
    overlap=0.5,
)
```
