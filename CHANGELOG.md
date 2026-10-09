## [UNRELEASED] · YYYY-MM-DD
### ✨ Added
- Add `streams` parameter to `read_raw_xdf()`, which selects the XDF streams to load (all streams containing samples by default) and the mode for each stream (`"continuous"`, `"discrete"`, or `"annotations"`), and `stream_modes()` (via `mnextend.io.xdf`) to list the possible modes of a stream ([#15](https://github.com/cbrnr/mnextend/pull/15) by [Clemens Brunner](https://github.com/cbrnr))

### 🔧 Fixed
- Fix resampling of XDF trigger streams, which are now loaded as discrete stim channels (resampled without filtering by holding the previous value) instead of being filtered like signals ([#15](https://github.com/cbrnr/mnextend/pull/15) by [Clemens Brunner](https://github.com/cbrnr))

### 🌀 Changed
- Deprecate `stream_ids` and `marker_ids` in `read_raw_xdf()` in favor of `streams` (they will be removed in MNEXTEND 0.6.0) ([#15](https://github.com/cbrnr/mnextend/pull/15) by [Clemens Brunner](https://github.com/cbrnr))
- Convert numeric XDF streams with a nominal sampling frequency of 0 Hz to annotations by default, or load them as discrete channels (instead of continuous channels) ([#15](https://github.com/cbrnr/mnextend/pull/15) by [Clemens Brunner](https://github.com/cbrnr))
- Make all parameters of `read_raw_xdf()` except `fname` and `streams` keyword-only ([#15](https://github.com/cbrnr/mnextend/pull/15) by [Clemens Brunner](https://github.com/cbrnr))
- Raise a `ValueError` for XDF stream IDs that do not exist in the file (instead of a `KeyError`) ([#15](https://github.com/cbrnr/mnextend/pull/15) by [Clemens Brunner](https://github.com/cbrnr))

## [0.4.0] · 2026-10-07
### ✨ Added
- Support more units when reading XDF files ([#13](https://github.com/cbrnr/mnextend/pull/13) by [Clemens Brunner](https://github.com/cbrnr))

### 🔧 Fixed
- Fix exporting EEGLAB `.set` files and ICLabel tests with newer MNE and NumPy versions ([#14](https://github.com/cbrnr/mnextend/pull/14) by [Clemens Brunner](https://github.com/cbrnr))

## [0.3.0] · 2026-09-04
### ✨ Added
- Add fast line-noise removal using fitted sinusoids ([#8](https://github.com/cbrnr/mnextend/pull/8) by [Clemens Brunner](https://github.com/cbrnr))

## [0.2.2] · 2026-07-06
### ✨ Added
- Expose `resolve_streams()` (via `mnextend.io.xdf`) and `read_bvrf_header()` (via `mnextend.io.bvrf`) so downstream packages can inspect XDF/BVRF files before reading them ([#7](https://github.com/cbrnr/mnextend/pull/7) by [Clemens Brunner](https://github.com/cbrnr))

## [0.2.1] · 2026-07-02
### ✨ Added
- Add `__version__` attribute to the package ([#5](https://github.com/cbrnr/mnextend/pull/5) by [Clemens Brunner](https://github.com/cbrnr))

## [0.2.0] · 2026-06-25
### ✨ Added
- Add `run_iclabel()` for automatic ICA component classification using the ICLabel algorithm and `plot_ica_components()` for visualizing the results ([#4](https://github.com/cbrnr/mnextend/pull/4) by [Clemens Brunner](https://github.com/cbrnr))

## [0.1.0] · 2026-06-24
### ✨ Added
- Add readers and writers for additional file formats (XDF, MAT, NPY, BVRF) to provide a unified interface for reading and writing electrophysiological data ([#1](https://github.com/cbrnr/mnextend/pull/1) by [Clemens Brunner](https://github.com/cbrnr))
- Add support for importing and exporting epoch data from/to EEGLAB `.set` files ([#2](https://github.com/cbrnr/mnextend/pull/2) by [Clemens Brunner](https://github.com/cbrnr))
