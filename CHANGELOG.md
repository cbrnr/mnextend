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
