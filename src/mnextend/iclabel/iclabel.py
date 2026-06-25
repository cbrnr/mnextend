# © MNEXTEND developers
#
# License: BSD (3-clause)

"""Automatic ICA component labeling using ICLabel.

The ONNX model (`ICLabelNet.onnx`) and parts of the feature extraction logic are based
on the `mne-icalabel` package (https://github.com/mne-tools/mne-icalabel).
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import numpy as np
import onnx
from mne import BaseEpochs
from mne.io import BaseRaw, RawArray
from scipy.signal import resample_poly

IC_LABELS = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]


def _get_topomaps(inst, icawinv, picks):
    """Generate topographic maps for ICA components.

    Parameters
    ----------
    inst : instance of Raw | Epochs
        The data instance. Used to retrieve channel locations (montage).
    icawinv : np.ndarray
        The inverse ICA weights matrix (mixing matrix).
    picks : list
        List of channel names to include in the topoplot generation.

    Returns
    -------
    np.ndarray, shape (n_components, 1, 32, 32)
        The interpolated topographic maps.

    Notes
    -----
    Adapted from `mne-icalabel` (`_eeg_topoplot` and `_topoplotFast`).
    """
    rd, th = _get_eeglab_coords(inst, picks)
    th = np.pi / 180 * th  # convert degrees to radians

    n_comp = icawinv.shape[-1]
    topo = np.zeros((n_comp, 32, 32), dtype=np.float32)

    grid_scale = 32
    rmax = 0.5

    x = rd * np.cos(th)
    y = rd * np.sin(th)

    plotrad = min(1, np.max(rd) * 1.02)
    plotrad = max(plotrad, 0.5)

    squeezefac = rmax / plotrad
    x *= squeezefac
    y *= squeezefac

    xmin, xmax = min(-rmax, x.min()), max(rmax, x.max())
    ymin, ymax = min(-rmax, y.min()), max(rmax, y.max())

    xi = np.linspace(xmin, xmax, grid_scale)
    yi = np.linspace(ymin, ymax, grid_scale)
    XQ, YQ = np.meshgrid(xi, yi)

    mask = np.sqrt(XQ**2 + YQ**2) <= rmax

    xy_grid = (XQ + 1j * YQ).flatten()
    xy_sensors = (x + 1j * y).flatten()
    d = np.abs(xy_grid[:, None] - xy_sensors[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        g = (d**2) * (np.log(d) - 1)
    g[np.isnan(g)] = 0

    d_s = np.abs(xy_sensors[:, None] - xy_sensors[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        g_s = (d_s**2) * (np.log(d_s) - 1)
    np.fill_diagonal(g_s, 0)

    weights, *_ = np.linalg.lstsq(g_s, icawinv, rcond=None)
    Zi_all = g @ weights

    for it in range(n_comp):
        Zi = Zi_all[:, it].reshape(grid_scale, grid_scale)
        Zi = Zi.T
        Zi[~mask] = 0

        max_val = np.max(np.abs(Zi))
        if max_val != 0:
            Zi /= max_val

        topo[it, :, :] = Zi

    return topo[:, np.newaxis, :, :]


def _get_eeglab_coords(inst, picks):
    """Calculate normalized polar coordinates (rd, th) for the topomaps.

    Parameters
    ----------
    inst : instance of Raw | Epochs
        The data instance to retrieve channel locations.
    picks : list
        List of channel names.

    Returns
    -------
    rd : np.ndarray, shape (1, n_channels)
        Normalized radial distances.
    th : np.ndarray, shape (1, n_channels)
        Azimuth angles in degrees.

    Notes
    -----
    Adapted from `mne-icalabel` (`_mne_to_eeglab_locs`).
    """
    montage = inst.get_montage()
    if montage is None:
        raise ValueError("Montage must be set before ICLabel classification.")

    positions = montage.get_positions()
    full_ch_pos = positions["ch_pos"]
    ch_pos = {ch: full_ch_pos[ch] for ch in picks} if picks is not None else full_ch_pos
    empty = [key for key in ch_pos if np.all(np.isnan(ch_pos[key]))]
    if len(empty) != 0:
        raise ValueError("Channel positions are missing.")

    locs = np.vstack(list(ch_pos.values()))

    x = locs[:, 1]
    y = -1 * locs[:, 0]
    z = locs[:, 2]

    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))

    rd = (np.pi / 2 - elevation) / np.pi
    th = np.degrees(-azimuth)

    return rd.reshape([1, -1]), th.reshape([1, -1])


def _get_autocorrelation(ica_act, sfreq):
    """Compute autocorrelation features for ICA components.

    Parameters
    ----------
    ica_act : np.ndarray
        The ICA activations.
    sfreq : float
        The sampling frequency of the data.

    Returns
    -------
    np.ndarray, shape (n_components, 1, 1, 100)
        The autocorrelation features reshaped for classification.

    Notes
    -----
    Adapted from `mne-icalabel` (`_eeg_autocorr_welch`, `_eeg_autocorr`, and
    `_eeg_autocorr_fftw`).
    """
    n_lags = int(sfreq) + 1
    if ica_act.ndim == 3:  # epoched data: (n_comp, n_times, n_epochs)
        n_comp, n_times, _ = ica_act.shape

        nfft = 2 ** int(np.ceil(np.log2(2 * n_times - 1)))

        fft_data = np.fft.rfft(ica_act, nfft, axis=1)
        psd = np.abs(fft_data) ** 2
        psd_mean = np.mean(psd, axis=2)

        ac = np.fft.irfft(psd_mean, n=nfft, axis=1)

        if ac.shape[1] > n_lags:
            ac = ac[:, :n_lags]
        else:
            padding = np.zeros((n_comp, n_lags - ac.shape[1]))
            ac = np.hstack([ac, padding])

        var = ac[:, 0:1]
        var[var == 0] = 1.0
        ac = ac / var
    else:  # continuous data
        n_comp, n_points = ica_act.shape

        if n_points > 5 * sfreq:  # Welch method for long signals
            n_seg = int(min(n_points, 3 * sfreq))
            n_overlap = n_seg // 2

            nfft = 2 ** int(np.ceil(np.log2(2 * n_seg - 1)))

            step = n_seg - n_overlap
            cutoff = (n_points // n_seg) * n_seg
            starts = np.arange(0, cutoff - n_seg + step, step)
            if len(starts) == 0:
                starts = [0]

            ac = np.zeros((n_comp, nfft // 2 + 1))

            for start in starts:
                segment = ica_act[:, start : start + n_seg]
                fft_segment = np.fft.rfft(segment, nfft, axis=1)
                ac += np.abs(fft_segment) ** 2

            ac_mean = ac / len(starts)
            ac = np.fft.irfft(ac_mean, n=nfft, axis=1)

            ac = ac[:, :n_lags]
            lags = np.arange(n_seg, n_seg - n_lags, -1)
            lags[lags <= 0] = 1

            denom = ac[:, 0:1] * (lags / n_seg)
            denom[denom == 0] = 1.0
            ac = ac / denom

        else:  # FFT-based for short signals
            nfft = 2 ** int(np.ceil(np.log2(2 * n_points - 1)))

            fft_data = np.fft.rfft(ica_act, nfft, axis=1)
            psd = np.abs(fft_data) ** 2
            ac = np.fft.irfft(psd, n=nfft, axis=1)

            if ac.shape[1] > n_lags:
                ac = ac[:, :n_lags]
            else:
                padding = np.zeros((n_comp, n_lags - ac.shape[1]))
                ac = np.hstack([ac, padding])

            var = ac[:, 0:1]
            var[var == 0] = 1.0
            ac = ac / var

    down = int(sfreq)
    target_fs = 100

    if 101 < ac.shape[1] * 100 / down:
        down += 1
    ac_resampled = resample_poly(ac, up=target_fs, down=down, axis=1)

    final_ac = ac_resampled[:, 1:]  # lag 0 is constant, drop it

    return final_ac[:, np.newaxis, np.newaxis, :].astype(np.float32)


def _get_psd(ica_act, sfreq):
    """Compute the Power Spectral Density (PSD) of ICA components.

    Parameters
    ----------
    ica_act : np.ndarray
        The ICA component activations.
    sfreq : float
        The sampling frequency of the data.

    Returns
    -------
    np.ndarray, shape (n_components, 1, 1, 100)
        The computed PSD values in dB.

    Notes
    -----
    Adapted from `mne-icalabel` (`_eeg_rpsd_constants`, `_eeg_rpsd_compute_psdmed`,
    and `_eeg_rpsd_format`).
    """
    if ica_act.ndim == 2:
        ica_act = ica_act[:, :, np.newaxis]

    n_comp, n_times, n_epochs = ica_act.shape

    nyquist = int(np.floor(sfreq / 2))
    n_freqs = nyquist if nyquist < 100 else 100
    n_points = min(n_times, int(sfreq))

    cutoff = np.floor(n_times / n_points) * n_points
    range_ = np.ceil(np.arange(0, cutoff - n_points + n_points / 2, n_points / 2))

    index = np.tile(range_, (n_points, 1)).T + np.arange(0, n_points)
    index = index.T.astype(int)

    n_segments_per_epoch = index.shape[1]

    window = np.hamming(n_points)
    window = window[:, np.newaxis]
    denominator = sfreq * np.sum(window**2)

    psd_list = []

    for comp_idx in range(n_comp):
        comp_data = ica_act[comp_idx]

        windowed_data = comp_data[index, :]

        time_segments = windowed_data.reshape(
            n_points, n_segments_per_epoch * n_epochs, order="F"
        )

        time_segments *= window

        fft = np.fft.fft(time_segments, n=n_points, axis=0)
        power_spectrum = np.abs(fft) ** 2

        subset_power = power_spectrum[1 : n_freqs + 1, :] * 2 / denominator

        if n_freqs == nyquist:
            subset_power[-1, :] /= 2

        median_power = np.median(subset_power, axis=1)
        psd_comp = 20 * np.log10(median_power)
        psd_list.append(psd_comp)

    psd = np.array(psd_list)

    if n_freqs < 100:
        psd = np.pad(psd, ((0, 0), (0, 100 - n_freqs)), mode="edge")

    for linenoise_ind in [49, 59]:
        neighbors_idx = [linenoise_ind - 1, linenoise_ind + 1]
        difference = psd[:, neighbors_idx] - psd[:, linenoise_ind, np.newaxis]
        is_notch_artifact = np.all(difference > 5, axis=1)

        if np.any(is_notch_artifact):
            psd[is_notch_artifact, linenoise_ind] = np.mean(
                psd[is_notch_artifact][:, neighbors_idx], axis=1
            )

    max_abs = np.max(np.abs(psd), axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    psd /= max_abs

    return psd[:, np.newaxis, np.newaxis, :].astype(np.float32)


def _get_ica_data(inst, ica):
    """Compute ICA inverse weights and activations.

    Parameters
    ----------
    inst : instance of Raw | Epochs
        The data instance containing the channel data.
    ica : ICA
        The fitted ICA instance.

    Returns
    -------
    icawinv : np.ndarray, shape (n_channels, n_components)
        The inverse ICA weights matrix (mixing matrix).
    icaact : np.ndarray, shape (n_components, n_times[, n_epochs])
        The ICA component activations.

    Notes
    -----
    Adapted from `mne-icalabel` (`_retrieve_eeglab_icawinv` and
    `_compute_ica_activations`).
    """
    weights = ica.unmixing_matrix_ @ ica.pca_components_[: ica.n_components_]
    icawinv = np.linalg.pinv(weights)

    data = inst.get_data(picks=ica.ch_names) * 1e6

    icaact = weights @ data

    if isinstance(inst, BaseEpochs):
        icaact = icaact.transpose(1, 2, 0)

    return icawinv, icaact


def _get_features(inst, ica):
    """Extract and format topographic, PSD, and autocorrelation features.

    Parameters
    ----------
    inst : instance of Raw | Epochs
        The data instance containing the channel data.
    ica : ICA
        The fitted ICA instance.

    Returns
    -------
    topo_features_formatted : np.ndarray, shape (4 * n_components, 1, 32, 32)
        The formatted topographic map features.
    psd_formatted : np.ndarray, shape (4 * n_components, 1, 1, 100)
        The formatted PSD features.
    autocorr_formatted : np.ndarray, shape (4 * n_components, 1, 1, 100)
        The formatted autocorrelation features.

    Notes
    -----
    Adapted from `mne-icalabel` (`get_iclabel_features` and `_format_input`).
    """
    icawinv, ica_act = _get_ica_data(inst, ica)

    topo_features = _get_topomaps(inst, icawinv, ica.ch_names)
    psd_features = _get_psd(ica_act, inst.info["sfreq"])
    autocorr_features = _get_autocorrelation(ica_act, inst.info["sfreq"])

    topo_features *= 0.99
    psd_features *= 0.99
    autocorr_features *= 0.99

    # the model expects 4 versions of each topomap (normal, negated, H-flipped, both)
    topo_features_formatted = np.concatenate(
        [
            topo_features,
            -topo_features,
            np.flip(topo_features, axis=3),
            np.flip(-topo_features, axis=3),
        ],
        axis=0,
    )
    psd_formatted = np.tile(psd_features, (4, 1, 1, 1))
    autocorr_formatted = np.tile(autocorr_features, (4, 1, 1, 1))

    return topo_features_formatted, psd_formatted, autocorr_formatted


def _load_onnx_weights(onnx_path):
    """Load weights and biases from ONNX model.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model file.

    Returns
    -------
    dict
        Dictionary containing all weights and biases from the model.
    """
    model = onnx.load(str(onnx_path))

    weights = {}
    for initializer in model.graph.initializer:
        name = initializer.name
        if initializer.data_type == 1:  # float32
            weights[name] = np.frombuffer(
                initializer.raw_data, dtype=np.float32
            ).reshape(initializer.dims)

    return weights


def _im2col(x, kernel_h, kernel_w, stride=1, padding=0):
    """Convert image batch to column matrix for vectorized convolution.

    Parameters
    ----------
    x : np.ndarray, shape (batch, channels, height, width)
        Input tensor.
    kernel_h : int
        Kernel height.
    kernel_w : int
        Kernel width.
    stride : int
        Stride of the convolution.
    padding : int or tuple
        Padding added to sides. Can be int or (pad_h, pad_w).

    Returns
    -------
    cols : np.ndarray, shape (batch, channels*kernel_h*kernel_w, h_out*w_out)
        Column matrix.
    h_out : int
        Output height.
    w_out : int
        Output width.
    """
    batch, channels, h, w = x.shape

    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
        h, w = x.shape[2], x.shape[3]

    h_out = (h - kernel_h) // stride + 1
    w_out = (w - kernel_w) // stride + 1

    cols = np.zeros((batch, channels, kernel_h, kernel_w, h_out, w_out), dtype=x.dtype)

    for i in range(kernel_h):
        i_max = i + stride * h_out
        for j in range(kernel_w):
            j_max = j + stride * w_out
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]

    cols = cols.reshape(batch, channels * kernel_h * kernel_w, h_out * w_out)

    return cols, h_out, w_out


def _conv2d_numpy(x, weight, bias, stride=1, padding=0):
    """Perform 2D convolution using vectorized im2col approach.

    Parameters
    ----------
    x : np.ndarray, shape (batch, in_channels, height, width)
        Input tensor.
    weight : np.ndarray, shape (out_channels, in_channels, kh, kw)
        Convolution kernel.
    bias : np.ndarray, shape (out_channels,)
        Bias term.
    stride : int
        Stride of the convolution.
    padding : int or tuple
        Padding added to both sides. Can be int or (pad_h, pad_w).

    Returns
    -------
    out : np.ndarray
        Output tensor after convolution.
    """
    batch, in_channels, h, w = x.shape
    out_channels, _, kh, kw = weight.shape

    x_col, h_out, w_out = _im2col(x, kh, kw, stride, padding)

    weight_col = weight.reshape(out_channels, -1)

    out = np.zeros((batch, out_channels, h_out * w_out), dtype=np.float32)
    for b in range(batch):
        out[b] = weight_col @ x_col[b]

    out += bias[None, :, None]

    return out.reshape(batch, out_channels, h_out, w_out)


def _leaky_relu(x, alpha=0.2):
    """Apply Leaky ReLU activation."""
    return np.where(x > 0, x, alpha * x)


def _softmax(x, axis=-1):
    """Apply softmax activation."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _iclabel_forward_numpy(images, psd, autocorr, weights):
    """Forward pass through ICLabel network.

    Parameters
    ----------
    images : np.ndarray, shape (batch, 1, 32, 32)
        Topographic maps.
    psd : np.ndarray, shape (batch, 1, 1, 100)
        Power spectral density.
    autocorr : np.ndarray, shape (batch, 1, 1, 100)
        Autocorrelation.
    weights : dict
        Dictionary containing model weights and biases.

    Returns
    -------
    probs : np.ndarray, shape (batch // 4, 7)
        Class probabilities for each component.
    """
    x_img = _conv2d_numpy(
        images,
        weights["img_conv.conv1.weight"],
        weights["img_conv.conv1.bias"],
        stride=2,
        padding=1,
    )
    x_img = _leaky_relu(x_img)
    x_img = _conv2d_numpy(
        x_img,
        weights["img_conv.conv2.weight"],
        weights["img_conv.conv2.bias"],
        stride=2,
        padding=1,
    )
    x_img = _leaky_relu(x_img)
    x_img = _conv2d_numpy(
        x_img,
        weights["img_conv.conv3.weight"],
        weights["img_conv.conv3.bias"],
        stride=2,
        padding=1,
    )
    x_img = _leaky_relu(x_img)

    x_psd = _conv2d_numpy(
        psd,
        weights["psds_conv.conv1.weight"],
        weights["psds_conv.conv1.bias"],
        padding=(0, 1),
    )
    x_psd = _leaky_relu(x_psd)
    x_psd = _conv2d_numpy(
        x_psd,
        weights["psds_conv.conv2.weight"],
        weights["psds_conv.conv2.bias"],
        padding=(0, 1),
    )
    x_psd = _leaky_relu(x_psd)
    x_psd = _conv2d_numpy(
        x_psd,
        weights["psds_conv.conv3.weight"],
        weights["psds_conv.conv3.bias"],
        padding=(0, 1),
    )
    x_psd = _leaky_relu(x_psd)

    x_autocorr = _conv2d_numpy(
        autocorr,
        weights["autocorr_conv.conv1.weight"],
        weights["autocorr_conv.conv1.bias"],
        padding=(0, 1),
    )
    x_autocorr = _leaky_relu(x_autocorr)
    x_autocorr = _conv2d_numpy(
        x_autocorr,
        weights["autocorr_conv.conv2.weight"],
        weights["autocorr_conv.conv2.bias"],
        padding=(0, 1),
    )
    x_autocorr = _leaky_relu(x_autocorr)
    x_autocorr = _conv2d_numpy(
        x_autocorr,
        weights["autocorr_conv.conv3.weight"],
        weights["autocorr_conv.conv3.bias"],
        padding=(0, 1),
    )
    x_autocorr = _leaky_relu(x_autocorr)

    batch, img_channels, img_h, img_w = x_img.shape

    x_psd_flat = x_psd.reshape(batch, -1, 1, 1)
    x_autocorr_flat = x_autocorr.reshape(batch, -1, 1, 1)

    x_psd_tiled = np.tile(x_psd_flat, (1, 1, img_h, img_w))
    x_autocorr_tiled = np.tile(x_autocorr_flat, (1, 1, img_h, img_w))

    x = np.concatenate([x_img, x_psd_tiled, x_autocorr_tiled], axis=1)

    x = _conv2d_numpy(x, weights["conv.weight"], weights["conv.bias"])

    x = _softmax(x, axis=1)
    x = np.mean(x, axis=(2, 3))

    # average over 4 augmented versions per component
    # original order: [comp0_v0, comp1_v0, ..., compN_v0, comp0_v1, ...]
    x = x.reshape(4, batch // 4, -1).transpose(1, 0, 2).mean(axis=1)

    return x


def run_iclabel(inst, ica):
    """Run ICLabel classification on ICA components.

    Parameters
    ----------
    inst : instance of Raw | Epochs
        The data instance containing the channel data. A montage must be set.
    ica : ICA
        The fitted ICA instance.

    Returns
    -------
    np.ndarray, shape (n_components, 7)
        Estimated probabilities for each component class. Columns correspond to:
        brain, muscle, eye, heart, line noise, channel noise, other.
    """
    onnx_path = Path(__file__).parent / "ICLabelNet.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ICLabel ONNX model not found at {onnx_path}")

    images, psd, autocorrelation = _get_features(inst, ica)

    weights = _load_onnx_weights(onnx_path)
    probs = _iclabel_forward_numpy(images, psd, autocorrelation, weights)

    return probs


def plot_ica_components(inst, ica, probs, picks=None, show=True, **kwargs):
    """Plot ICA component properties with ICLabel classification probabilities.

    Wraps `ICA.plot_properties` and adds a bar chart of the seven ICLabel class
    probabilities below the topographic map for each component.

    Parameters
    ----------
    inst : instance of Raw | Epochs
        The data instance.
    ica : ICA
        The fitted ICA instance.
    probs : np.ndarray, shape (n_components, 7)
        ICLabel class probabilities as returned by `run_iclabel`. Columns correspond
        to: brain, muscle, eye, heart, line noise, channel noise, other.
    picks : int | list of int | None
        Indices of components to plot. If `None`, all components are plotted.
    show : bool
        Whether to show the figures immediately.
    **kwargs
        Additional keyword arguments forwarded to `ICA.plot_properties` (e.g., `dB`,
        `plot_std`, `topomap_args`, `psd_args`, `figsize`, `reject`).

    Returns
    -------
    list of Figure
        One `matplotlib.figure.Figure` per component.
    """
    if picks is None:
        picks = list(range(ica.n_components_))
    elif isinstance(picks, int):
        picks = [picks]

    bar_labels = [label.replace("_", "\n") for label in IC_LABELS]

    figs = []
    for comp_id in picks:
        if isinstance(inst, BaseRaw):
            try:
                (fig,) = ica.plot_properties(inst, picks=comp_id, show=False, **kwargs)
            except RuntimeError:
                good_data = inst.get_data(reject_by_annotation="omit")
                inst_plot = (
                    RawArray(good_data, inst.info.copy(), verbose=False)
                    if good_data.shape[1] >= int(2 * inst.info["sfreq"])
                    else inst
                )
                (fig,) = ica.plot_properties(
                    inst_plot,
                    picks=comp_id,
                    show=False,
                    reject_by_annotation=False,
                    **{k: v for k, v in kwargs.items() if k != "reject_by_annotation"},
                )
        else:
            (fig,) = ica.plot_properties(inst, picks=comp_id, show=False, **kwargs)

        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h + 0.5)

        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 0.3, 1])
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[:2, 0], hspace=0.2, height_ratios=[3.75, 1]
        )
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[:2, 1], hspace=0, height_ratios=[3.5, 1]
        )

        fig.axes[0].set_subplotspec(gs_left[0])
        fig.axes[1].set_subplotspec(gs_right[0])
        fig.axes[2].set_subplotspec(gs_right[1])
        fig.axes[3].set_subplotspec(gs[2, 0])
        fig.axes[4].set_subplotspec(gs[2, 1])

        ic_probs = probs[comp_id]
        ax = fig.add_subplot(gs_left[1])

        colors = ["#4c72b0"] * len(bar_labels)
        colors[np.argmax(ic_probs)] = "#228B22"

        x_pos = range(len(bar_labels))
        ax.bar(x_pos, ic_probs, color=colors)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, ha="center", fontsize=7)
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_facecolor("none")

        for spine_name, spine in ax.spines.items():
            if spine_name != "bottom":
                spine.set_visible(False)

        for i, v in enumerate(ic_probs):
            ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)

        fig.align_ylabels([ax, fig.axes[3]])
        fig.tight_layout()

        if show:
            fig.show()

        figs.append(fig)

    return figs
