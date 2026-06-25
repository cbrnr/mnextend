# © MNEXTEND developers
#
# License: BSD (3-clause)

from pathlib import Path

import mne
import numpy as np
import pytest

from mnextend.iclabel import IC_LABELS, run_iclabel
from mnextend.iclabel.iclabel import _get_features


@pytest.fixture
def iclabel_dataset(request):
    """Load one of the pre-generated iclabel npz datasets."""
    data_path = Path(__file__).parent / "data" / f"iclabel_{request.param}.npz"
    data = np.load(data_path)

    info = mne.create_info(list(data["ch_names"]), data["sfreq"], ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    if data["raw"].ndim == 3:
        inst = mne.EpochsArray(data["raw"], info)
    else:
        inst = mne.io.RawArray(data["raw"], info)

    ica = mne.preprocessing.ICA(n_components=int(data["n_components"]))
    ica.unmixing_matrix_ = data["unmixing_matrix"]
    ica.pca_components_ = data["pca_components"]
    ica.pca_mean_ = data["pca_mean"]
    ica.n_components_ = int(data["n_components"])
    ica.info = info

    return inst, ica, data


@pytest.mark.parametrize(
    "iclabel_dataset", ["continuous", "short", "epoched"], indirect=True
)
def test_iclabel_features(iclabel_dataset):
    inst, ica, data = iclabel_dataset
    topo, psd, autocorr = _get_features(inst, ica)

    np.testing.assert_allclose(topo, data["ref_topo"], atol=1e-5)
    np.testing.assert_allclose(psd, data["ref_psd"], atol=1e-5)
    np.testing.assert_allclose(autocorr, data["ref_autocorr"], atol=1e-5)


@pytest.mark.parametrize(
    "iclabel_dataset", ["continuous", "short", "epoched"], indirect=True
)
def test_iclabel_probabilities(iclabel_dataset):
    inst, ica, data = iclabel_dataset
    prob = run_iclabel(inst, ica)

    np.testing.assert_allclose(prob, data["ref_prob"], atol=1e-5)


@pytest.mark.parametrize(
    "iclabel_dataset", ["continuous", "short", "epoched"], indirect=True
)
def test_iclabel_output_shape(iclabel_dataset):
    inst, ica, data = iclabel_dataset
    prob = run_iclabel(inst, ica)

    assert prob.shape == (int(data["n_components"]), 7)


@pytest.mark.parametrize(
    "iclabel_dataset", ["continuous", "short", "epoched"], indirect=True
)
def test_iclabel_probabilities_sum_to_one(iclabel_dataset):
    inst, ica, data = iclabel_dataset
    prob = run_iclabel(inst, ica)

    np.testing.assert_allclose(
        prob.sum(axis=1), np.ones(int(data["n_components"])), atol=1e-5
    )


def test_ic_labels():
    assert len(IC_LABELS) == 7
    assert IC_LABELS[0] == "brain"
    assert IC_LABELS[-1] == "other"


def test_iclabel_no_montage():
    info = mne.create_info(["Fp1", "Fp2", "Fz"], sfreq=256, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((3, 512)), info)

    ica = mne.preprocessing.ICA(n_components=2)
    ica.unmixing_matrix_ = np.eye(2)
    ica.pca_components_ = np.eye(2, 3)
    ica.pca_mean_ = np.zeros(3)
    ica.n_components_ = 2
    ica.info = info
    ica.ch_names = ["Fp1", "Fp2", "Fz"]

    with pytest.raises(ValueError, match="Montage must be set"):
        run_iclabel(raw, ica)
