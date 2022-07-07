from matplotlib import pyplot as plt
import mne
import numpy as np
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
import sys
sys.path.append("D:/Projects/eeg_tools/src/")
import setup_eeg_tools as set
import pathlib
import os

# TODO: make PCA() executable for evoked objects too.
# TODO: fix quality_check().


def noise_rms(epochs):
    epochs_tmp = epochs.copy()
    n_epochs = epochs_tmp.get_data().shape[0]
    if not n_epochs % 2 == 0:
        epochs_tmp = epochs_tmp[:-1]
    n_epochs = epochs_tmp.get_data().shape[0]
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]
    evoked = epochs_tmp.average().get_data()
    rms = np.sqrt(np.mean(evoked**2))
    del epochs_tmp
    return rms, evoked


def snr(epochs):
    signals = []
    n_rms, n_evoked = noise_rms(epochs)
    for epoch in epochs:
        signal = epoch - n_evoked
        signals.append(signal)
    signals = np.array(signals)
    s_rms = np.sqrt(np.mean(signals**2, axis=0))
    snr = np.mean(s_rms/n_rms)  # signal rms divided by noise rms
    return snr


def PCA(epochs, n_components=5):
    X = epochs.get_data()
    pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
    pca_data = pca.fit_transform(X)
    evoked = mne.EvokedArray(np.mean(pca_data, axis=0),
                         mne.create_info(n_components, epochs.info['sfreq'],
                                         ch_types='eeg'), tmin=epochs.tmin)
    return evoked, pca_data


def quality_check(ids, fig_size=(60,60), out_folder="D:/EEG/vocal_effort/qc"):
    if not os.path.isdir(out_folder):
        os.makedirs(pathlib.Path(out_folder))
    id = ids[0]
    _fig_folder = pathlib.Path(f"D:/EEG/vocal_effort/data/{id}/figures")
    n_plots = len(os.listdir(_fig_folder))
    for n, subplots in enumerate(range(n_plots)):
        axs_size = int(round(np.sqrt(len(ids)) + 0.5))  # round up
        fig, axs = plt.subplots(axs_size, axs_size, figsize=fig_size)
        axs = axs.flatten()
        for i, id in enumerate(ids):
            _fig_folder = pathlib.Path(f"D:/EEG/vocal_effort/data/{id}/figures")
            figures = os.listdir(_fig_folder)
            figure_path = _fig_folder / figures[n]
            img = plt.imread(figure_path)
            axs[i].imshow(img)
            axs[i].set_axis_off()
            fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(pathlib.Path(out_folder) / figures[n])
        plt.close()

if __name__ == "__main__":
    root_dir = pathlib.Path("D:/EEG")
    header_files = set.load_file(type="header", dir=root_dir)
    ids = set.get_ids(header_files)
    data_path = sample.data_path()
    # Load and filter data, set up epochs
    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
    event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'
    tmin, tmax = -0.1, 0.3
    event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.filter(1, 20, fir_design='firwin')
    events = mne.read_events(event_fname)

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                        picks=picks, baseline=None, preload=True,
                        verbose=False)
    X = epochs.get_data()
    n_epochs, n_channels, n_times = epochs.get_data().shape
    X -= np.expand_dims(X.mean(axis=2), axis=2)  # center data on 0
    X = np.transpose(epochs._data,
                     (1, 0, 2)).reshape(n_channels,
                                        n_epochs * n_times).T  # concatenate
    C0 = X.T @ X  # Data covariance Matrix
    n_components = 5
    D, P = np.linalg.eig(C0)  # eigendecomposition of C0
    idx = np.argsort(D)[::-1][0:n_components]   # sort array
    # by descending magnitude
    D = D[idx]
    P = P[:, idx]  # rotation matrix
    pca_evokeds = []
    for cond in epochs.event_id.keys():
        # use rotation matrix on every single condition
        n_epochs, n_channels, n_times = epochs[cond]._data.shape
        X = epochs[cond]._data
        X -= np.expand_dims(X.mean(axis=2), axis=2)  # center data on 0
        X = np.transpose(epochs[cond]._data,
                         (1, 0, 2)).reshape(n_channels, n_epochs * n_times).T
        Y = X @ P  # get principle components
        pca = np.reshape(Y.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
        pca_evoked = mne.EvokedArray(np.mean(pca, axis=0),
                                     mne.create_info(
                                     n_components, epochs[cond].info["sfreq"],
                                     ch_types="eeg"),
                                     tmin=epochs[cond].tmin)
        pca_evoked.pick_channels(ch_names=["0"])
        pca_evokeds.append(pca_evoked)
