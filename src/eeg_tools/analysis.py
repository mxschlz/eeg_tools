from matplotlib import pyplot as plt
import mne
import numpy as np
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
import sys
import os
path = os.getcwd() + "\\src\\" + "eeg_tools"
sys.path.append(path)
import setup_eeg_tools as set
import pathlib

# TODO: fix quality_check().
# make PCA work for more than one component. Right now only works if n_components=1.

def snr(epochs):
    epochs_tmp = epochs.copy()
    n_epochs = epochs_tmp.get_data().shape[0]
    if not n_epochs % 2 == 0:
        epochs_tmp = epochs_tmp[:-1]
    n_epochs = epochs_tmp.get_data().shape[0]
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]
    noises = epochs_tmp.average().get_data()
    signals = list()
    for noise in noises:
        for epoch in epochs.average().get_data():
            signal = epoch-noise
        signals.append(signal)
    snr = signals / noises
    rms = np.mean(np.sqrt(snr**2))
    return rms


def PCA(epochs, n_components=1):
    X = epochs.get_data()
    n_epochs, n_channels, n_times = epochs.get_data().shape
    X -= np.expand_dims(X.mean(axis=2), axis=2)  # center data on 0
    X = np.transpose(epochs._data,
                     (1, 0, 2)).reshape(n_channels,
                                        n_epochs * n_times).T  # concatenate
    C0 = X.T @ X  # Data covariance Matrix
    D, P = np.linalg.eig(C0)  # eigendecomposition of C0
    idx = np.argsort(D)[::-1][0:n_components]   # sort array
    # by descending magnitude
    D = D[idx]
    P = P[:, idx]  # rotation matrix
    pca_evokeds = dict()
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
        for component in range(n_components):
            # pca_evokeds[cond] = pca_evoked.pick_channels(ch_names=list(str(component)) for component in range(n_components)))
             pca_evokeds[cond] = pca_evoked.pick_channels(ch_names=list(str(component)))
    return pca_evokeds


def quality_check(ids, out_folder, n_figs=12, fig_size=(60,60)):
    if not os.path.isdir(out_folder):
        os.makedirs(pathlib.Path(out_folder))
    for n, subplots in enumerate(range(n_figs)):
        axs_size = int(round(np.sqrt(len(ids)) + 0.5))  # round up
        fig, axs = plt.subplots(axs_size, axs_size, figsize=fig_size)
        axs = axs.flatten()
        for i, id in enumerate(ids):
            _fig_folder = pathlib.Path(f"D:/EEG/distance_perception/pinknoise/data/{id}/figures")
            figures = os.listdir(_fig_folder)
            figure_path = _fig_folder / figures[n]
            img = plt.imread(figure_path)
            axs[i].imshow(img)
            axs[i].set_axis_off()
            fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(pathlib.Path(out_folder) / figures[n])
        plt.close()


def get_evokeds(ids, root_dir, return_average=False):
    all_evokeds = dict()
    for id in ids:
        evokeds = set.read_object("evokeds", root_dir, id)
        for condition in evokeds:
            if condition.comment not in all_evokeds.keys():
                all_evokeds[condition.comment] = [condition]
            else:
                all_evokeds[condition.comment].append(condition)
    if return_average == True:
        evokeds_avrgd = dict()
        for key in all_evokeds:
            evokeds_avrgd[key] = mne.grand_average(all_evokeds[key])
        return all_evokeds, evokeds_avrgd
    else:
        return all_evokeds
    
