from matplotlib import pyplot as plt
import mne
import numpy as np
from mne.datasets import sample
_scaling = 10**6

# TODO: make PCA() executable for evoked objects too.


def noise_rms(epochs):
    global scaling
    epochs_tmp = epochs.copy()
    n_epochs = epochs.get_data().shape[0]
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]
    evoked = epochs_tmp.average().get_data()
    rms = np.sqrt(np.mean(evoked**2)) * _scaling
    del epochs_tmp
    return rms


def snr(epochs, signal_interval=(0.15, 0.2)):
    """
    Compute signal-to-noise ratio. Take root mean square of noise
    plus signal (interval where evoked activity is expected)
    and return the quotient.
    """
    signal = epochs.copy()
    signal.crop(signal_interval[0], signal_interval[1])
    n_rms = noise_rms(epochs)
    s_rms = np.sqrt(np.mean(signal.average().get_data()**2)) * _scaling
    snr = s_rms / n_rms  # signal rms divided by noise rms
    return snr


def PCA(epochs, n_components=5):
if __name__ == "__main__":
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
