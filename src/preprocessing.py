import sys
sys.path.append("D:/Projects/eeg_tools/src/")
import analysis
import setup_eeg_tools as stp
import os
import pathlib
import numpy as np
import mne
from matplotlib import pyplot as plt, patches
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA
import json
import glob
_scaling = 10**6

# TODO: describe workflow of processing data.


def make_raw(header_files, id, ref_ch="FCz", preload=True, add_ref_ch=True,
             mapping=None, montage=None):
    raw_files = []
    for header_file in header_files:
        if id in header_file:
            raw_files.append(mne.io.read_raw_brainvision(
                header_file, preload=preload))  # read BrainVision files.
    raw = mne.concatenate_raws(raw_files)  # make raw files
    if mapping:
        raw.rename_channels(mapping)
    if add_ref_ch:
        raw.add_reference_channels(ref_ch)
    if montage:
        raw.set_montage(montage)
    return raw


def filtering(data, notch=None, highpass=None, lowpass=None, plot=True):
    """
    Apply FIR filter to the raw dataset. Make a 2 by 2 plot with time
    series data and power spectral density before and after.
    """
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle("Power Spectral Density")
    ax[0].set_title("before filtering")
    ax[1].set_title("after filtering")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    if plot == True:
        # FCz has zero voltage at this point.
        data.plot_psd(ax=ax[0], show=False, exclude=["FCz"])
    if notch is not None:  # ZapLine Notch filter
        X = data.get_data().T
        # remove power line noise with the zapline algorithm
        X, _ = dss_line_iter(X, fline=cfg["filtering"]["notch"],
                             sfreq=data.info["sfreq"],
                             nfft=cfg["filtering"]["nfft"])
        data._data = X.T  # put the data back into variable
        del X
    if lowpass is not None:
        data.filter(h_freq=lowpass, l_freq=None)
    if highpass is not None:
        data.filter(h_freq=None, l_freq=highpass)
    if plot == True:
        data.plot_psd(ax=ax[1], show=False, exclude=["FCz"])
    if lowpass is not None and highpass == None:
        fig.savefig(
            _fig_folder / pathlib.Path("lowpassfilter.pdf"), dpi=800)
    if highpass is not None and lowpass == None:
        fig.savefig(
            _fig_folder / pathlib.Path("highpassfilter.pdf"), dpi=800)
    if highpass and lowpass is not None:
        fig.savefig(
            _fig_folder / pathlib.Path("bandpassfilter.pdf"), dpi=800)
    if notch is not None:
        fig.savefig(
            _fig_folder / pathlib.Path("ZapLine_filter.pdf"), dpi=800)
    plt.close()
    return data


def reref(epochs, ransac_parameters, type="average", plot=True):
    """
    If type "average": Create a robust average reference by first interpolating the bad channels
    to exclude outliers. Take mean voltage over all inlier channels as reference.
    If type "rest": use reference electrode standardization technique (point at infinity).
    epochs: mne.Epoch object.
    type: string --> "average", "rest", "lm" (linked mastoids)
    """
    if type == "average":
        epochs_clean = epochs.copy()
        ransac = Ransac(**ransac_parameters)  # optimize speed
        ransac.fit(epochs_clean)
        epochs_clean.average().plot(exclude=[])
        bads = input("Visual inspection for bad sensors: ").split()
        if len(bads) != 0 and bads not in ransac.bad_chs_:
            ransac.bad_chs_.extend(bads)
        epochs_clean = ransac.transform(epochs_clean)
        evoked = epochs.average()
        evoked_clean = epochs_clean.average()
        evoked.info['bads'] = ransac.bad_chs_
        evoked_clean.info['bads'] = ransac.bad_chs_
        fig, ax = plt.subplots(2)
        evoked.plot(exclude=[], axes=ax[0], show=False)
        evoked_clean.plot(exclude=[], axes=ax[1], show=False)
        ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
        ax[1].set_title("After RANSAC")
        fig.tight_layout()
        fig.savefig(
            _fig_folder / pathlib.Path("RANSAC_results.pdf"), dpi=800)
        plt.close()
        epochs = epochs_clean.copy()
        epochs_clean.set_eeg_reference(ref_channels="average", projection=True)
        average_reference = epochs_clean.info["projs"]
        epochs_clean.add_proj(average_reference)
        epochs_clean.apply_proj()
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_clean)
        epochs_reref = epochs_clean.copy()
    if type == "rest":
        sphere = mne.make_sphere_model("auto", "auto", epochs.info)
        src = mne.setup_volume_source_space(
            sphere=sphere, exclude=30., pos=5.)
        forward = mne.make_forward_solution(
            epochs.info, trans=None, src=src, bem=sphere)
        epochs_reref = epochs.copy().set_eeg_reference("REST", forward=forward)
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_reref)
    if type == "lm":
        epochs_reref = epochs.copy().set_eeg_reference(["TP9", "TP10"])
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_reref)
    if plot == True:
        fig, ax = plt.subplots(2)
        epochs.average().plot(axes=ax[0], show=False)
        epochs_reref.average().plot(axes=ax[1], show=False)
        ax[0].set_title(f"FCz, SNR={snr_pre:.2f}")
        ax[1].set_title(f"{type}, SNR={snr_post:.2f}")
        fig.tight_layout()
        fig.savefig(
            _fig_folder / pathlib.Path(f"{type}_reference.pdf"), dpi=800)
        plt.close()
    return epochs_reref


def apply_ICA(epochs, reference, n_components=None, method="fastica",
              threshold="auto"):
    """
    Run independent component analysis. Fit all epochs to the mne.ICA class, use
    reference_ica.fif to show the algorithm how blinks and saccades look like.
    Apply ica and save components to keep track of the excluded component topography.
    """
    epochs_ica = epochs.copy()
    snr_pre_ica = analysis.snr(epochs_ica)
    # ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    # ar.fit(epochs_ica)
    # epochs_ar, reject_log = ar.transform(epochs_ica, return_log=True)
    ica = ICA(n_components=n_components, method=method)
    # ica.fit(epochs_ica[~reject_log.bad_epochs])
    ica.fit(epochs_ica)
    # reference ICA containing blink and saccade components.
    ref = reference
    # .labels_ dict must contain "blinks" key with int values.
    components = ref.labels_["blinks"]
    for component in components:
        mne.preprocessing.corrmap([ref, ica], template=(0, components[component]),
                                  label="blinks", plot=False, threshold=cfg["ica"]["threshold"])
        ica.apply(epochs_ica, exclude=ica.labels_["blinks"])  # apply ICA
    ica.plot_components(ica.labels_["blinks"], show=False)
    plt.savefig(_fig_folder / pathlib.Path("ICA_components.pdf"), dpi=800)
    plt.close()
    ica.plot_sources(inst=epochs, show=False, start=0,
                     stop=10, show_scrollbars=False)
    plt.savefig(_fig_folder / pathlib.Path(f"ICA_sources.pdf"), dpi=800)
    plt.close()
    snr_post_ica = analysis.snr(epochs_ica)
    ica.plot_overlay(epochs.average(), exclude=ica.labels_["blinks"],
                     show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
    plt.savefig(_fig_folder / pathlib.Path("ICA_results.pdf"), dpi=800)
    plt.close()
    return epochs_ica


def autoreject_epochs(epochs,
                      n_interpolate=[1, 4, 8, 16],
                      consensus=None,
                      cv=10,
                      thresh_method="bayesian optimization",
                      n_jobs=-1,
                      random_state=None):
    """
    Automatically reject epochs via AutoReject algorithm:
    Computation of sensor-wise peak-to-peak-amplitude thresholds
    via cross-validation.
    """
    ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    fig, ax = plt.subplots(2)
    # plotipyt histogram of rejection thresholds
    ax[0].set_title("Rejection Thresholds")
    ax[0].hist(1e6 * np.array(list(ar.threshes_.values())), 30,
               color="g", alpha=0.4)
    ax[0].set(xlabel="Threshold (μV)", ylabel="Number of sensors")
    # plot cross validation error:
    loss = ar.loss_["eeg"].mean(axis=-1)  # losses are stored by channel type.
    im = ax[1].matshow(loss.T * 1e6, cmap=plt.get_cmap("viridis"))
    ax[1].set_xticks(range(len(ar.consensus)))
    ax[1].set_xticklabels(["%.1f" % c for c in ar.consensus])
    ax[1].set_yticks(range(len(ar.n_interpolate)))
    ax[1].set_yticklabels(ar.n_interpolate)
    # Draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor="r", facecolor="none")
    ax[1].add_patch(rect)
    ax[1].xaxis.set_ticks_position("bottom")
    ax[1].set(xlabel=r"Consensus percentage $\kappa$",
              ylabel=r"Max sensors interpolated $\rho$",
              title="Mean cross validation error (x 1e6)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(_fig_folder / pathlib.Path("autoreject_best_fit.pdf"), dpi=800)
    plt.close()
    evoked_bad = epochs[reject_log.bad_epochs].average()
    snr_ar = analysis.snr(epochs_ar)
    plt.plot(evoked_bad.times, evoked_bad.data.T * 1e06, 'r', zorder=-1)
    epochs_ar.average().plot(axes=plt.gca(), show=False,
                             titles=f"SNR: {snr_ar:.2f}")
    plt.savefig(
        _fig_folder / pathlib.Path("autoreject_results.pdf"), dpi=800)
    plt.close()
    epochs_ar.plot_drop_log(show=False)
    plt.savefig(
        _fig_folder / pathlib.Path("epochs_drop_log.pdf"), dpi=800)
    plt.close()
    return epochs_ar


def run_pipeline(raw, config, fig_folder, ica_ref=None, save=True):
    global _fig_folder
    _fig_folder = fig_folder
    if config == None:
        raise FileNotFoundError(
            "Need config file to preprocess data according to parameters!")
    else:
        if "filtering" in config:
            raw = filtering(data=raw, **config["filtering"])
        if "epochs" in config:
            epochs = mne.Epochs(raw, events=mne.events_from_annotations(raw)[0],
                                **config["epochs"], preload=True)
        if "rereference" in config:
            epochs = reref(epochs=epochs, **config["rereference"])
        if "ica" in config:
            epochs = apply_ICA(
                epochs=epochs, **config["ica"], reference=ica_ref)
        if "autoreject" in config:
            epochs = autoreject_epochs(epochs=epochs, **config["autoreject"])
    return epochs


def make_evokeds(epochs):
    evokeds = [epochs[condition].average() for condition in epochs.event_id.keys()]
    return evokeds

if __name__ == "__main__":  # workflow
    root_dir = pathlib.Path("D:/EEG")
    cfg = stp.load_file("config")
    mapping = stp.load_file("mapping")
    montage = stp.load_file("montage")
    header_files = stp.load_file("header", dir=root_dir)
    ica_ref = stp.load_file(type="ica", format=".fif")
    # r"" == raw string
    # \b matches on a change from a \w (a word character) to a \W (non word character)
    # \w{6} == six alphanumerical characters
    # RegEx expression to match subject ids (6 alphanumerical characters)
    pattern = r'\b\w{6}\b'
    ids = stp.get_ids(header_files=header_files, pattern=pattern)
    for id in ids[:1]:
        stp.make_folders(root_dir=root_dir, id=id)
        _fig_folder = pathlib.Path(f"D:/EEG/vocal_effort/data/{id}/figures")
        raw = make_raw(header_files, id, mapping=mapping, montage=montage)
        stp.save_object(raw, root_dir, id)
        epochs = run_pipeline(
            raw, config=cfg, ica_ref=ica_ref, fig_folder=_fig_folder)
        del raw  #  save working memory
        stp.save_object(epochs, root_dir, id)
        evokeds = make_evokeds(epochs)
        stp.save_object(evokeds, root_dir, id)
        del epochs, evokeds  # save working memory
