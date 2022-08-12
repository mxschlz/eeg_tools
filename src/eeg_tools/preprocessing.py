import settings
from mne.datasets import sample
import analysis
import setup_eeg_tools as set
import os
import pathlib
import numpy as np
import mne
from matplotlib import pyplot as plt, patches
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA
import json
import glob
import sys


path = os.getcwd() + "\\src\\" + "eeg_tools"
sys.path.append(path)

# TODO: describe workflow of processing data.
# TODO: go through all the functions and make them more elegant.


def run_pipeline(raw, fig_folder, config, ica_ref=None, exclude=None):
    """
    Automated preprocessing pipeline for raw EEG data.

    The pipeline takes an mne.io.Raw instance and preprocesses it according to
    the configuration parameters. Currently available preprocessing options are:
    filtering (highpass, lowpass, bandpass, notch);
    epoching;
    rereferencing (robust average, REST, single or linked electrodes);
    ocular artifact rejection (independent component analysis);
    automated threshold rejection (AutoReject).

    Args:
        raw (mne.io.Raw): raw instance containing continuous time series EEG data.
        fig_folder (string): folder path in which preprocessing steps are documented.
        config (dict): JSON file containing preprocessing parameters.
        ica_ref (mne.preprocessing.ica.ICA): ica reference template for artifact rejection via correlation mapping.
        exclude (int): exclude events by stimulus annotation. Defaults to None.

    Returns:
        epochs (mne.Epochs): preprocessed epoched EEG data.
    """
    global plot
    global _fig_folder
    _fig_folder = fig_folder
    if config == None:
        raise FileNotFoundError(
            "Need config file to preprocess data according to parameters!")
    elif fig_folder == None:
        plot = False
    else:
        if "filtering" in config:
            raw = filtering(data=raw, **config["filtering"])
        if "epochs" in config:
            events = mne.pick_events(events=mne.events_from_annotations(raw)[0],
                                     exclude=exclude)
            epochs = mne.Epochs(raw, events=events,
                                **config["epochs"], preload=True)
            epochs.plot(show=False, show_scalebars=False,
                        show_scrollbars=False, n_channels=20)
            plt.savefig(_fig_folder / pathlib.Path("epochs.jpg"), dpi=800)
            plt.close()
        if "rereference" in config:
            epochs = set_ref(epochs=epochs, **config["rereference"])
        if "ica" in config:
            epochs = apply_ICA(
                epochs=epochs, **config["ica"], reference=ica_ref)
        if "autoreject" in config:
            epochs = autoreject_epochs(epochs=epochs, **config["autoreject"])
    return epochs


def make_raw(header_files, id, fig_folder, mapping, montage, ref_ch="FCz",
             preload=True, add_ref_ch=True, plot=True):
    """
    Merges EEG files into an mne.io.Raw instance.

    Takes the BrainVision header files (format: .vhdr) and merges them subject-wise.
    Applies information about electrode positions and renames them according to the 10-20 system.
    Optionally adds a reference channel to the data which has zero voltage.

    Args:
        header_files (list of strings): header files of one subject (format: .vhdr).
        id (str): subject id.
        ref_ch (str): reference channel to be added.
        preload (bool): preload the data. Defaults to True.
        add_ref_ch (bool): add the reference channel. Defaults to True.
        mapping (dict): JSON file containing the electrode names.
        monateg (dict): JSON file containing information about sensor coordinates.

    Returns:
        raw (mne.io.Raw): continuous time series data which makes up the starting point for the preprocessing pipeline.
    """

    global _fig_folder
    _fig_folder = fig_folder
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
    if plot is True:
        raw.plot(show=False, show_scrollbars=False,
                 show_scalebars=False, start=2000.0, n_channels=20)
        plt.savefig(_fig_folder / pathlib.Path("raw.jpg"), dpi=800)
        plt.close()
    return raw


def filtering(data, notch=None, highpass=None, lowpass=None, plot=True):
    """
    Applies FIR (finite impulse response) filter to the data.

    Filter can be either highpass, lowpass, bandpass or a notch filter.
    Optionally, save the data plots before and after filtering.

    Args:
        data (mne.io.Raw, mne.Epochs, mne.Evoked): data to be filtered. Usually,
        continuous time series data is preferred to avoid disturbances by the filter design.
        notch (int): frequency to be filtered. Example: 50 to filter out power line noise.
        highpass (int / float): applies a highpass filter at given frequency.
        lowpass (int / float): applies a lowpass filter at given frequency.
        plot (bool): if True, saves the data as a plot before and after filtering.

    Returns:
        data: filtered mne instance.
    """

    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle("Power Spectral Density")
    ax[0].set_title("Before filtering")
    ax[1].set_title("After filtering")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    if plot == True:
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
            _fig_folder / pathlib.Path("lowpassfilter.jpg"), dpi=800)
    if highpass is not None and lowpass == None:
        fig.savefig(
            _fig_folder / pathlib.Path("highpassfilter.jpg"), dpi=800)
    if highpass and lowpass is not None:
        fig.savefig(
            _fig_folder / pathlib.Path("bandpassfilter.jpg"), dpi=800)
    if notch is not None:
        fig.savefig(
            _fig_folder / pathlib.Path("ZapLine_filter.jpg"), dpi=800)
    plt.close()
    return data


def set_ref(epochs, ransac_parameters=None, type="average", elecs=None, plot=True):
    """
    Rereferences the data.

    Applies a reference to the data according to the desired type.
    Reference options are:
    robust average over all electrodes, exclude outlier sensors beforehand (RANSAC),
    Reference Electrode Standardization Technique (REST),
    linked mastoids (TP9, TP10),
    or any other desired sensor.

    Args:
        epochs (mne.Epochs): mne.Epochs instance to be rereferenced.
        ransac_parameters (dict / None): parameters for the RANSAC algorithm. (https://autoreject.github.io/stable/generated/autoreject.Ransac.html#autoreject.Ransac)
                                         If None, use default parameters.
        type (str/None): reference type. Can be "average", "lm", "rest" or None.
                         If None, see elecs argument.
        elecs (str/list of str/None): If type == None, enter reference name as string.
                                      If reference should consist of more than one electrode, insert list of strings.
        plot (bool): if True, saves a figure of the data before and after rereferencing.

    Returns:
        epochs: rereferenced epochs.
    """

    if type == "average":
        epochs_clean = epochs.copy()
        ransac = Ransac(**ransac_parameters)  # optimize speed
        ransac.fit(epochs_clean)
        epochs_clean.average().plot(exclude=[])
        bads = input(
            "Enter bad sensors here (separate several bad sensors via spacebar): ").split()
        if len(bads) != 0 and bads not in ransac.bad_chs_:
            ransac.bad_chs_.extend(bads)
        epochs_clean = ransac.transform(epochs_clean)
        evoked = epochs.average()
        evoked_clean = epochs_clean.average()
        evoked.info['bads'] = ransac.bad_chs_
        evoked_clean.info['bads'] = ransac.bad_chs_
        if plot:
            fig, ax = plt.subplots(2)
            evoked.plot(exclude=[], axes=ax[0], show=False)
            evoked_clean.plot(exclude=[], axes=ax[1], show=False)
            ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
            ax[1].set_title("After RANSAC")
            fig.tight_layout()
            fig.savefig(
                _fig_folder / pathlib.Path("RANSAC_results.jpg"), dpi=800)
            plt.close()
        epochs_clean.set_eeg_reference(ref_channels="average", projection=True)
        epochs_clean.apply_proj()
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_clean)
        epochs_reref = epochs_clean.copy()
    elif type == "rest":
        sphere = mne.make_sphere_model("auto", "auto", epochs.info)
        src = mne.setup_volume_source_space(
            sphere=sphere, exclude=30., pos=5.)
        forward = mne.make_forward_solution(
            epochs.info, trans=None, src=src, bem=sphere)
        epochs_reref = epochs.copy().set_eeg_reference("REST", forward=forward)
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_reref)
    elif type == "lm":
        epochs_reref = epochs.copy().set_eeg_reference(["TP9", "TP10"])
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_reref)
    elif type == None:
        epochs_reref = epochs.copy().set_eeg_reference(elecs)
    if plot == True:
        fig, ax = plt.subplots(2)
        epochs.average().plot(axes=ax[0], show=False)
        epochs_reref.average().plot(axes=ax[1], show=False)
        ax[0].set_title(f"FCz, SNR={snr_pre:.2f}")
        if type is not None:
            ax[1].set_title(f"{type}, SNR={snr_post:.2f}")
            fig.tight_layout()
            fig.savefig(
                _fig_folder / pathlib.Path(f"{type}_reference.jpg"), dpi=800)
            plt.close()
        if type is None:
            ax[1].set_title(f"{elecs}, SNR={snr_post:.2f}")
            fig.tight_layout()
            fig.savefig(
                _fig_folder / pathlib.Path(f"{elecs}_reference.jpg"), dpi=800)
            plt.close()
    return epochs_reref


def apply_ICA(epochs, reference=None, n_components=None, method="fastica",
              threshold="auto", rejection="manual"):
    """
    Applies independent component analysis to the data.

    Uses a reference template and computes a correlation map for ocular artifact rejection.


    Args:
        epochs (mne.Epochs):

    Returns:
        type: description

    Raises:
        Exception: description

    """

    epochs_ica = epochs.copy()
    snr_pre_ica = analysis.snr(epochs_ica)
    # ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    # ar.fit(epochs_ica)
    # epochs_ar, reject_log = ar.transform(epochs_ica, return_log=True)
    if rejection == "automatic":
        ica = ICA(n_components=n_components, method=method)
        # ica.fit(epochs_ica[~reject_log.bad_epochs])
        ica.fit(epochs_ica)
        # reference ICA containing blink and saccade components.
        ref = reference
        # .labels_ dict must contain "blinks" key with int values.
        labels = list(ref.labels_.keys())
        components = list(ref.labels_.values())
        for component, label in zip(components, labels):
            mne.preprocessing.corrmap([ref, ica], template=(0, component[0]),
                                      label=label, plot=False, threshold=threshold)
            ica.apply(epochs_ica, exclude=ica.labels_["blinks"])  # apply ICA
        ica.plot_components(ica.labels_["blinks"], show=False)
        plt.savefig(_fig_folder / pathlib.Path("ICA_components.jpg"), dpi=800)
        plt.close()
        ica.plot_sources(inst=epochs, show=False, start=0,
                         stop=15, show_scrollbars=False)
        plt.savefig(_fig_folder / pathlib.Path(f"ICA_sources.jpg"), dpi=800)
        plt.close()
        snr_post_ica = analysis.snr(epochs_ica)
        ica.plot_overlay(epochs.average(), exclude=ica.labels_["blinks"],
                         show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
        plt.savefig(_fig_folder / pathlib.Path("ICA_results.jpg"), dpi=800)
        plt.close()
        return epochs_ica
    if rejection == "manual":
        ica = ICA(n_components=n_components, method=method)
        # ica.fit(epochs_ica[~reject_log.bad_epochs])
        ica.fit(epochs_ica)
        ica.plot_components(picks=[x for x in range(20)])
        ica.plot_sources(epochs_ica, start=0, stop=15, show_scrollbars=False, block=True)
        ica.exclude = list((input("Enter components to exclude here (separate several components via spacebar): ").split()))
        ica.exclude = [int(x) for x in ica.exclude]
        ica.apply(epochs_ica, exclude=ica.exclude)
        ica.plot_components(ica.exclude, show=False)
        plt.savefig(_fig_folder / pathlib.Path("ICA_components.jpg"), dpi=800)
        plt.close()
        ica.plot_sources(inst=epochs, show=False, start=0,
                         stop=15, show_scrollbars=False)
        plt.savefig(_fig_folder / pathlib.Path(f"ICA_sources.jpg"), dpi=800)
        plt.close()
        snr_post_ica = analysis.snr(epochs_ica)
        ica.plot_overlay(epochs.average(), exclude=ica.exclude,
                         show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
        plt.savefig(_fig_folder / pathlib.Path("ICA_results.jpg"), dpi=800)
        plt.close()
        return epochs_ica


def autoreject_epochs(epochs,
                      n_interpolate=[1, 4, 32],
                      consensus=None,
                      cv=10,
                      thresh_method="bayesian optimization",
                      n_jobs=-1,
                      random_state=None):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description
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
    fig.savefig(_fig_folder / pathlib.Path("autoreject_best_fit.jpg"), dpi=800)
    plt.close()
    evoked_bad = epochs[reject_log.bad_epochs].average()
    snr_ar = analysis.snr(epochs_ar)
    plt.plot(evoked_bad.times, evoked_bad.data.T * 1e06, 'r', zorder=-1)
    epochs_ar.average().plot(axes=plt.gca(), show=False,
                             titles=f"SNR: {snr_ar:.2f}")
    plt.savefig(
        _fig_folder / pathlib.Path("autoreject_results.jpg"), dpi=800)
    plt.close()
    epochs_ar.plot_drop_log(show=False)
    plt.savefig(
        _fig_folder / pathlib.Path("epochs_drop_log.jpg"), dpi=800)
    plt.close()
    return epochs_ar


def make_evokeds(epochs, plot=True, baseline=None):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description
    """

    if baseline is not None:
        epochs.apply_baseline(baseline)
    # evokeds = [epochs[condition].average()
    #           for condition in epochs.event_id.keys()]
    evokeds = epochs.average(by_event_type=True)
    if plot is True and _fig_folder is not None:
        snr = analysis.snr(epochs)
        avrgd = mne.grand_average(evokeds)
        avrgd.plot_joint(show=False, title=f"SNR: {snr:.2f}")
        plt.savefig(_fig_folder / pathlib.Path("evokeds.jpg", dpi=800))
        plt.close()
    return evokeds

if "__name__" == "__main__":
    epochs = set.read_object("epochs", settings.root_dir, settings.ids[5])
    ica = ICA()
    # ica.fit(epochs_ica[~reject_log.bad_epochs])
    ica.fit(epochs)
    ica.plot_components()
    ica.plot_sources(epochs)
    ica.exclude = list((input("Enter components to exclude here (separate several components via spacebar): ").split()))
    ica.exclude = [int(x) for x in ica.exclude]
    ica.apply(epochs, exclude=ica.exclude)
    epochs.average().plot()
