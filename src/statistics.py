import mne
from mne.stats import spatio_temporal_cluster_test
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("D:/Projects/eeg_tools/src/")
import setup_eeg_tools as set


def cluster_permuttest(data, n_permutations=1000, alpha=0.05, start=0.2, step=0.2, n_jobs=-1):
    return significant_points

if "__name__" == "__main__":
    DIR = pathlib.Path("D:/EEG")
    header_files = set.find(path=DIR, mode="pattern", pattern="*.vhdr")
    ids = set.get_ids(header_files=header_files)
    cfg = set.load_file("config", DIR)
    evokeds, evokeds_avrgd, evokeds_data = cfg["epochs"]["event_id"].copy(
    ), cfg["epochs"]["event_id"].copy(), cfg["epochs"]["event_id"].copy()
    for key in cfg["epochs"]["event_id"]:
        evokeds[key], evokeds_avrgd[key], evokeds_data[key] = list(), list(), list()
        # get evokeds for every condition and subject.
    for id in ids:
        evoked = set.read_object("evokeds", DIR, id)
        for condition in evoked:
            if condition.comment in evokeds:
                evokeds[condition.comment].append(condition)
                if len(evokeds[condition.comment]) == len(ids):
                    evokeds_avrgd[condition.comment] = mne.grand_average(
                        evokeds[condition.comment])
                else:
                    continue
    for condition in evokeds:
        evokeds_data[condition] = np.array(
            [evokeds[condition][e].get_data() for e in range(len(ids))])
        evokeds_data[condition] = evokeds_data[condition].transpose(0, 2, 1)
    evoked_diff = mne.combine_evoked(
        [evokeds_avrgd["deviant"], evokeds_avrgd["vocal_effort/5"]], weights=[1, -1])
    evoked_diff.plot_image()  # plot difference
    evoked_diff.plot_joint()
    adjacency, _ = mne.channels.find_ch_adjacency(
        evokeds_avrgd["deviant"].info, None)
    X = [evokeds_data["deviant"], evokeds_data["vocal_effort/5"]]
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
        X, threshold=dict(start=.2, step=.2), adjacency=adjacency, n_permutations=1000, n_jobs=-1)
    significant_points = cluster_pv.reshape(t_obs.shape).T < .05
    evoked_diff.plot_image(mask=significant_points,
                           show_names="all", titles=None)
