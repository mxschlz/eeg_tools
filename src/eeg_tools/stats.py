import mne
import scipy.stats
from mne.stats import spatio_temporal_cluster_test
import numpy as np
import matplotlib.pyplot as plt


def target_test(evokeds, time_windows, electrodes, conditions, rel=True, parametric=False):
    index = "time"
    report = "{electrode}, time: {tmin}-{tmax} s; stat={statistic:.3f}, p={p}"
    print("Targeted t-test results:")
    for tmin, tmax in time_windows:
        cond0 = mne.grand_average(evokeds[conditions[0]]).copy().crop(tmin, tmax).to_data_frame(index=index)
        cond1 = mne.grand_average(evokeds[conditions[1]]).copy().crop(tmin, tmax).to_data_frame(index=index)
        for electrode in electrodes:
            # extract data
            A = cond0[electrode]
            B = cond1[electrode]
            # conduct t test
            if rel and not parametric:
                s, p = scipy.stats.wilcoxon(A, B)
            elif rel and parametric:
                s, p = scipy.stats.ttest_rel(A, B)
            elif not rel and not parametric:
                s, p = scipy.stats.mannwhitneyu(A, B)
            elif not rel and parametric:
                s, p = scipy.stats.ttest_ind(A, B)
            else:
                print("Desired test not found. Aborting ... ")
            # display results
            format_dict = dict(electrode=electrode, tmin=tmin, tmax=tmax, statistic=s, p=p)
            print(report.format(**format_dict))
