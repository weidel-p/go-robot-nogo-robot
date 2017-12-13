import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.stats import alpha
from scipy.stats import pearsonr
import pylab as pl
import seaborn
import sys
import json
import yaml
sys.path.append("code/striatal_model")
import params
from colors import colors
from plot_tools2 import *
import copy
import matplotlib.patches as patches
import seaborn as sbn

seaborn.set_context('paper', font_scale=3.0, rc={"lines.linewidth": 2.5})
seaborn.set_style('whitegrid', {"axes.linewidth": 2.5})



num_trials = 5
print len(sys.argv), sys.argv

all_spikes_fn = sys.argv[1:1 + num_trials]
all_channels_fn = sys.argv[1 + 1 * num_trials:1 + 2 * num_trials]
experiment_fn = sys.argv[1 + 2 * num_trials]
hemisphere = sys.argv[2 + 2 * num_trials]
out_fn = sys.argv[3 + 2 * num_trials]


all_ccs_short = []
all_ccs_long = []
all_ccs_shuffled_short = []
all_ccs_shuffled_long = []
all_ccs_shuffled_independent_short = []
all_ccs_shuffled_independent_long = []

for trial in range(num_trials):
    print "trial", trial

    spikes_fn = all_spikes_fn[trial]
    channels_fn = all_channels_fn[trial]

    # spike data of the channels
    data = np.loadtxt(spikes_fn)
    senders = data[:, 0]
    unique_senders = np.unique(senders)  # all active senders
    times = data[:, 1]

    with open(channels_fn, "r+") as f:
        channels = json.load(f)
        channels = channels['channels']

    with open(experiment_fn, "r+") as f:
        cfg = yaml.load(f)

    stim_times_start, stim_times_stop = get_stim_times(cfg, hemisphere, params, mask=False, scale=1)

    all_d1 = np.ravel([c['d1'] for c in channels])
    all_d2 = np.ravel([c['d2'] for c in channels])

    spikes_d1 = np.hstack([times[np.where(senders == nid)[0]] for nid in all_d1])
    spikes_d2 = np.hstack([times[np.where(senders == nid)[0]] for nid in all_d2])

    stepsize = 100.  # ms
    window_size_short = int(0.3 * (stim_times_stop[0] - stim_times_start[0]))
    window_size_long = int(1.2 * (stim_times_stop[0] - stim_times_start[0]))

    spikes_mask_d1 = get_spikes_mask(senders, times, all_d1, scale=1)
    spikes_mask_d2 = get_spikes_mask(senders, times, all_d2, scale=1)
    print("mask done")

    exp_filter = np.exp(np.arange(0, 5, 0.001) / -0.3)
    filtered_all_d1 = filter_spikes(spikes_mask_d1, exp_filter, scale=1)
    filtered_all_d2 = filter_spikes(spikes_mask_d2, exp_filter, scale=1)
    print("filter done")

    hist_all_d1 = np.mean(filtered_all_d1, axis=0)[1000:]
    hist_all_d2 = np.mean(filtered_all_d2, axis=0)[1000:]

    hist_all_d1_shuffled_independent = copy.deepcopy(hist_all_d1)
    hist_all_d2_shuffled_independent = copy.deepcopy(hist_all_d2)

    np.random.shuffle(hist_all_d1_shuffled_independent)
    np.random.shuffle(hist_all_d2_shuffled_independent)

    r = np.random.permutation(range(len(hist_all_d1)))

    hist_all_d1_shuffled = hist_all_d1[r]
    hist_all_d2_shuffled = hist_all_d2[r]

    ccs_short = []
    ccs_long = []
    ccs_shuffled_short = []
    ccs_shuffled_long = []
    ccs_shuffled_independent_short = []
    ccs_shuffled_independent_long = []

    for t in np.arange(0, len(hist_all_d1) - window_size_short, stepsize):
        t = int(t)

        hist_d1 = hist_all_d1[t:t + window_size_short]
        hist_d2 = hist_all_d2[t:t + window_size_short]
        ccs_short.append(correlate2(hist_d1, hist_d2)[0, 1])

        hist_d1_shuffled = hist_all_d1_shuffled[t:t + window_size_short]
        hist_d2_shuffled = hist_all_d2_shuffled[t:t + window_size_short]
        ccs_shuffled_short.append(correlate2(hist_d1_shuffled, hist_d2_shuffled)[0, 1])

        hist_d1_shuffled_independent = hist_all_d1_shuffled_independent[t:t + window_size_short]
        hist_d2_shuffled_independent = hist_all_d2_shuffled_independent[t:t + window_size_short]
        ccs_shuffled_independent_short.append(correlate2(
            hist_d1_shuffled_independent, hist_d2_shuffled_independent)[0, 1])

    for t in np.arange(0, len(hist_all_d1) - window_size_long, stepsize):
        t = int(t)

        hist_d1 = hist_all_d1[t:t + window_size_long]
        hist_d2 = hist_all_d2[t:t + window_size_long]
        ccs_long.append(correlate2(hist_d1, hist_d2)[0, 1])

        hist_d1_shuffled = hist_all_d1_shuffled[t:t + window_size_long]
        hist_d2_shuffled = hist_all_d2_shuffled[t:t + window_size_long]
        ccs_shuffled_long.append(correlate2(hist_d1_shuffled, hist_d2_shuffled)[0, 1])

        hist_d1_shuffled_independent = hist_all_d1_shuffled_independent[t:t + window_size_long]
        hist_d2_shuffled_independent = hist_all_d2_shuffled_independent[t:t + window_size_long]
        ccs_shuffled_independent_long.append(correlate2(
            hist_d1_shuffled_independent, hist_d2_shuffled_independent)[0, 1])

    all_ccs_short.append(ccs_short)
    all_ccs_long.append(ccs_long)
    all_ccs_shuffled_short.append(ccs_shuffled_short)
    all_ccs_shuffled_long.append(ccs_shuffled_long)
    all_ccs_shuffled_independent_short.append(ccs_shuffled_independent_short)
    all_ccs_shuffled_independent_long.append(ccs_shuffled_independent_long)


time_short = np.arange(window_size_short / 2., len(hist_all_d1) -
                       window_size_short / 2., stepsize) / 1000.
time_long = np.arange(window_size_long / 2., len(hist_all_d1) -
                      window_size_long / 2., stepsize) / 1000.


fig = pl.figure(0, figsize=[16, 10])

ax = fig.add_subplot(2, 1, 1)

ax.plot(np.arange(len(hist_all_d1)) / 1000., hist_all_d1, label='D1', color=colors[0])
ax.plot(np.arange(len(hist_all_d2)) / 1000., hist_all_d2, label='D2', color=colors[1])
rateMax = np.max([np.max(hist_all_d1), np.max(hist_all_d2)])
rateMin = np.min([np.min(hist_all_d1), np.min(hist_all_d2)])
histMax = np.max([np.max(hist_all_d1), np.max(hist_all_d2)])
ax.set_ylabel("Mean activity (spks/s)", fontweight='bold')

ax.add_patch(patches.Rectangle(
    (1., rateMin - 0.01), 1 + window_size_long / 1000., (rateMax - rateMin) * 1.10, edgecolor=colors[3], linewidth=3.5, facecolor='none'

))
ax.add_patch(patches.Rectangle(
    (1., rateMin - 0.01), 1 + window_size_short / 1000., rateMax - rateMin, edgecolor=colors[2], linewidth=3.5, facecolor='none'

))
ax.set_ylim(min(0, rateMin), rateMax * 1.1)

for x in ax.get_xticklabels():
    x.set_fontweight('bold')

ax.set_xticklabels([])



ax2 = fig.add_subplot(2, 1, 2)

sbn.tsplot(all_ccs_short, time=time_short, color=colors[2], ax=ax2, linewidth=2.5, marker='o')
sbn.tsplot(all_ccs_shuffled_independent_short, time=time_short, color=colors[4], ax=ax2, linewidth=2.5, marker='o')
sbn.tsplot(all_ccs_long, time=time_long, color=colors[3], ax=ax2, linewidth=2.5, marker='o')
sbn.tsplot(all_ccs_shuffled_independent_long, time=time_long, color=colors[4], ax=ax2, linewidth=2.5, marker='o')

ax2.set_xlim([0, int(params.runtime) / 1000.])
ax2.hlines(0, 0, int(params.runtime) / 1000., colors='k', linestyle="dashed")
ax2.set_ylabel("CC", fontweight='bold')

y_max = max(np.ravel([ccs_short, ccs_shuffled_independent_short]))
y_min = min(np.ravel([ccs_short, ccs_shuffled_independent_short]))
ax2.set_ylim(-1, 1.0)
for x in ax2.get_xticklabels():
    x.set_fontweight('bold')

for x in ax2.get_yticklabels():
    x.set_fontweight('bold')

ax2.grid('off')
ax2.set_xlabel("Time (s)", fontweight='bold')

pl.savefig(out_fn)
