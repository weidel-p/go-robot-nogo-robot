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
sys.path.append("code/two_hemisphere_model")
import params
from colors import colors
from plot_tools2 import *


spikes_fn = sys.argv[1]
channels_fn = sys.argv[2]
experiment_fn = sys.argv[3]
hemisphere = sys.argv[4]
out_fn = sys.argv[5]

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

stim_times_start, stim_times_stop = get_stim_times(
    cfg, hemisphere, params, mask=False)

all_d1 = np.ravel([c['d1'] for c in channels])
all_d2 = np.ravel([c['d2'] for c in channels])


spikes_d1 = np.hstack([times[np.where(senders == nid)[0]] for nid in all_d1])
spikes_d2 = np.hstack([times[np.where(senders == nid)[0]] for nid in all_d2])


spikes_d1_stim = np.array([])
spikes_d2_stim = np.array([])
spikes_d1_bckgrnd = np.array([])
spikes_d2_bckgrnd = np.array([])

for i, t in enumerate(spikes_d1):
    for boundaries in zip(stim_times_start, stim_times_stop):
        if t >= boundaries[0] + transient_duration and t < boundaries[1]:
            spikes_d1_stim = np.append(spikes_d1_stim, t)


for i, t in enumerate(spikes_d2):
    for boundaries in zip(stim_times_start, stim_times_stop):
        if t >= boundaries[0] + transient_duration and t < boundaries[1]:
            spikes_d2_stim = np.append(spikes_d2_stim, t)


for i, t in enumerate(spikes_d1):
    for boundaries in zip(stim_times_start, stim_times_stop):
        if t < boundaries[0] or t >= boundaries[1] + transient_duration:
            spikes_d1_bckgrnd = np.append(spikes_d1_bckgrnd, t)


for i, t in enumerate(spikes_d2):
    for boundaries in zip(stim_times_start, stim_times_stop):
        if t < boundaries[0] or t >= boundaries[1] + transient_duration:
            spikes_d2_bckgrnd = np.append(spikes_d2_bckgrnd, t)


cc_all = []
cc_stim = []
cc_stim_var = []
cc_stim_shuff = []
cc_stim_shuff_var = []
cc_bckgrnd = []
cc_bckgrnd_var = []
cc_bckgrnd_shuff = []
cc_bckgrnd_shuff_var = []

binsizes = np.linspace(10, 600, 30)

for binsize in binsizes:

    # runtime for correlations is 10 times longer
    bins = np.arange(0, int(params.runtime) * 10, binsize)
    stim_bins = np.array([])
    bckgrnd_bins = np.array([])

    for t in bins:
        for boundaries in zip(stim_times_start, stim_times_stop):
            if t >= boundaries[0] and t < boundaries[1]:
                stim_bins = np.append(stim_bins, t)

    bckgrnd_bins = np.array([b for b in bins if not b in stim_bins])

    hist_d1 = np.histogram(spikes_d1, bins=bins)[0].astype(
        'float') * 1000. / (binsize * len(all_d1))
    hist_d2 = np.histogram(spikes_d2, bins=bins)[0].astype(
        'float') * 1000. / (binsize * len(all_d2))

    hist_d1_stim = np.histogram(spikes_d1_stim, bins=stim_bins)[
        0].astype('float') * 1000. / (binsize * len(all_d1))
    hist_d2_stim = np.histogram(spikes_d2_stim, bins=stim_bins)[
        0].astype('float') * 1000. / (binsize * len(all_d2))

    hist_d1_bckgrnd = np.histogram(spikes_d1_bckgrnd, bins=bckgrnd_bins)[
        0].astype('float') * 1000. / (binsize * len(all_d1))
    hist_d2_bckgrnd = np.histogram(spikes_d2_bckgrnd, bins=bckgrnd_bins)[
        0].astype('float') * 1000. / (binsize * len(all_d2))

    # split the histograms into 3 parts, to have 3 data points for corrcoef
    hist_d1_bckgrnd_split = np.array_split(hist_d1_bckgrnd, 3)
    hist_d2_bckgrnd_split = np.array_split(hist_d2_bckgrnd, 3)
    if experiment_fn == 'sequences.yaml' or experiment_fn == 'sequencesd1d2.yaml' or experiment_fn == 'competingActions.yaml':
        split_num = 3
    else:
        split_num = 3   # Too short sequences for sequences<x>.yaml

    hist_d1_stim_split = np.array_split(hist_d1_stim, split_num)
    hist_d2_stim_split = np.array_split(hist_d2_stim, split_num)

    cc_bck_split = []
    for x, y in zip(hist_d1_bckgrnd_split, hist_d2_bckgrnd_split):
        cc_bck_split.append(correlate2(x, y)[1, 0])

    cc_stim_split = []
    for x, y in zip(hist_d1_stim_split, hist_d2_stim_split):
        cc_stim_split.append(correlate2(x, y)[1, 0])

    cc_all.append(correlate2(hist_d1, hist_d2)[1, 0])
    cc_stim.append(np.mean(cc_stim_split))
    cc_stim_var.append(np.std(cc_stim_split))
    cc_bckgrnd.append(np.mean(cc_bck_split))
    cc_bckgrnd_var.append(np.std(cc_bck_split))

    hist_d1_stim_shuf = np.copy(hist_d1_stim)
    hist_d2_stim_shuf = np.copy(hist_d2_stim)
    hist_d1_bckgrnd_shuf = np.copy(hist_d1_bckgrnd)
    hist_d2_bckgrnd_shuf = np.copy(hist_d2_bckgrnd)

    temp_stim = []
    temp_bckgrnd = []
    for x in xrange(10):  # 10 trials
        np.random.shuffle(hist_d1_stim_shuf)
        np.random.shuffle(hist_d2_stim_shuf)
        np.random.shuffle(hist_d1_bckgrnd_shuf)
        np.random.shuffle(hist_d2_bckgrnd_shuf)
        temp_stim.append(correlate2(
            hist_d1_stim_shuf, hist_d2_stim_shuf)[1, 0])
        temp_bckgrnd.append(correlate2(
            hist_d1_bckgrnd_shuf, hist_d2_bckgrnd_shuf)[1, 0])
    cc_stim_shuff.append(np.mean(temp_stim))
    cc_stim_shuff_var.append(np.std(temp_stim))

    cc_bckgrnd_shuff.append(np.mean(temp_bckgrnd))
    cc_bckgrnd_shuff_var.append(np.std(temp_bckgrnd))


fig = pl.figure(0, figsize=[16, 10])

ax = fig.add_subplot(2, 1, 1)
ax.plot(np.arange(len(hist_d1)) / (1000. / binsize), hist_d1, label='D1')
ax.plot(np.arange(len(hist_d2)) / (1000. / binsize), hist_d2, label='D2')
ax.set_ylabel("Mean activity (spikes/sec)", fontsize=20, fontweight='bold')
ax.set_xlabel("Time (s)", fontsize=20, fontweight='bold')
ax.legend(prop={'size': 15, 'weight': 'bold'}, loc='best')
for x in ax.get_xticklabels():
    x.set_fontweight('bold')
    x.set_fontsize(14)
for x in ax.get_yticklabels():
    x.set_fontweight('bold')
    x.set_fontsize(14)


ax = fig.add_subplot(2, 1, 2)
ax.plot(binsizes, cc_stim, '.-', label='stimulation',
        color=colors[1], markersize=20.)
ax.fill_between(binsizes, np.array(cc_stim) - np.array(cc_stim_var),
                np.array(cc_stim) + np.array(cc_stim_var), color=colors[1], alpha=0.2)

ax.plot(binsizes, cc_stim_shuff, '.--', label='stimulation-shuffled',
        color=colors[2], markersize=20.)
ax.fill_between(binsizes, np.array(cc_stim_shuff) - np.array(cc_stim_shuff_var),
                np.array(cc_stim_shuff) + np.array(cc_stim_shuff_var), color=colors[2], alpha=0.2)

ax.plot(binsizes, cc_bckgrnd, '.-', label='background',
        color=colors[3], markersize=20.)
ax.fill_between(binsizes, np.array(cc_bckgrnd) - np.array(cc_bckgrnd_var),
                np.array(cc_bckgrnd) + np.array(cc_bckgrnd_var), color=colors[3], alpha=0.2)

ax.plot(binsizes, cc_bckgrnd_shuff, '.--',
        label='background-shuffled', color=colors[4], markersize=20.)
ax.fill_between(binsizes, np.array(cc_bckgrnd_shuff) - np.array(cc_bckgrnd_shuff_var),
                np.array(cc_bckgrnd_shuff) + np.array(cc_bckgrnd_shuff_var), color=colors[4], alpha=0.2)

ax.set_xlabel("Bin size (ms)", fontsize=20, fontweight='bold')
ax.set_ylabel("Average cross correlation", fontsize=20, fontweight='bold')
ax.legend(prop={'size': 15, 'weight': 'bold'}, loc='best')
for x in ax.get_xticklabels():
    x.set_fontweight('bold')
    x.set_fontsize(14)
for x in ax.get_yticklabels():
    x.set_fontweight('bold')
    x.set_fontsize(14)


pl.savefig(out_fn)
