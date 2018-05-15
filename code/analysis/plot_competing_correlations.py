import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.stats import pearsonr
import pylab as pl
import sys
import json
import yaml
sys.path.append("code/striatal_model")
import params
import colors
from plot_tools2 import *
import matplotlib.pyplot as plt
import pandas
import seaborn as sbn
import matplotlib.collections as collections
import scalebars


grid_dimensions = [6, 6]
num_trials = 5
print len(sys.argv), sys.argv

all_spikes_fn = sys.argv[1:1 + num_trials]
all_channels_fn = sys.argv[1 + 1 * num_trials:1 + 2 * num_trials]
experiment_fn = sys.argv[1 + 2 * num_trials]
hemisphere = sys.argv[2 + 2 * num_trials]
traces_out_fn = sys.argv[3 + 2 * num_trials]
corr_out_fn = sys.argv[4 + 2 * num_trials]
corr_data_out = sys.argv[5 + 2 * num_trials]


df = pandas.DataFrame({"type": [], "channel": [], "CC": [], "trial": []})

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

    stim_times = get_stim_times(cfg, hemisphere, params, mask=True)[
        0].astype('int')

    chan_go_left = 21
    chan_go_right = 22

    exp_filter = np.exp(np.arange(0, 5, 0.001) / -0.3)

    spike_masks_go_left_d1 = get_spikes_mask(
        senders, times, channels[chan_go_left]['d1'])
    spike_masks_go_left_d2 = get_spikes_mask(
        senders, times, channels[chan_go_left]['d2'])

    spike_masks_go_right_d1 = get_spikes_mask(
        senders, times, channels[chan_go_right]['d1'])
    spike_masks_go_right_d2 = get_spikes_mask(
        senders, times, channels[chan_go_right]['d2'])

    print "mask done"

    filtered_spikes_go_left_d1 = filter_spikes(
        spike_masks_go_left_d1, exp_filter)
    filtered_spikes_go_left_d2 = filter_spikes(
        spike_masks_go_left_d2, exp_filter)
    filtered_spikes_go_right_d1 = filter_spikes(
        spike_masks_go_right_d1, exp_filter)
    filtered_spikes_go_right_d2 = filter_spikes(
        spike_masks_go_right_d2, exp_filter)

    filtered_spikes_go_left_d1 = np.mean(filtered_spikes_go_left_d1, axis=0)[
        np.where(stim_times)]
    filtered_spikes_go_left_d2 = np.mean(filtered_spikes_go_left_d2, axis=0)[
        np.where(stim_times)]
    filtered_spikes_go_right_d1 = np.mean(filtered_spikes_go_right_d1, axis=0)[
        np.where(stim_times)]
    filtered_spikes_go_right_d2 = np.mean(filtered_spikes_go_right_d2, axis=0)[
        np.where(stim_times)]

    filtered_spikes_go_left_d1 -= np.mean(filtered_spikes_go_left_d1)
    filtered_spikes_go_left_d2 -= np.mean(filtered_spikes_go_left_d2)
    filtered_spikes_go_right_d1 -= np.mean(filtered_spikes_go_right_d1)
    filtered_spikes_go_right_d2 -= np.mean(filtered_spikes_go_right_d2)

    print "filter done"

    df = df.append({"channel": 'go left', 'type': 'd1d2', 'trial': trial, 'CC': correlate2(
        filtered_spikes_go_left_d1, filtered_spikes_go_left_d2)[0, 1]}, ignore_index=True)

    df = df.append({"channel": 'go right', 'type': 'd1d2', 'trial': trial, 'CC': correlate2(
        filtered_spikes_go_right_d1, filtered_spikes_go_right_d2)[0, 1]}, ignore_index=True)

    df = df.append({"channel": 'between', 'type': 'd1d1', 'trial': trial, 'CC': correlate2(
        filtered_spikes_go_left_d1, filtered_spikes_go_right_d1)[0, 1]}, ignore_index=True)

    df = df.append({"channel": 'between', 'type': 'd2d2', 'trial': trial, 'CC': correlate2(
        filtered_spikes_go_left_d2, filtered_spikes_go_right_d2)[0, 1]}, ignore_index=True)


window_size = 250  # (ms)
step_size = 10    # (ms)

switching_times = np.arange(20000, 50000, step_size).astype('int')
switching = np.zeros(len(switching_times))

for i, t in enumerate(switching_times):
    window_go_left = filtered_spikes_go_left_d2[t -
                                                window_size / 2:t + window_size / 2]
    window_go_right = filtered_spikes_go_right_d2[t -
                                                  window_size / 2:t + window_size / 2]

    if all(window_go_left - window_go_right > 0.0):
        switching[i] = 1
    elif all(window_go_right - window_go_left > 0.0):
        switching[i] = -1


colors.seaborn.set_context('paper', font_scale=3.0,
                           rc={"lines.linewidth": 1.5})
colors.seaborn.set_style('whitegrid', {"axes.linewidth": 1.5})


lw = 1.5

fig = pl.figure(figsize=[16, 10])
fig.set_tight_layout(True)

ax0 = fig.add_subplot(1, 1, 1)

scalebars.add_scalebar(ax0, matchx=False, matchy=False, hidex=False,
                       hidey=False, size=3, label="3 Hz", horizontal=False)

x = np.arange(0, 30, 0.001 * step_size)
collection = collections.BrokenBarHCollection.span_where(
    x, ymin=26.5, ymax=29.5, where=switching > 0, facecolor=colors.colors[0], alpha=1.0)
ax0.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=26.5, ymax=29.5, where=switching < 0, facecolor=colors.colors[2], alpha=1.0)
ax0.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=26.5, ymax=29.5, where=switching == 0, facecolor=colors.colors[-1], alpha=1.0)
ax0.add_collection(collection)

ax0.plot(np.arange(0, 30, 0.001),
         filtered_spikes_go_left_d1[20000:50000] + 21, label="D1 go left")
ax0.plot(np.arange(0, 30, 0.001),
         filtered_spikes_go_left_d2[20000:50000] + 14, label="D2 go left")
ax0.plot(np.arange(0, 30, 0.001),
         filtered_spikes_go_right_d1[20000:50000] + 7, label="D1 go right")
ax0.plot(np.arange(0, 30, 0.001),
         filtered_spikes_go_right_d2[20000:50000] + 0, label="D2 go right")
ax0.set_xlabel("Time (s)", fontweight='bold')
pl.yticks([0, 7, 14, 21, 28], ["turn right D2", "turn right D1",
                               "turn left D2", "turn left D1", "winning channel"], rotation=40)
ax0.set_ylim([-4, 29.5])


fig.savefig(traces_out_fn)


fig = pl.figure(figsize=[16, 10])
fig.set_tight_layout(True)

ax2 = fig.add_subplot(1, 1, 1)
sbn.stripplot(x='channel', y='CC', hue='type',
              data=df, size=10., alpha=0.5, ax=ax2)
sbn.violinplot(x='channel', y='CC', hue='type',
               data=df, size=15., scale='width', ax=ax2)
ax2.legend_.remove()
ax2.set_xlabel("Channel", fontweight='bold')
ax2.set_ylabel("CC", fontweight='bold')
ax2.set_ylim([-1.2, 1.2])

fig.savefig(corr_out_fn)

df.to_json(corr_data_out)
