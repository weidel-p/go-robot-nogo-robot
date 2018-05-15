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

left_spikes_fn = sys.argv[1]
right_spikes_fn = sys.argv[2]
left_channels_fn = sys.argv[3]
right_channels_fn = sys.argv[4]
experiment_fn = sys.argv[5]
traces_out_fn = sys.argv[6]



def get_filtered_spikes(spikes_fn, channels_fn, hemisphere):

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

    stim_times = get_stim_times(cfg, hemisphere, params, mask=True, scale=1)[
        0].astype('int')

    chan_go_left = 21
    chan_go_right = 22

    exp_filter = np.exp(np.arange(0, 5, 0.001) / -0.3)

    spike_masks_go_left_d1 = get_spikes_mask(
        senders, times, channels[chan_go_left]['d1'], scale=1)
    spike_masks_go_left_d2 = get_spikes_mask(
        senders, times, channels[chan_go_left]['d2'], scale=1)

    spike_masks_go_right_d1 = get_spikes_mask(
        senders, times, channels[chan_go_right]['d1'], scale=1)
    spike_masks_go_right_d2 = get_spikes_mask(
        senders, times, channels[chan_go_right]['d2'], scale=1)

    print "mask done"

    filtered_spikes_go_left_d1 = filter_spikes(
        spike_masks_go_left_d1, exp_filter, scale=1)
    filtered_spikes_go_left_d2 = filter_spikes(
        spike_masks_go_left_d2, exp_filter, scale=1)
    filtered_spikes_go_right_d1 = filter_spikes(
        spike_masks_go_right_d1, exp_filter, scale=1)
    filtered_spikes_go_right_d2 = filter_spikes(
        spike_masks_go_right_d2, exp_filter, scale=1)

    filtered_spikes_go_left_d1 = np.mean(filtered_spikes_go_left_d1, axis=0)
    filtered_spikes_go_left_d2 = np.mean(filtered_spikes_go_left_d2, axis=0)
    filtered_spikes_go_right_d1 = np.mean(filtered_spikes_go_right_d1, axis=0)
    filtered_spikes_go_right_d2 = np.mean(filtered_spikes_go_right_d2, axis=0)

    filtered_spikes_go_left_d1 -= np.mean(filtered_spikes_go_left_d1)
    filtered_spikes_go_left_d2 -= np.mean(filtered_spikes_go_left_d2)
    filtered_spikes_go_right_d1 -= np.mean(filtered_spikes_go_right_d1)
    filtered_spikes_go_right_d2 -= np.mean(filtered_spikes_go_right_d2)

    print "filter done"

    return {'go_left_d1': filtered_spikes_go_left_d1, 'go_left_d2': filtered_spikes_go_left_d2, 'go_right_d1': filtered_spikes_go_right_d1, 'go_right_d2': filtered_spikes_go_right_d2}

left_hemi_data = get_filtered_spikes(left_spikes_fn, left_channels_fn, 'left')
right_hemi_data = get_filtered_spikes(right_spikes_fn, right_channels_fn, 'right')

window_size = 250  # (ms)
step_size = 1    # (ms)

switching_times = np.arange(0, int(params.runtime), step_size).astype('int')
left_switching = np.zeros(len(switching_times))
right_switching = np.zeros(len(switching_times))



for i, t in enumerate(switching_times):
    if t < 2000 or t > 18000:
        # plot winning channels only during stimulation times 
        left_switching[i] = 0
        right_switching[i] = 0
        continue 

    window_go_left = left_hemi_data['go_left_d2'][t -window_size / 2:t + window_size / 2]
    window_go_right = left_hemi_data['go_right_d2'][t -window_size / 2:t + window_size / 2]

    if all(window_go_left - window_go_right > 0.0):
        left_switching[i] = 1
    elif all(window_go_right - window_go_left > 0.0):
        left_switching[i] = -1

    window_go_left = right_hemi_data['go_left_d2'][t -window_size / 2:t + window_size / 2]
    window_go_right = right_hemi_data['go_right_d2'][t -window_size / 2:t + window_size / 2]

    if all(window_go_left - window_go_right > 0.0):
        right_switching[i] = 1
    elif all(window_go_right - window_go_left > 0.0):
        right_switching[i] = -1


switching = left_switching + right_switching 
switching[np.where(switching == -1)] = 0
switching[np.where(switching == 1)] = 0


colors.seaborn.set_context('paper', font_scale=3.0,
                           rc={"lines.linewidth": 1.5})
colors.seaborn.set_style('whitegrid', {"axes.linewidth": 1.5})


lw = 1.5

fig = pl.figure(figsize=[16, 10])
fig.set_tight_layout(True)

ax0 = fig.add_subplot(1, 1, 1)

scalebars.add_scalebar(ax0, matchx=False, matchy=False, hidex=False,
                       hidey=False, size=3, label="3 Hz", horizontal=False)

x = np.arange(0, 20, 0.001 * step_size)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=78.5, ymax=81.5, where=switching > 1, facecolor=colors.colors[5], alpha=0.8)
ax0.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=78.5, ymax=81.5, where=switching < -1, facecolor=colors.colors[3], alpha=0.8)
ax0.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=78.5, ymax=81.5, where=switching == 0, facecolor=colors.colors[4], alpha=0.8)
ax0.add_collection(collection)


collection = collections.BrokenBarHCollection.span_where(
    x, ymin=31.5, ymax=33.5, where=left_switching > 0, facecolor=colors.colors[5], alpha=0.8)
ax0.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=31.5, ymax=33.5, where=left_switching < 0, facecolor=colors.colors[3], alpha=0.8)
ax0.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=31.5, ymax=33.5, where=left_switching == 0, facecolor=colors.colors[4], alpha=0.8)
ax0.add_collection(collection)


collection = collections.BrokenBarHCollection.span_where(
    x, ymin=73.5, ymax=75.5, where=right_switching > 0, facecolor=colors.colors[5], alpha=0.8)
ax0.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=73.5, ymax=75.5, where=right_switching < 0, facecolor=colors.colors[3], alpha=0.8)
ax0.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=73.5, ymax=75.5, where=right_switching == 0, facecolor=colors.colors[4], alpha=0.8)
ax0.add_collection(collection)



ax0.plot(np.arange(0, 20, 0.001), left_hemi_data['go_left_d1'] + 27, label="left hemi D1 go left", color=colors.colors[5])
ax0.plot(np.arange(0, 20, 0.001), left_hemi_data['go_left_d2'] + 18, label="left hemi D2 go left", color=colors.colors[5])
ax0.plot(np.arange(0, 20, 0.001), left_hemi_data['go_right_d1'] + 9, label="left hemi D1 go right", color=colors.colors[3])
ax0.plot(np.arange(0, 20, 0.001), left_hemi_data['go_right_d2'] + 0, label="left hemi D2 go right", color=colors.colors[3])

ax0.plot(np.arange(0, 20, 0.001), right_hemi_data['go_left_d1'] + 27 + 43, label="right hemi D1 go left", color=colors.colors[5])
ax0.plot(np.arange(0, 20, 0.001), right_hemi_data['go_left_d2'] + 18 + 43, label="right hemi D2 go left", color=colors.colors[5])
ax0.plot(np.arange(0, 20, 0.001), right_hemi_data['go_right_d1'] + 9 + 43, label="right hemi D1 go right", color=colors.colors[3])
ax0.plot(np.arange(0, 20, 0.001), right_hemi_data['go_right_d2'] + 0 + 43, label="right hemi D2 go right", color=colors.colors[3])

ax0.set_xlabel("Time (s)", fontweight='bold')

pl.yticks([-4, 5, 14, 23, 28, 39, 48, 57, 66, 71, 77], ["turn right D2", "turn right D1", "turn left D2", "turn left D1", "winning channel", "turn right D2", "turn right D1", "turn left D2", "turn left D1", "winning channel", "action selected"], rotation=40)

pl.text(-3.7, 17, "Left Hemisphere", rotation=90, fontsize=20, fontweight='bold')
pl.text(-3.7, 59, "Right Hemisphere", rotation=90, fontsize=20, fontweight='bold')

ax0.set_ylim([-8, 82.5])


fig.savefig(traces_out_fn)


