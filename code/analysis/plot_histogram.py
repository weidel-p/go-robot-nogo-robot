import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
import matplotlib.cm as cm
import os
import pickle
import pdb
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), '..', 'two_hemisphere_model/'))
import params as p
import json
import matplotlib.pyplot as pl
from itertools import combinations
import itertools
import yaml
from pylab import *
import colors


colors.seaborn.set_context('paper', font_scale=3.0,
                           rc={"lines.linewidth": 1.5})
colors.seaborn.set_style('whitegrid', {"axes.linewidth": 1.5})

# def channelHistogram(fn, dirname, hemis):       # Also calculate the within and between channel signal-noise-ratio

experiment = sys.argv[1]
hemis = sys.argv[2]
trial = 0
fn = "data/{}/{}/{}_hemisphere.gdf".format(experiment, trial, hemis)
dirname = "data/{}/{}/".format(experiment, trial)
fn_out = sys.argv[3]

spike_data = np.loadtxt(fn)
all_senders = spike_data[:, 0]
all_spike_times = spike_data[:, 1]

if hemis == 'left':
    with open(dirname + "neuron_ids_left_hemisphere.json", "r+") as f:
        hemisphere_neuron_ids = json.load(f)
else:
    with open(dirname + "neuron_ids_right_hemisphere.json", "r+") as f:
        hemisphere_neuron_ids = json.load(f)
# This is to separate the times of activity with and without external input
if experiment == 'sequencesMultTrials.yaml' or experiment == 'sequencesMultTrialsd2.yaml':
    with open("code/two_hemisphere_model/experiments/{}".format(experiment), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    for i, c in enumerate(cfg['stim-params']):
        if c['hemisphere'] == hemis:
            start_times = c['start_times']
            stop_times = c['stop_times']
            d1Actip = []
            d1Actwoip = []
            d2Actip = []
            d2Actwoip = []
hists = []
binsize = 200.
binning = np.arange(0, p.runtime, binsize)

channel_id = 0
all_d1_spikes = np.array([])
all_d2_spikes = np.array([])

fig = pl.figure(figsize=[16, 10])

signalD1 = []  # D1 activity of channel receiving the input
signalD2 = []  # D2 activity of channel receiving the input
noiseD1 = []    # D1 activity of all other channels
noiseD2 = []     # D2 activity of all other channels
neighborD1 = []  # D1 activity of immediately neighbouring channels
neighborD2 = []  # D2 activity of immediately neighbouring channels

subFigsHands = []
maxYVals = []
for i, channel in enumerate(hemisphere_neuron_ids['channels']):

    # Making grid plot instead of all rows plot
    ax = fig.add_subplot(p.grid_size[0][0] + 1, p.grid_size[0][1], i + 1)

    subFigsHands.append(ax)
    binsize = 200.          # Different bin size for individual channel histograms, to see better
    binning = np.arange(0, p.runtime, binsize)

    channel_n_ids = channel['d1']
    channel_spike_data = np.array([])

    # get spikes for this channel
    mask = np.hstack([np.where(n_id == all_senders)[0]
                      for n_id in channel_n_ids])
    channel_spike_senders = all_senders[mask]
    channel_spike_times = all_spike_times[mask]

    all_d1_spikes = np.append(all_d1_spikes, channel_spike_times)

    hist = np.histogram(channel_spike_times, bins=binning)
    binsizeInSecs = binsize / 1000.
    if channel['row'] == 3 and channel['col'] == 3:
        ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs), color=colors.colors[0],
                linewidth=1.0, alpha=1.0)  # Normalize by channel size to get rate per neuron
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['top'].set_edgecolor('black')
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['left'].set_edgecolor('black')
        ax.spines['right'].set_linewidth(2.5)
        ax.spines['right'].set_edgecolor('black')
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['bottom'].set_edgecolor('black')

        if hemis == 'right':
            signalD1.append(
                hist[0] / float(len(channel_n_ids) * binsizeInSecs))
            if experiment == 'sequencesMultTrials.yaml' or experiment == 'sequencesMultTrialsd2.yaml':
                currStart = 500.
                for start, stop in zip(start_times, stop_times):
                    currStop = start - 500.
                    rated1ip = len(channel_spike_times[np.logical_and(channel_spike_times >= start,
                                                                      channel_spike_times <= stop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                    rated1woip = len(channel_spike_times[np.logical_and(channel_spike_times >= currStart,
                                                                        channel_spike_times <= currStop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                    d1Actip.append(rated1ip)
                    d1Actwoip.append(rated1woip)
                    currStart = stop + 500.
    elif channel['row'] == 3 and channel['col'] == 4:
        ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs),
                linewidth=1.0, color=colors.colors[0], alpha=1.0)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['top'].set_edgecolor('orange')
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['left'].set_edgecolor('orange')
        ax.spines['right'].set_linewidth(2.5)
        ax.spines['right'].set_edgecolor('orange')
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['bottom'].set_edgecolor('orange')

        if hemis == 'left':
            signalD1.append(
                hist[0] / float(len(channel_n_ids) * binsizeInSecs))
            if experiment == 'sequencesMultTrials.yaml' or experiment == 'sequencesMultTrialsd2.yaml':
                for start, stop in zip(start_times, stop_times):
                    currStart = stop + 500.
                    currStop = stop + 2000. - 500.  # dirty manual pause time
                    rated1ip = len(channel_spike_times[np.logical_and(channel_spike_times >= start,
                                                                      channel_spike_times <= stop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                    rated1woip = len(channel_spike_times[np.logical_and(channel_spike_times >= currStart,
                                                                        channel_spike_times <= currStop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                    d1Actip.append(rated1ip)
                    d1Actwoip.append(rated1woip)

    else:
        ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs),
                linewidth=1.0, color=colors.colors[0], alpha=1.0)
        if hemis == 'right' and (channel['row'] == 2 or channel['col'] == 2):
            neighborD1.append(
                hist[0] / float(len(channel_n_ids) * binsizeInSecs))
        elif hemis == 'left' and (channel['row'] == 2 or channel['col'] == 3):
            neighborD1.append(
                hist[0] / float(len(channel_n_ids) * binsizeInSecs))
        else:
            noiseD1.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))

    d1max = np.max(hist[0] / float(len(channel_n_ids) * binsizeInSecs))

    channel_n_ids = channel['d2']
    channel_spike_data = np.array([])

    # get spikes for this channel
    mask = np.hstack([np.where(n_id == all_senders)[0]
                      for n_id in channel_n_ids])
    channel_spike_senders = all_senders[mask]
    channel_spike_times = all_spike_times[mask]

    all_d2_spikes = np.append(all_d2_spikes, channel_spike_times)

    hist = np.histogram(channel_spike_times, bins=binning)
    if channel['row'] == 3 and channel['col'] == 4:
        ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs),
                linewidth=1.0, color=colors.colors[1], alpha=1.0)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['top'].set_edgecolor('orange')
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['left'].set_edgecolor('orange')
        ax.spines['right'].set_linewidth(2.5)
        ax.spines['right'].set_edgecolor('orange')
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['bottom'].set_edgecolor('orange')

        if hemis == 'left':
            signalD2.append(
                hist[0] / float(len(channel_n_ids) * binsizeInSecs))
            if experiment == 'sequencesMultTrials.yaml' or experiment == 'sequencesMultTrialsd2.yaml':
                for start, stop in zip(start_times, stop_times):
                    currStart = stop + 500.
                    currStop = stop + 2000. - 500.  # dirty manual pause time
                    rated2ip = len(channel_spike_times[np.logical_and(channel_spike_times >= start,
                                                                      channel_spike_times <= stop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                    rated2woip = len(channel_spike_times[np.logical_and(channel_spike_times >= currStart,
                                                                        channel_spike_times <= currStop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                    d2Actip.append(rated2ip)
                    d2Actwoip.append(rated2woip)

    elif channel['row'] == 3 and channel['col'] == 3:
        ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs),
                linewidth=1.0, color=colors.colors[1], alpha=1.0)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['top'].set_edgecolor('black')
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['left'].set_edgecolor('black')
        ax.spines['right'].set_linewidth(2.5)
        ax.spines['right'].set_edgecolor('black')
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['bottom'].set_edgecolor('black')

        if hemis == 'right':
            signalD2.append(
                hist[0] / float(len(channel_n_ids) * binsizeInSecs))
            if experiment == 'sequencesMultTrials.yaml' or experiment == 'sequencesMultTrialsd2.yaml':
                currStart = 500.
                for start, stop in zip(start_times, stop_times):
                    currStop = start - 500.
                    rated2ip = len(channel_spike_times[np.logical_and(channel_spike_times >= start,
                                                                      channel_spike_times <= stop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                    rated2woip = len(channel_spike_times[np.logical_and(channel_spike_times >= currStart,
                                                                        channel_spike_times <= currStop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                    d2Actip.append(rated2ip)
                    d2Actwoip.append(rated2woip)
                    currStart = stop + 500.

    else:
        ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs),
                linewidth=1.0, color=colors.colors[1], alpha=1.0)
        if hemis == 'right' and (channel['row'] == 2 or channel['col'] == 2):
            neighborD2.append(
                hist[0] / float(len(channel_n_ids) * binsizeInSecs))
        elif hemis == 'left' and (channel['row'] == 2 or channel['col'] == 3):
            neighborD2.append(
                hist[0] / float(len(channel_n_ids) * binsizeInSecs))
        else:
            noiseD2.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))

    d2max = np.max(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
    maxYVals.append(np.max([d1max, d2max]))
    # Ideally thsi should be for all sequences, since stimulation times is still set to old values, even if simulation times are increased to 200 seconds
    if experiment == 'sequencesd1d2.yaml' or experiment == 'competingActions.yaml':
        ax.set_xlim(0, 20000.)
    channel_id += 1
    for x in ax.get_xticklabels():
        x.set_visible(False)
    ax.grid(False)
    if channel['col'] == 0:
        for x in ax.get_yticklabels()[1::2]:
            x.set_visible(False)
    else:
        for x in ax.get_yticklabels():
            x.set_visible(False)

    for x in ax.get_yticklabels():
        x.set_fontweight('bold')

# Set the same y lim for all subplots
for ax in subFigsHands:
    ax.set_ylim(0, np.max(maxYVals))


binsize = 200.
binning = np.arange(0, p.runtime, binsize)

ax = plt.subplot2grid((p.grid_size[0][0] + 1, p.grid_size[0][1]),
                      (p.grid_size[0][0], 0), colspan=p.grid_size[0][1])
hist = np.histogram(all_d1_spikes, bins=binning)
ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * p.num_channels * binsizeInSecs),
        label="all_D1", color=colors.colors[0])  # Assuming all channels are equal size
ax.hlines(y=np.mean(hist[0] / float(len(channel_n_ids) * p.num_channels * binsizeInSecs)),
          xmin=np.min(hist[1][:-1]), xmax=np.max(hist[1][:-1]), linestyles='dashed', colors=colors.colors[0], linewidth=1.5)
hist = np.histogram(all_d2_spikes, bins=binning)
ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * p.num_channels *
                                      binsizeInSecs), label="all_D2", color=colors.colors[1])
ax.hlines(y=np.mean(hist[0] / float(len(channel_n_ids) * p.num_channels * binsizeInSecs)),
          xmin=np.min(hist[1][:-1]), xmax=np.max(hist[1][:-1]), linestyles='dashed', colors=colors.colors[1], linewidth=1.5)
ax.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
for x in ax.get_yticklabels()[1::2]:
    x.set_visible(False)
for x in ax.get_yticklabels():
    x.set_fontweight('bold')
for x in ax.get_xticklabels():
    x.set_fontweight('bold')
ax.grid(False)
# Ideally thsi should be for all sequences, since stimulation times is still set to old values, even if simulation times are increased to 200 seconds
if experiment == 'sequencesd1d2.yaml' or experiment == 'competingActions.yaml':
    ax.set_xlim(0, 20000.)

fhand = file(dirname + "allD1Spikes.gdf", 'w')
np.savetxt(fhand, all_d1_spikes)
fhand.close()

fhand = file(dirname + "allD2Spikes.gdf", 'w')
np.savetxt(fhand, all_d2_spikes)
fhand.close()

fig.savefig(fn_out)
