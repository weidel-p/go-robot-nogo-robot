import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.stats import pearsonr
import pylab as pl
import sys
import json
import yaml
sys.path.append("code/two_hemisphere_model")
import params
from colors import colors
from plot_tools2 import *
import matplotlib.pyplot as plt

grid_dimensions = [6, 6]

spikes_fn = sys.argv[1]
channels_fn = sys.argv[2]
experiment_fn = sys.argv[3]
hemisphere = sys.argv[4]
corr_out_with_stim_fn = sys.argv[5]
corr_out_bw_stim_fn = sys.argv[6]
corr_out_with_bckgrnd_fn = sys.argv[7]
corr_out_bw_bckgrnd_fn = sys.argv[8]

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

stim_times, bckgrnd_times = get_stim_times(cfg, hemisphere, params, mask=True)


if hemisphere == 'left':
    # place of stimulation [3, 3] # Jyotika:Actually it is turnleft in right hemipshere and turn right in left hemipshere, definitely not the middle channel. changing this
    center_channel = 22
if hemisphere == 'right':
    # place of stimulation [3, 3] # Jyotika:Actually it is turnleft in right hemipshere and turn right in left hemipshere, definitely not the middle channel. changing this
    center_channel = 21

exp_filter = np.exp(np.arange(0, 5, 0.001) / -0.3)

spike_masks_center_d1 = get_spikes_mask(
    senders, times, channels[center_channel]['d1'])
filtered_spikes_center_d1 = filter_spikes(spike_masks_center_d1, exp_filter)

filtered_spikes_center_d1_stim = np.array(filtered_spikes_center_d1)[
    :, np.where(stim_times == 1)[0]]
filtered_spikes_center_d1_bckgrnd = np.array(filtered_spikes_center_d1)[
    :, np.where(bckgrnd_times == 1)[0]]


spike_masks_center_d2 = get_spikes_mask(
    senders, times, channels[center_channel]['d2'])
filtered_spikes_center_d2 = filter_spikes(spike_masks_center_d2, exp_filter)

filtered_spikes_center_d2_stim = np.array(filtered_spikes_center_d2)[
    :, np.where(stim_times == 1)[0]]
filtered_spikes_center_d2_bckgrnd = np.array(filtered_spikes_center_d2)[
    :, np.where(bckgrnd_times == 1)[0]]


fig_within_stim = pl.figure("with_stim", figsize=[16, 10])
fig_between_stim = pl.figure("bw_stim", figsize=[16, 10])
fig_within_bckgrnd = pl.figure("with_bckgrnd", figsize=[16, 10])
fig_between_bckgrnd = pl.figure("bw_bckgrnd", figsize=[16, 10])
lw = 1.5
for chan_id, channel in enumerate(channels):

    spike_masks_d1 = get_spikes_mask(senders, times, channel['d1'])
    filtered_spikes_d1 = filter_spikes(spike_masks_d1, exp_filter)

    filtered_spikes_d1_stim = np.array(filtered_spikes_d1)[
        :, np.where(stim_times == 1)[0]]
    filtered_spikes_d1_bckgrnd = np.array(filtered_spikes_d1)[
        :, np.where(bckgrnd_times == 1)[0]]

    spike_masks_d2 = get_spikes_mask(senders, times, channel['d2'])
    filtered_spikes_d2 = filter_spikes(spike_masks_d2, exp_filter)

    filtered_spikes_d2_stim = np.array(filtered_spikes_d2)[
        :, np.where(stim_times == 1)[0]]
    filtered_spikes_d2_bckgrnd = np.array(filtered_spikes_d2)[
        :, np.where(bckgrnd_times == 1)[0]]

    # Between center channel and other channels
    cc_d1_bw_stim = correlate2(
        filtered_spikes_center_d1_stim, filtered_spikes_d1_stim, 2)
    cc_d2_bw_stim = correlate2(
        filtered_spikes_center_d2_stim, filtered_spikes_d2_stim, 2)
    cc_d1_d2_bw_stim = correlate2(
        filtered_spikes_center_d1_stim, filtered_spikes_d2_stim, 2)

    cc_d1_bw_bckgrnd = correlate2(
        filtered_spikes_center_d1_bckgrnd, filtered_spikes_d1_bckgrnd, 2)
    cc_d2_bw_bckgrnd = correlate2(
        filtered_spikes_center_d2_bckgrnd, filtered_spikes_d2_bckgrnd, 2)
    cc_d1_d2_bw_bckgrnd = correlate2(
        filtered_spikes_center_d1_bckgrnd, filtered_spikes_d2_bckgrnd, 2)

    # Within each channels
    cc_d1_with_stim = correlate2(
        filtered_spikes_d1_stim, filtered_spikes_d1_stim, 2)
    cc_d2_with_stim = correlate2(
        filtered_spikes_d2_stim, filtered_spikes_d2_stim, 2)
    cc_d1_d2_with_stim = correlate2(
        filtered_spikes_d1_stim, filtered_spikes_d2_stim, 2)

    cc_d1_with_bckgrnd = correlate2(
        filtered_spikes_d1_bckgrnd, filtered_spikes_d1_bckgrnd, 4)
    cc_d2_with_bckgrnd = correlate2(
        filtered_spikes_d2_bckgrnd, filtered_spikes_d2_bckgrnd, 4)
    cc_d1_d2_with_bckgrnd = correlate2(
        filtered_spikes_d1_bckgrnd, filtered_spikes_d2_bckgrnd, 2)

    # plotting
    ax_with_stim = fig_within_stim.add_subplot(
        grid_dimensions[0], grid_dimensions[1], chan_id + 1)
    ax_bw_stim = fig_between_stim.add_subplot(
        grid_dimensions[0], grid_dimensions[1], chan_id + 1)
    ax_with_bckgrnd = fig_within_bckgrnd.add_subplot(
        grid_dimensions[0], grid_dimensions[1], chan_id + 1)
    ax_bw_bckgrnd = fig_between_bckgrnd.add_subplot(
        grid_dimensions[0], grid_dimensions[1], chan_id + 1)

    if chan_id == center_channel:
        ax_with_stim.patch.set_facecolor(colors[-1])
        ax_bw_stim.patch.set_facecolor(colors[-1])
        ax_with_bckgrnd.patch.set_facecolor(colors[-1])
        ax_bw_bckgrnd.patch.set_facecolor(colors[-1])

    hist, edges = np.histogram(
        cc_d1_with_stim, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_with_stim.plot(edges[:-1], hist, color=colors[1],
                      label="D1-D1", linewidth=lw)
    hist, edges = np.histogram(
        cc_d2_with_stim, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_with_stim.plot(edges[:-1], hist, color=colors[2],
                      label="D2-D2", linewidth=lw)
    hist, edges = np.histogram(
        cc_d1_d2_with_stim, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_with_stim.plot(edges[:-1], hist, color=colors[3],
                      label="D1-D2", linewidth=lw)

    ax_with_stim.set_xlim(-0.4, 1.0)
    ax_with_stim.set_ylim(0, 6.)
    ax_with_stim.vlines(0, 0, 8, linestyles='dashed')
    ax_with_stim.legend()

    hist, edges = np.histogram(
        cc_d1_bw_stim, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_bw_stim.plot(edges[:-1], hist, color=colors[1],
                    label="D1-D1", linewidth=lw)
    hist, edges = np.histogram(
        cc_d2_bw_stim, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_bw_stim.plot(edges[:-1], hist, color=colors[2],
                    label="D2-D2", linewidth=lw)
    hist, edges = np.histogram(
        cc_d1_d2_bw_stim, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_bw_stim.plot(edges[:-1], hist, color=colors[3],
                    label="D1-D2", linewidth=lw)

    ax_bw_stim.set_xlim(-0.6, 0.6)
    ax_bw_stim.set_ylim(0, 6.)
    ax_bw_stim.vlines(0, 0, 8, linestyles='dashed')
    ax_bw_stim.legend()

    hist, edges = np.histogram(
        cc_d1_with_bckgrnd, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_with_bckgrnd.plot(
        edges[:-1], hist, color=colors[1], label="D1-D1", linewidth=lw)
    hist, edges = np.histogram(
        cc_d2_with_bckgrnd, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_with_bckgrnd.plot(
        edges[:-1], hist, color=colors[2], label="D2-D2", linewidth=lw)
    hist, edges = np.histogram(
        cc_d1_d2_with_bckgrnd, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_with_bckgrnd.plot(
        edges[:-1], hist, color=colors[3], label="D1-D2", linewidth=lw)

    ax_with_bckgrnd.set_xlim(-0.4, 1.0)
    ax_with_bckgrnd.set_ylim(0, 6.)
    ax_with_bckgrnd.vlines(0, 0, 8, linestyles='dashed')
    ax_with_bckgrnd.legend()

    hist, edges = np.histogram(
        cc_d1_bw_bckgrnd, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_bw_bckgrnd.plot(edges[:-1], hist, color=colors[1],
                       label="D1-D1", linewidth=lw)
    hist, edges = np.histogram(
        cc_d2_bw_bckgrnd, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_bw_bckgrnd.plot(edges[:-1], hist, color=colors[2],
                       label="D2-D2", linewidth=lw)
    hist, edges = np.histogram(
        cc_d1_d2_bw_bckgrnd, bins=np.linspace(-0.6, 1.1, 15), normed=True)
    ax_bw_bckgrnd.plot(edges[:-1], hist, color=colors[3],
                       label="D1-D2", linewidth=lw)
    ax_bw_bckgrnd.set_xlim(-0.6, 0.6)
    ax_bw_bckgrnd.set_ylim(0, 6.)
    ax_bw_bckgrnd.vlines(0, 0, 8, linestyles='dashed')
    ax_bw_bckgrnd.legend()

    if chan_id % 6 != 0 and chan_id != center_channel:
        ax_with_stim.tick_params(labelleft='off')
        ax_bw_stim.tick_params(labelleft='off')
        ax_with_bckgrnd.tick_params(labelleft='off')
        ax_bw_bckgrnd.tick_params(labelleft='off')
    else:
        if chan_id == center_channel:
            ax_with_stim.set_xlim(-0.6, 1.0)
            ax_bw_stim.set_xlim(-0.6, 1.0)
            ax_with_bckgrnd.set_xlim(-0.6, 1.0)
            ax_bw_bckgrnd.set_xlim(-0.6, 1.0)
            ax_with_stim.tick_params(labelleft='on')
            ax_bw_stim.tick_params(labelleft='on')
            ax_with_bckgrnd.tick_params(labelleft='on')
            ax_bw_bckgrnd.tick_params(labelleft='on')
        plt.setp(ax_with_stim.get_yticklabels()[1::2], visible=False)
        plt.setp(ax_with_stim.get_yticklabels()[
                 ::2], visible=True, fontweight='bold', fontsize=12)

        plt.setp(ax_bw_stim.get_yticklabels()[1::2], visible=False)
        plt.setp(ax_bw_stim.get_yticklabels()[
                 ::2], visible=True, fontweight='bold', fontsize=12)

        plt.setp(ax_with_bckgrnd.get_yticklabels()[1::2], visible=False)
        plt.setp(ax_with_bckgrnd.get_yticklabels()[
                 ::2], visible=True, fontweight='bold', fontsize=12)

        plt.setp(ax_bw_bckgrnd.get_yticklabels()[1::2], visible=False)
        plt.setp(ax_bw_bckgrnd.get_yticklabels()[
                 ::2], visible=True, fontweight='bold', fontsize=12)

    if chan_id < 30 and chan_id != center_channel:
        ax_with_stim.tick_params(labelbottom='off')
        ax_bw_stim.tick_params(labelbottom='off')
        ax_with_bckgrnd.tick_params(labelbottom='off')
        ax_bw_bckgrnd.tick_params(labelbottom='off')
    else:
        if chan_id == center_channel:
            ax_with_stim.set_xlim(-0.6, 1.0)
            ax_bw_stim.set_xlim(-0.6, 1.0)
            ax_with_bckgrnd.set_xlim(-0.6, 1.0)
            ax_bw_bckgrnd.set_xlim(-0.6, 1.0)
            ax_with_stim.tick_params(labelleft='on')
            ax_bw_stim.tick_params(labelleft='on')
            ax_with_bckgrnd.tick_params(labelleft='on')
            ax_bw_bckgrnd.tick_params(labelleft='on')
        plt.setp(ax_with_stim.get_xticklabels()[1::2], visible=False)
        plt.setp(ax_with_stim.get_xticklabels()[
                 ::2], visible=True, fontweight='bold', fontsize=12)

        plt.setp(ax_bw_stim.get_xticklabels()[1::2], visible=False)
        plt.setp(ax_bw_stim.get_xticklabels()[
                 ::2], visible=True, fontweight='bold', fontsize=12)

        plt.setp(ax_with_bckgrnd.get_xticklabels()[1::2], visible=False)
        plt.setp(ax_with_bckgrnd.get_xticklabels()[
                 ::2], visible=True, fontweight='bold', fontsize=12)

        plt.setp(ax_bw_bckgrnd.get_xticklabels()[1::2], visible=False)
        plt.setp(ax_bw_bckgrnd.get_xticklabels()[
                 ::2], visible=True, fontweight='bold', fontsize=12)

fig_within_stim.subplots_adjust(wspace=0.1, hspace=0.15)
fig_within_stim.savefig(corr_out_with_stim_fn)

fig_within_bckgrnd.subplots_adjust(wspace=0.1, hspace=0.15)
fig_within_bckgrnd.savefig(corr_out_with_bckgrnd_fn)

fig_between_stim.subplots_adjust(wspace=0.1, hspace=0.15)
fig_between_stim.savefig(corr_out_bw_stim_fn)

fig_between_bckgrnd.subplots_adjust(wspace=0.1, hspace=0.15)
fig_between_bckgrnd.savefig(corr_out_bw_bckgrnd_fn)
