import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.stats import pearsonr
import matplotlib.cm as cm
import pylab as pl
import sys
import json
import yaml
sys.path.append("code/striatal_model")
import params
from colors import colors
from plot_tools2 import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


grid_dimensions = [6, 6]

spikes_fn = sys.argv[1]
channels_fn = sys.argv[2]
experiment_fn = sys.argv[3]
hemisphere = sys.argv[4]
corr_out_with_stim_fn = sys.argv[5]
corr_out_with_bckgrnd_fn = sys.argv[6]

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

spike_masks_center_d1 = get_spikes_mask(senders, times, channels[center_channel]['d1'])
filtered_spikes_center_d1 = filter_spikes(spike_masks_center_d1, exp_filter)

filtered_spikes_center_d1_stim = np.array(filtered_spikes_center_d1)[:, np.where(stim_times == 1)[0]]
filtered_spikes_center_d1_bckgrnd = np.array(filtered_spikes_center_d1)[:, np.where(bckgrnd_times == 1)[0]]


spike_masks_center_d2 = get_spikes_mask(senders, times, channels[center_channel]['d2'])
filtered_spikes_center_d2 = filter_spikes(spike_masks_center_d2, exp_filter)

filtered_spikes_center_d2_stim = np.array(filtered_spikes_center_d2)[:, np.where(stim_times == 1)[0]]
filtered_spikes_center_d2_bckgrnd = np.array(filtered_spikes_center_d2)[:, np.where(bckgrnd_times == 1)[0]]
plt.rcParams["axes.axisbelow"] = False

fig_stim = pl.figure("stim", figsize=[16, 5])
fig_bckgrnd = pl.figure("bckgrnd", figsize=[16, 5])
lw = 0.1


filtered_spikes_d1_stim = []
filtered_spikes_d1_bckgrnd = []
filtered_spikes_d2_stim = []
filtered_spikes_d2_bckgrnd = []
step = 5  # considers every nth neuron
for chan_id, channel in enumerate(channels):

    spike_masks_d1 = get_spikes_mask(senders, times, channel['d1'][::step])
    filtered_spikes_d1 = filter_spikes(spike_masks_d1, exp_filter)

    temp1 = np.array(filtered_spikes_d1)[:, np.where(stim_times == 1)[0]]
    for x in temp1:
        filtered_spikes_d1_stim.append(x)
    temp2 = np.array(filtered_spikes_d1)[:, np.where(bckgrnd_times == 1)[0]]
    for x in temp2:
        filtered_spikes_d1_bckgrnd.append(x)

    spike_masks_d2 = get_spikes_mask(senders, times, channel['d2'][::step])
    filtered_spikes_d2 = filter_spikes(spike_masks_d2, exp_filter)
    temp3 = np.array(filtered_spikes_d2)[:, np.where(stim_times == 1)[0]]
    for x in temp3:
        filtered_spikes_d2_stim.append(x)
    temp4 = np.array(filtered_spikes_d2)[:, np.where(bckgrnd_times == 1)[0]]
    for x in temp4:
        filtered_spikes_d2_bckgrnd.append(x)


#  each channels
cc_d1_stim = correlate2(filtered_spikes_d1_stim, filtered_spikes_d1_stim, 2)    #
cc_d2_stim = correlate2(filtered_spikes_d2_stim, filtered_spikes_d2_stim, 2)
cc_d1_d2_stim = correlate2(filtered_spikes_d1_stim, filtered_spikes_d2_stim, 2)

cc_d1_bckgrnd = correlate2(filtered_spikes_d1_bckgrnd, filtered_spikes_d1_bckgrnd, 2)
cc_d2_bckgrnd = correlate2(filtered_spikes_d2_bckgrnd, filtered_spikes_d2_bckgrnd, 2)
cc_d1_d2_bckgrnd = correlate2(filtered_spikes_d1_bckgrnd, filtered_spikes_d2_bckgrnd, 2)

vmin = -0.2
vmax = 0.2
gridLen = (params.num_neurons_per_channel * params.num_channels) / step
gridStep = params.num_neurons_per_channel / step
minor_ticks = np.arange(0, gridLen + gridStep, gridStep)  # ticks for grids
# plotting
ax_stim11 = fig_stim.add_subplot(1, 3, 1, aspect='equal')  # D1-D1
ax_stim22 = fig_stim.add_subplot(1, 3, 2, aspect='equal')  # D2-D2
ax_stim12 = fig_stim.add_subplot(1, 3, 3, aspect='equal')  # D1-D2

cenR, cenC = get_grid_pos_fromChanNum(center_channel)
nearNeigh = get_NearNeighbors(cenR, cenC)
nearNeighChan = [get_ChanNum_gridPos(x[0], x[1]) for x in nearNeigh]
farNeigh = get_FarNeigbours(cenR, cenC)
farNeighChan = [get_ChanNum_gridPos(x[0], x[1]) for x in farNeigh]

cmap = cm.RdBu_r

if 'sequences.yaml' in experiment_fn or 'sequencesd1d2.yaml' in experiment_fn or 'competingActions.yaml' in experiment_fn or 'competingActionsNoD2Conn.yaml' in experiment_fn:
    minLim = vmin
    maxLim = vmax * 2.
    cmap1 = shiftedColorMap(cmap, min_val=minLim, max_val=maxLim, name="shifted")

else:
    minLim = vmin
    maxLim = vmax * 4.
    cmap1 = shiftedColorMap(cmap, min_val=minLim, max_val=maxLim, name="shifted")


ax_stim11.pcolormesh(cc_d1_stim[:len(cc_d1_stim) / 2, len(cc_d1_stim) / 2:], vmin=minLim, vmax=maxLim, cmap=cmap1)

ax_stim11.set_xlabel("D1-MSNs", fontsize=15, fontweight='bold')
ax_stim11.set_ylabel("D1-MSNs", fontsize=15, fontweight='bold')
ax_stim11.set_xticks(minor_ticks, minor=True)
ax_stim11.set_yticks(minor_ticks, minor=True)
ax_stim11.xaxis.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw,
                     alpha=0.5)
ax_stim11.yaxis.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw, alpha=0.5)
ax_stim11.set_xlim(100, len(cc_d1_stim) / 2)
ax_stim11.set_ylim(100, len(cc_d1_stim) / 2)

# near Neighbors of the near neighbors of the stimulated channel
nearNearNeigh = []
for chan in nearNeighChan:
    cenR1, cenC1 = get_grid_pos_fromChanNum(chan)
    nearGrid = get_NearNeighbors(cenR1, cenC1)
    nearChans = [get_ChanNum_gridPos(x[0], x[1]) for x in nearGrid]
    nearNearNeigh.append(nearChans)


# Mark nearby channels
for chan in nearNeighChan:
    ax_stim11.add_patch(patches.Rectangle((chan * 8, center_channel * 8), gridStep,
                                          gridStep, fill=False, edgecolor='fuchsia', linewidth=lw * 25, alpha=1.0))

# Mark the stimulated channel
ax_stim11.add_patch(patches.Rectangle((center_channel * 8, center_channel * 8), gridStep,
                                      gridStep, fill=False, edgecolor='black', linewidth=lw * 25, alpha=1.0))


ax_stim22.pcolormesh(cc_d2_stim[:len(cc_d2_stim) / 2, len(cc_d2_stim) / 2:], vmin=minLim, vmax=maxLim, cmap=cmap1)

ax_stim22.set_xlabel("D2-MSNs", fontsize=15, fontweight='bold')
ax_stim22.set_ylabel("D2-MSNs", fontsize=15, fontweight='bold')
ax_stim22.set_xticks(minor_ticks, minor=True)
ax_stim22.set_yticks(minor_ticks, minor=True)
ax_stim22.xaxis.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw, alpha=0.5)
ax_stim22.yaxis.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw, alpha=0.5)
ax_stim22.set_xlim(100, len(cc_d2_stim) / 2)
ax_stim22.set_ylim(100, len(cc_d2_stim) / 2)

# Mark nearby channels
for chan in nearNeighChan:
    ax_stim22.add_patch(patches.Rectangle((chan * 8, center_channel * 8), gridStep,
                                          gridStep, fill=False, edgecolor='fuchsia', linewidth=lw * 25, alpha=1.0))

# Mark the stimulated channel
ax_stim22.add_patch(patches.Rectangle((center_channel * 8, center_channel * 8), gridStep,
                                      gridStep, fill=False, edgecolor='black', linewidth=lw * 25, alpha=1.0))

_pcolor12_stim = ax_stim12.pcolormesh(
    cc_d1_d2_stim[:len(cc_d1_d2_stim) / 2, len(cc_d1_d2_stim) / 2:], vmin=minLim, vmax=maxLim, cmap=cmap1)
ax_stim12.set_xlabel("D2-MSNs", fontsize=15, fontweight='bold')
ax_stim12.set_ylabel("D1-MSNs", fontsize=15, fontweight='bold')
ax_stim12.set_xticks(minor_ticks, minor=True)
ax_stim12.set_yticks(minor_ticks, minor=True)
ax_stim12.xaxis.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw, alpha=0.5)
ax_stim12.yaxis.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw, alpha=0.5)
ax_stim12.set_xlim(100, len(cc_d1_d2_stim) / 2)
ax_stim12.set_ylim(100, len(cc_d1_d2_stim) / 2)

# Mark nearby channels
for chan in nearNeighChan:
    ax_stim12.add_patch(patches.Rectangle((chan * 8, center_channel * 8), gridStep,
                                          gridStep, fill=False, edgecolor='fuchsia', linewidth=lw * 25, alpha=1.0))

# Mark the stimulated channel
ax_stim12.add_patch(patches.Rectangle((center_channel * 8, center_channel * 8), gridStep,
                                      gridStep, fill=False, edgecolor='black', linewidth=lw * 25, alpha=1.0))


cbar_ax2 = fig_stim.add_axes([0.935, 0.25, 0.02, 0.5])
fig_stim.colorbar(_pcolor12_stim, cax=cbar_ax2)


plt.rcParams["axes.axisbelow"] = False

ax_bckgrnd11 = fig_bckgrnd.add_subplot(1, 3, 1, aspect='equal')
ax_bckgrnd22 = fig_bckgrnd.add_subplot(1, 3, 2, aspect='equal')
ax_bckgrnd12 = fig_bckgrnd.add_subplot(1, 3, 3, aspect='equal')

ax_bckgrnd11.pcolormesh(cc_d1_bckgrnd[:len(cc_d1_bckgrnd) / 2,
                                      len(cc_d1_bckgrnd) / 2:], vmin=vmin, vmax=vmax, cmap=cmap)
ax_bckgrnd11.set_xlabel("D1-MSNs", fontsize=15, fontweight='bold')
ax_bckgrnd11.set_ylabel("D1-MSNs", fontsize=15, fontweight='bold')
ax_bckgrnd11.set_xticks(minor_ticks, minor=True)
ax_bckgrnd11.set_yticks(minor_ticks, minor=True)

ax_bckgrnd11.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw, alpha=0.5)
ax_bckgrnd11.set_xlim(100, len(cc_d1_bckgrnd) / 2)
ax_bckgrnd11.set_ylim(100, len(cc_d1_bckgrnd) / 2)
ax_bckgrnd11.add_patch(patches.Rectangle((center_channel * 8, center_channel * 8), gridStep,
                                         gridStep, fill=False, edgecolor='black', linewidth=lw * 25, alpha=1.0))

# Mark nearby channels
for chan in nearNeighChan:
    ax_bckgrnd11.add_patch(patches.Rectangle((chan * 8, center_channel * 8), gridStep,
                                             gridStep, fill=False, edgecolor='fuchsia', linewidth=lw * 25, alpha=1.0))

# Mark the stimulated channel
ax_bckgrnd11.add_patch(patches.Rectangle((center_channel * 8, center_channel * 8), gridStep,
                                         gridStep, fill=False, edgecolor='black', linewidth=lw * 25, alpha=1.0))


ax_bckgrnd22.pcolormesh(cc_d2_bckgrnd[:len(cc_d2_bckgrnd) / 2,
                                      len(cc_d2_bckgrnd) / 2:], vmin=vmin, vmax=vmax, cmap=cmap)
ax_bckgrnd22.set_xlabel("D2-MSNs", fontsize=15, fontweight='bold')
ax_bckgrnd22.set_ylabel("D2-MSNs", fontsize=15, fontweight='bold')
ax_bckgrnd22.set_xticks(minor_ticks, minor=True)
ax_bckgrnd22.set_yticks(minor_ticks, minor=True)


ax_bckgrnd22.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw, alpha=0.5)
ax_bckgrnd22.set_xlim(100, len(cc_d2_bckgrnd) / 2)
ax_bckgrnd22.set_ylim(100, len(cc_d2_bckgrnd) / 2)

# Mark nearby channels
for chan in nearNeighChan:
    ax_bckgrnd22.add_patch(patches.Rectangle((chan * 8, center_channel * 8), gridStep,
                                             gridStep, fill=False, edgecolor='fuchsia', linewidth=lw * 25, alpha=1.0))
# Mark the stimulated channel
ax_bckgrnd22.add_patch(patches.Rectangle((center_channel * 8, center_channel * 8), gridStep,
                                         gridStep, fill=False, edgecolor='black', linewidth=lw * 25, alpha=1.0))


_pcolor12_bckgrnd = ax_bckgrnd12.pcolormesh(cc_d1_d2_bckgrnd[:len(
    cc_d1_d2_bckgrnd) / 2, len(cc_d1_d2_bckgrnd) / 2:], vmin=vmin, vmax=vmax, cmap=cmap)
ax_bckgrnd12.set_xlabel("D2-MSNs", fontsize=15, fontweight='bold')
ax_bckgrnd12.set_ylabel("D1-MSNs", fontsize=15, fontweight='bold')
ax_bckgrnd12.set_xticks(minor_ticks, minor=True)
ax_bckgrnd12.set_yticks(minor_ticks, minor=True)
ax_bckgrnd12.grid(which='minor', b=True, color='k', linestyle=':', linewidth=lw, alpha=0.5)
ax_bckgrnd12.set_xlim(100, len(cc_d1_d2_bckgrnd) / 2)
ax_bckgrnd12.set_ylim(100, len(cc_d1_d2_bckgrnd) / 2)


# Mark nearby channels
for chan in nearNeighChan:
    ax_bckgrnd12.add_patch(patches.Rectangle((chan * 8, center_channel * 8), gridStep,
                                             gridStep, fill=False, edgecolor='fuchsia', linewidth=lw * 25, alpha=1.0))

# Mark the stimulated channel
ax_bckgrnd12.add_patch(patches.Rectangle((center_channel * 8, center_channel * 8), gridStep,
                                         gridStep, fill=False, edgecolor='black', linewidth=lw * 25, alpha=1.0))

cbar_ax1 = fig_bckgrnd.add_axes([0.935, 0.25, 0.02, 0.5])
fig_bckgrnd.colorbar(_pcolor12_bckgrnd, cax=cbar_ax1)

fig_stim.subplots_adjust(wspace=0.2, hspace=0.15)
fig_stim.savefig(corr_out_with_stim_fn)

fig_bckgrnd.subplots_adjust(wspace=0.2, hspace=0.15)
fig_bckgrnd.savefig(corr_out_with_bckgrnd_fn)
