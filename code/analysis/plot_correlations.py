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

grid_dimensions = [6, 6]

spikes_fn = sys.argv[1]
channels_fn = sys.argv[2]
experiment_fn = sys.argv[3]
hemisphere = sys.argv[4]
corr_out_fn = sys.argv[5]

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

#alpha_filter = alpha.pdf(np.arange(0, 20, 0.005), 1)
exp_filter = np.exp(np.arange(0, 5, 0.001) / -0.3)

# within < 1, near < 2, far >= 2
cc_by_dist = {'d1d1_far_stim': [], 'd2d2_far_stim': [], 'd1d2_far_stim': [],
              'd1d1_within_stim': [], 'd2d2_within_stim': [], 'd1d2_within_stim': [],
              'd1d1_near_stim': [], 'd2d2_near_stim': [], 'd1d2_near_stim': [],
              'd1d1_far_bckgrnd': [], 'd2d2_far_bckgrnd': [], 'd1d2_far_bckgrnd': [],
              'd1d1_within_bckgrnd': [], 'd2d2_within_bckgrnd': [], 'd1d2_within_bckgrnd': [],
              'd1d1_near_bckgrnd': [], 'd2d2_near_bckgrnd': [], 'd1d2_near_bckgrnd': []}

selected_neurons = np.random.choice(unique_senders, 500, replace=False)
spikes = get_spikes_mask(senders, times, selected_neurons)
filtered_spikes = filter_spikes(spikes, exp_filter)

filtered_spikes_stim = np.array(filtered_spikes)[:, np.where(stim_times == 1)[0]]
filtered_spikes_bckgrnd = np.array(filtered_spikes)[:, np.where(bckgrnd_times == 1)[0]]

correlations_stim = correlate(filtered_spikes_stim)
correlations_bckgrnd = correlate(filtered_spikes_bckgrnd)


for i, n0 in enumerate(selected_neurons):
    cell_type_n0, pos_n0 = get_grid_pos(n0, channels)
    for j, n1 in enumerate(selected_neurons):
        if j <= i:  # only scan upper half of correlation matix
            continue

        cell_type_n1, pos_n1 = get_grid_pos(n1, channels)
        dist = get_dist(pos_n0, pos_n1)

        if dist < 1.:
            if cell_type_n0 == 'd1' and cell_type_n1 == 'd1':
                cc_by_dist['d1d1_within_stim'].append(correlations_stim[i, j])
                cc_by_dist['d1d1_within_bckgrnd'].append(correlations_bckgrnd[i, j])
            elif cell_type_n0 == 'd2' and cell_type_n1 == 'd2':
                cc_by_dist['d2d2_within_stim'].append(correlations_stim[i, j])
                cc_by_dist['d2d2_within_bckgrnd'].append(correlations_bckgrnd[i, j])
            elif cell_type_n0 == 'd1' and cell_type_n1 == 'd2':
                cc_by_dist['d1d2_within_stim'].append(correlations_stim[i, j])
                cc_by_dist['d1d2_within_bckgrnd'].append(correlations_bckgrnd[i, j])
        elif dist < 2.:
            if cell_type_n0 == 'd1' and cell_type_n1 == 'd1':
                cc_by_dist['d1d1_near_stim'].append(correlations_stim[i, j])
                cc_by_dist['d1d1_near_bckgrnd'].append(correlations_bckgrnd[i, j])
            elif cell_type_n0 == 'd2' and cell_type_n1 == 'd2':
                cc_by_dist['d2d2_near_stim'].append(correlations_stim[i, j])
                cc_by_dist['d2d2_near_bckgrnd'].append(correlations_bckgrnd[i, j])
            elif cell_type_n0 == 'd1' and cell_type_n1 == 'd2':
                cc_by_dist['d1d2_near_stim'].append(correlations_stim[i, j])
                cc_by_dist['d1d2_near_bckgrnd'].append(correlations_bckgrnd[i, j])
        else:
            if cell_type_n0 == 'd1' and cell_type_n1 == 'd1':
                cc_by_dist['d1d1_far_stim'].append(correlations_stim[i, j])
                cc_by_dist['d1d1_far_bckgrnd'].append(correlations_bckgrnd[i, j])
            elif cell_type_n0 == 'd2' and cell_type_n1 == 'd2':
                cc_by_dist['d2d2_far_stim'].append(correlations_stim[i, j])
                cc_by_dist['d2d2_far_bckgrnd'].append(correlations_bckgrnd[i, j])
            elif cell_type_n0 == 'd1' and cell_type_n1 == 'd2':
                cc_by_dist['d1d2_far_stim'].append(correlations_stim[i, j])
                cc_by_dist['d1d2_far_bckgrnd'].append(correlations_bckgrnd[i, j])


matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

fig = pl.figure(figsize=[10, 16])
ax_d1d1_stim = fig.add_subplot("311")
ax_d2d2_stim = fig.add_subplot("312")
ax_d1d2_stim = fig.add_subplot("313")


if "no_stim" in experiment_fn:
    title = "No Stim"
elif "bilateral_D1" in experiment_fn:
    title = "Bilateral D1"
elif "bilateral_D2" in experiment_fn:
    title = "Bilateral D2"
elif "unilateral_D1" in experiment_fn:
    title = "Unilateral D1 Exc"
else:
    title = experiment_fn.split("/")[-1].replace(".yaml", "")

ax_d1d1_stim.set_title(title, fontsize=30, fontweight='bold')
ax_d1d1_stim.grid(linewidth=0.6)
ax_d2d2_stim.grid(linewidth=0.6)
ax_d1d2_stim.grid(linewidth=0.6)

ax_d1d1_stim.set_ylabel("D1-D1", fontsize=30, fontweight='bold')
ax_d2d2_stim.set_ylabel("D2-D2", fontsize=30, fontweight='bold')
ax_d1d2_stim.set_ylabel("D1-D2", fontsize=30, fontweight='bold')

ax_d1d2_stim.set_xlabel("CC", fontsize=30, fontweight='bold')


ylim = 7.0
ms = 15
lw = 3
for cc_type in cc_by_dist.keys():
    cell_type, dist, stim_type = cc_type.split('_')

    if dist == 'within':
        color = colors[0]
    elif dist == 'near':
        color = colors[3]       # Trying to different slightly more disparate color schemes
    elif dist == 'far':
        color = colors[1]

    hist, edges = np.histogram(cc_by_dist[cc_type], bins=np.linspace(-0.6, 1.1, 15),
                               normed=True)
    if cell_type == 'd1d1':
        if stim_type == 'stim':
            ax_d1d1_stim.plot(edges[:-1], hist, 'o-', color=color,
                              label="Dist: {} D1-D1".format(dist), linewidth=lw, markersize=ms)
            ax_d1d1_stim.set_ylim(-0.5, ylim)
            for x in ax_d1d1_stim.get_xticklabels():
                x.set_fontweight('bold')
                x.set_fontsize(20)
            for x in ax_d1d1_stim.get_yticklabels():
                x.set_fontweight('bold')
                x.set_fontsize(20)

    if cell_type == 'd2d2':
        if stim_type == 'stim':
            ax_d2d2_stim.plot(edges[:-1], hist, 'o-', color=color,
                              label="Dist: {} D2-D2".format(dist), linewidth=lw, markersize=ms)
            ax_d2d2_stim.set_ylim(-0.5, ylim)
            for x in ax_d2d2_stim.get_xticklabels():
                x.set_fontweight('bold')
                x.set_fontsize(20)
            for x in ax_d2d2_stim.get_yticklabels():
                x.set_fontweight('bold')
                x.set_fontsize(20)

    if cell_type == 'd1d2':
        if stim_type == 'stim':
            ax_d1d2_stim.plot(edges[:-1], hist, 'o-', color=color,
                              label="Dist: {} D1-D2".format(dist), linewidth=lw, markersize=ms)
            ax_d1d2_stim.set_ylim(-0.5, ylim)
            for x in ax_d1d2_stim.get_xticklabels():
                x.set_fontweight('bold')
                x.set_fontsize(20)
            for x in ax_d1d2_stim.get_yticklabels():
                x.set_fontweight('bold')
                x.set_fontsize(20)

fig.tight_layout(w_pad=0.2)
seaborn.despine()
pl.savefig(corr_out_fn)
