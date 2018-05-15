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
    __file__), '..', 'striatal_model/'))
import params as p
import json
import matplotlib.pyplot as pl
from itertools import combinations
import itertools
import yaml
from pylab import *
import colors
from plot_tools2 import *
import copy
import seaborn as sbn
import pandas
import matplotlib

colors.seaborn.set_context('paper', font_scale=3.0,
                           rc={"lines.linewidth": 1.5})
colors.seaborn.set_style('whitegrid', {"axes.linewidth": 1.5})

#matplotlib.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'stix'



trials = 5

doughnut_spikes_with_antagonist_fns = sys.argv[1:trials+1]
doughnut_spikes_without_antagonist_fns = sys.argv[trials+1:2*trials+1]
doughnut_channels_fns = sys.argv[2*trials+1:3*trials+1]
doughnut_with_antagonist_experiment_fn = sys.argv[3*trials+1]
doughnut_without_antagonist_experiment_fn = sys.argv[3*trials+2]

exp_spikes_with_antagonist_fns = sys.argv[3*trials+3:4*trials+3]
exp_spikes_without_antagonist_fns = sys.argv[4*trials+3:5*trials+3]
exp_channels_fns = sys.argv[5*trials+3:6*trials+3]
exp_with_antagonist_experiment_fn = sys.argv[6*trials+3]
exp_without_antagonist_experiment_fn = sys.argv[6*trials+4]

out_fn = sys.argv[6*trials+5]


def calc_min_dists(spikes_with_antagonist_fns, spikes_without_antagonist_fns, channels_fns, with_antagonist_experiment_fn, without_antagonist_experiment_fn):

    with open(with_antagonist_experiment_fn, "r+") as f:
        cfg_with = yaml.load(f)

    with open(without_antagonist_experiment_fn, "r+") as f:
        cfg_without = yaml.load(f)

    min_dists_distributions = []
    num_active_neurons = []

    for spikes_with_antagonist_fn, spikes_without_antagonist_fn, channels_fn in zip(spikes_with_antagonist_fns, spikes_without_antagonist_fns, channels_fns):

        with open(channels_fn, "r+") as f:
            channels = json.load(f)
            channels = channels['channels']

        # spike data of the channels
        data_with = np.loadtxt(spikes_with_antagonist_fn)
        senders_with_orig = data_with[:, 0]
        unique_senders_with = np.unique(
            senders_with_orig)  # all active senders
        times_with_orig = data_with[:, 1]

        data_without = np.loadtxt(spikes_without_antagonist_fn)
        senders_without_orig = data_without[:, 0]
        unique_senders_without = np.unique(
            senders_without_orig)  # all active senders
        times_without_orig = data_without[:, 1]

        random_choice = np.random.choice(unique_senders_with, 500)
        senders_with = senders_with_orig[[
            s in random_choice for s in senders_with_orig]]
        times_with = times_with_orig[[
            s in random_choice for s in senders_with_orig]]

        senders_without = senders_without_orig[[
            s in random_choice for s in senders_without_orig]]
        times_without = times_without_orig[[
            s in random_choice for s in senders_without_orig]]

        stim_times_start_with, stim_times_stop_with = get_stim_times(
            cfg_with, "left", p, mask=False, scale=1.)
        stim_times_start_without, stim_times_stop_without = get_stim_times(
            cfg_without, "left", p, mask=False, scale=1.)

        # Isolate the spike times only during stimulation
        inds_stims_with = [np.logical_and(times_with >= start, times_with <= stop)
                           for start, stop in zip(stim_times_start_with, stim_times_stop_with)]
        times_stims_with = [y for x in inds_stims_with for y in times_with[x]]
        senders_stims_with = [
            y for x in inds_stims_with for y in senders_with[x]]

        inds_stims_without = [np.logical_and(times_without >= start, times_without <= stop)
                              for start, stop in zip(stim_times_start_without, stim_times_stop_without)]
        times_stims_without = [
            y for x in inds_stims_without for y in times_without[x]]
        senders_stims_without = [
            y for x in inds_stims_without for y in senders_without[x]]

        uni_senders_with = np.unique(senders_stims_with)
        uni_senders_without = np.unique(senders_stims_without)

        # get position of active neurons
        pos_with = []
        pos_without = []

        for nid in uni_senders_without:
            pos_without.append(get_grid_pos(nid, channels))

        # calculate nearest distance between active neurons in
        # with / without antagonist experiment
        inhibited_neurons = list(
            set(uni_senders_with) - set(uni_senders_without))

        min_dists = []
        for nid in inhibited_neurons:
            pos_inh = get_grid_pos(nid, channels)
            dists = []
            for pos_wo in pos_without:
                dists.append(get_dist(pos_inh[1], pos_wo[1]))
            min_dists.append(min(dists))

        dist_hist, _ = np.histogram(min_dists, bins=all_possible_distances)

        min_dists_distributions.append(dist_hist)
        num_active_neurons.append(
            [len(uni_senders_without), len(uni_senders_with)])

    return [min_dists_distributions, num_active_neurons, senders_with, senders_without, times_with, times_without]


doughnut_min_dists_distributions, doughnut_num_active_neurons, senders_with, senders_without, times_with, times_without = calc_min_dists(
    doughnut_spikes_with_antagonist_fns, doughnut_spikes_without_antagonist_fns, doughnut_channels_fns, doughnut_with_antagonist_experiment_fn, doughnut_without_antagonist_experiment_fn)
exp_min_dists_distributions, exp_num_active_neurons, _, _, _, _ = calc_min_dists(
    exp_spikes_with_antagonist_fns, exp_spikes_without_antagonist_fns, exp_channels_fns, exp_with_antagonist_experiment_fn, exp_without_antagonist_experiment_fn)

print doughnut_num_active_neurons
doughnut_num_active_neurons = pandas.DataFrame(list(np.array(doughnut_num_active_neurons)[
                                               :, 0]) + list(np.array(doughnut_num_active_neurons)[:, 1]))
doughnut_num_active_neurons['# Active neurons'] = doughnut_num_active_neurons[0]
doughnut_num_active_neurons['gaba-present'] = ["False" if x <
                                               trials else "True" for x in xrange(len(doughnut_num_active_neurons))]
doughnut_num_active_neurons['Type'] = ['doughnut' for x in xrange(
    len(doughnut_num_active_neurons['gaba-present']))]

exp_num_active_neurons = pandas.DataFrame(list(np.array(exp_num_active_neurons)[
                                          :, 0]) + list(np.array(exp_num_active_neurons)[:, 1]))
exp_num_active_neurons['# Active neurons'] = exp_num_active_neurons[0]
exp_num_active_neurons['gaba-present'] = ["False" if x <
                                          trials else "True" for x in xrange(len(exp_num_active_neurons))]
exp_num_active_neurons['Type'] = ['exponential' for x in xrange(
    len(exp_num_active_neurons['gaba-present']))]

# split raster plot

plot_senders_without = senders_without[np.where(times_without < 10000)]
plot_times_without = times_without[np.where(times_without < 10000)]

plot_senders_with = senders_with[np.where(times_with > 10000)]
plot_times_with = times_with[np.where(times_with > 10000)]


with open(doughnut_with_antagonist_experiment_fn, "r+") as f:
    cfg_with = yaml.load(f)

with open(doughnut_without_antagonist_experiment_fn, "r+") as f:
    cfg_without = yaml.load(f)


stim_times_start_with, stim_times_stop_with = get_stim_times(
    cfg_with, "left", p, mask=False, scale=1.)
stim_times_start_without, stim_times_stop_without = get_stim_times(
    cfg_without, "left", p, mask=False, scale=1.)


# Isolate the spike times only during stimulation
inds_stims_with = [np.logical_and(times_with >= start, times_with <= stop)
                   for start, stop in zip(stim_times_start_with, stim_times_stop_with)]
times_stims_with = [y for x in inds_stims_with for y in times_with[x]]
senders_stims_with = [y for x in inds_stims_with for y in senders_with[x]]

inds_stims_without = [np.logical_and(times_without >= start, times_without <= stop)
                      for start, stop in zip(stim_times_start_without, stim_times_stop_without)]
times_stims_without = [y for x in inds_stims_without for y in times_without[x]]
senders_stims_without = [
    y for x in inds_stims_without for y in senders_without[x]]


#expName = (with_antagonist_experiment_fn.split('/')[-1]).replace('.yaml','')


fig = pl.figure(figsize=[16, 10])
t1 = fig.add_subplot(211)
#t1.set_title(expName, fontweight='bold')
# Plot the raster and mark the neurons stimulated
t1.plot(np.array(plot_times_without)/1000., plot_senders_without,
        '.', color=colors.colors[0], markersize=8)
t1.plot(np.array(plot_times_with)/1000., plot_senders_with,
        '.', color=colors.colors[0], markersize=8)
t1.plot(np.array(times_stims_without)/1000., senders_stims_without,
        '.', color=colors.colors[2], markersize=8)
t1.plot(np.array(times_stims_with)/1000., senders_stims_with,
        '.', color=colors.colors[2], markersize=8)
t1.set_xlim([0, p.runtime/1000.])
t1.set_ylabel("Neuron ids", fontweight='bold')
t1.set_xlabel("Time (s)", fontweight='bold')
t1.vlines(10000/1000., min(plot_senders_with), max(plot_senders_with),
          linestyles='dashed', color=colors.colors[3], linewidth=4.0)

t1.text(-0.10, 1., 'A',
        fontsize=30,
        fontweight='bold',
        horizontalalignment='center',
        verticalalignment='center',
        transform=t1.transAxes)

# print doughnut_num_active_neurons
# print exp_num_active_neurons
# Plot number of coactive groups in left and right hemisphere
t2 = fig.add_subplot(223)

# mergedDataFrame = exp_num_active_neurons.merge(doughnut_num_active_neurons,left_index=True)#,right_index=True)
mergedDataFrame = exp_num_active_neurons.append(doughnut_num_active_neurons)
# print mergedDataFrame
# sbn.boxplot(data=mergedDataFrame, ax=t2, width=0.2,hue='Type',x='gaba-present',y='# Active neurons')
g = sbn.barplot(data=mergedDataFrame, ax=t2, hue='Type',
            x='gaba-present', y='# Active neurons',palette=[colors.colors[1],colors.colors[2]])
#sbn.boxplot(doughnut_num_active_neurons, ax=t2, color=colors.colors[1], width=0.2)
t2.set_xticklabels(["w/o GABA Antagonist", "with GABA Antagonist"])
t2.set_xlabel(' ')
g.set_xticklabels(g.get_xticklabels(), rotation=10)
#t2.legend(prop={'size': 12, 'weight': 'bold'}, loc=0)
t2.legend([])
t2.set_ylabel("# Coactive neurons", fontweight='bold')

t2.text(-0.23, 1., 'B',
        fontsize=30,
        fontweight='bold',
        horizontalalignment='center',
        verticalalignment='center',
        transform=t2.transAxes)



paper_rc = {'lines.linewidth': 3.5, 'lines.markersize': 10}
sbn.set_context(rc=paper_rc)
# Plot the number of co-active groups vs distances, pooled results for left and right
# transform x ticks to micrometer
t3 = fig.add_subplot(224)
ax1 = sbn.tsplot(exp_min_dists_distributions, time=np.array(
    all_possible_distances[:-1])*40, ax=t3, color=colors.colors[1], marker='o')
t3.set_xlim(0, np.max(all_possible_distances)*40)
t3.set_xlabel('Neuronal pair distance ('+r'$\mathbf{\mu m}$)',
              fontweight='bold')
t3.tick_params('y', colors=colors.colors[1])
ax1.grid(False)


t31 = t3.twinx()
ax2 = sbn.tsplot(doughnut_min_dists_distributions, time=np.array(
    all_possible_distances[:-1])*40, ax=t31, color=colors.colors[2], marker='o')
t31.tick_params('y', colors=colors.colors[2])


t3.text(-0.15, 1., 'C',
        fontsize=30,
        fontweight='bold',
        horizontalalignment='center',
        verticalalignment='center',
        transform=t3.transAxes)



ax2.grid(False)
fig.subplots_adjust(hspace=0.35, top=0.9)
fig.savefig(out_fn)
