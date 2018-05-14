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

fn_in = sys.argv[1]
channels_fn = sys.argv[2]
fn_out = sys.argv[3]


with open(channels_fn, "r+") as f:
    channels = json.load(f)
    channels = channels['channels']

all_d1 = np.ravel([c['d1'] for c in channels])
all_d2 = np.ravel([c['d2'] for c in channels])

spike_data = np.loadtxt(fn_in)
senders = spike_data[:, 0]
times = spike_data[:, 1]

mask_d1 = [nid in all_d1 for nid in senders]
mask_d2 = [nid in all_d2 for nid in senders]


f = pl.figure(figsize=[32, 10])
ax = f.add_subplot(1, 1, 1)

ax.plot(times[mask_d1], senders[mask_d1], '.', color=colors.colors[0])
ax.plot(times[mask_d2], senders[mask_d2], '.', color=colors.colors[1])

ax.set_xticklabels(ax.get_xticks().astype('int') / 1000)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Neuron id")

pl.tight_layout()
pl.savefig(fn_out)
