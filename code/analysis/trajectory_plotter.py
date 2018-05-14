import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
import os
import pickle
import json
import rosbag
import sys
import transformations as trans
import math
import os
import scipy.signal as sciSig
import yaml
import colors
sys.path.append("code/two_hemisphere_model")
import params
from plot_tools2 import *
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


bag_fn = sys.argv[1]
spikes_left_fn = sys.argv[2]
spikes_right_fn = sys.argv[3]
channels_left_fn = sys.argv[4]
channels_right_fn = sys.argv[5]
experiment_fn = sys.argv[6]
traj_out_fn = sys.argv[7]
#turn_out_fn = sys.argv[8]
# spike data of the channels
data_left = np.loadtxt(spikes_left_fn)
senders_left = data_left[:, 0]
unique_senders_left = np.unique(senders_left)  # all active senders
times_left = data_left[:, 1]

with open(channels_left_fn, "r+") as f:
    channels_left = json.load(f)
    channels_left = channels_left['channels']


data_right = np.loadtxt(spikes_right_fn)
senders_right = data_right[:, 0]
unique_senders_right = np.unique(senders_right)  # all active senders
times_right = data_right[:, 1]

with open(channels_right_fn, "r+") as f:
    channels_right = json.load(f)
    channels_right = channels_right['channels']


with open(experiment_fn, "r+") as f:
    cfg = yaml.load(f)


stim_times_start_left, stim_times_stop_left = get_stim_times(
    cfg, "left", params, mask=False, scale=1.)
stim_times_start_right, stim_times_stop_right = get_stim_times(
    cfg, "right", params, mask=False, scale=1.)
stim_times_left = zip(stim_times_start_left / 1000.,
                      stim_times_stop_left / 1000.)
stim_times_right = zip(stim_times_start_right / 1000.,
                       stim_times_stop_right / 1000.)


all_d1_left = np.ravel([c['d1'] for c in channels_left])
all_d2_left = np.ravel([c['d2'] for c in channels_left])
all_d1_right = np.ravel([c['d1'] for c in channels_right])
all_d2_right = np.ravel([c['d2'] for c in channels_right])

spikes_d1_left = np.hstack(
    [times_left[np.where(senders_left == nid)[0]] for nid in all_d1_left])
spikes_d2_left = np.hstack(
    [times_left[np.where(senders_left == nid)[0]] for nid in all_d2_left])
spikes_d1_right = np.hstack(
    [times_right[np.where(senders_right == nid)[0]] for nid in all_d1_right])
spikes_d2_right = np.hstack(
    [times_right[np.where(senders_right == nid)[0]] for nid in all_d2_right])

binsize = 100

hist_all_d1_left = np.histogram(spikes_d1_left, bins=(int(params.runtime)) / binsize)[0].astype(
    'float') * 1000. / (binsize * len(all_d1_left))
hist_all_d2_left = np.histogram(spikes_d2_left, bins=(int(params.runtime)) / binsize)[0].astype(
    'float') * 1000. / (binsize * len(all_d2_left))

hist_all_d1_right = np.histogram(spikes_d1_right, bins=(int(params.runtime)) / binsize)[0].astype(
    'float') * 1000. / (binsize * len(all_d1_right))
hist_all_d2_right = np.histogram(spikes_d2_right, bins=(int(params.runtime)) / binsize)[0].astype(
    'float') * 1000. / (binsize * len(all_d2_right))


bag = rosbag.Bag(bag_fn)

x = np.array([])
y = np.array([])
time = np.array([])

rot_z = np.array([])

for topic, msg, t in bag.read_messages():
    time = np.append(time, t.to_time())
    x = np.append(x, msg.pose.pose.position.x)
    y = np.append(y, msg.pose.pose.position.y)

    quat = msg.pose.pose.orientation
    rot_z = np.append(rot_z, trans.euler_from_quaternion(
        [quat.x, quat.y, quat.z, quat.w])[2])

bag.close()

time -= time[0]

mask = np.where(time <= 200)
time = time[mask]
x = x[mask]
y = y[mask]
rot_z = rot_z[mask]

rot_z = np.diff(rot_z)
rot_z = rot_z[np.where(abs(rot_z) < np.pi / 2.)]

i = 0
xp = []
yp = []
rot_z_p = []
inds = []

for ind, t in enumerate(time):
    if t > i:
        xp.append(x[ind])
        yp.append(y[ind])
        if ind < len(rot_z):  # To prevent errors, sometimes gives an error ind is > len(rot_z)
            rot_z_p.append(rot_z[ind])
            inds.append(ind)
        # transform maximal robotic time time[-1] (see launch trial script) by NEST simtime (20s)
        i += time[-1] / 20.


def isDuringStimulation(t):
    if "no_stim" in experiment_fn:
        return False

    for stim_time in stim_times_left:
        if t >= stim_time[0] and t < stim_time[1]:
            return True

    for stim_time in stim_times_right:
        if t >= stim_time[0] and t < stim_time[1]:
            return True

    return False


colors.seaborn.set_context('paper', font_scale=3.0,
                           rc={"lines.linewidth": 2.5})
colors.seaborn.set_style('whitegrid', {"axes.linewidth": 2.5})

fig_traj = pl.figure(figsize=[16, 10])
ax_left_hist = pl.subplot2grid((2, 2), (0, 0))
ax_right_hist = pl.subplot2grid((2, 2), (1, 0))
ax_traj = pl.subplot2grid((2, 2), (0, 1), rowspan=2)

x_ = x[::1]
y_ = y[::1]

points = np.array([x_, y_]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

conv = mcolors.ColorConverter().to_rgb


cmap = LinearSegmentedColormap.from_list(
    "my_cmap", [conv(colors.colors[0]), conv(colors.colors[1])], N=1000)

lc = LineCollection(segments, cmap=cmap,
                    norm=pl.Normalize(0, 10))

t = np.linspace(0, 10, len(x_))

lc.set_array(t)
lc.set_linewidth(7)

for t, _ in enumerate(xp):
    if isDuringStimulation(t):
        ax_traj.plot(xp[t], yp[t], '.', color=colors.colors[2], markersize=40.)
    else:
        ax_traj.plot(xp[t], yp[t], '.', color=colors.colors[4], markersize=40.)


ax_traj.plot(xp[0], yp[0], 'g*', markersize=50)
ax_traj.set_xlabel("x", fontweight='bold')
ax_traj.set_ylabel("y ", fontweight='bold')


for tick_label in ax_traj.get_xticklabels():
    tick_label.set_fontweight('bold')
for tick_label in ax_traj.get_yticklabels():
    tick_label.set_fontweight('bold')

ax_traj.set_xlim([np.min(x) - 2, np.max(x) + 2])
ax_traj.set_ylim([np.min(y) - 2, np.max(y) + 2])

ax_traj.add_collection(lc)

ax_left_hist.plot(np.arange(len(hist_all_d1_left)) /
                  (1000. / binsize), hist_all_d1_left)
ax_left_hist.plot(np.arange(len(hist_all_d2_left)) /
                  (1000. / binsize), hist_all_d2_left)

ax_right_hist.plot(np.arange(len(hist_all_d1_right)) /
                   (1000. / binsize), hist_all_d1_right)
ax_right_hist.plot(np.arange(len(hist_all_d2_right)) /
                   (1000. / binsize), hist_all_d2_right)
ax_left_hist.text(-0.20, 0.97, 'L',
                  transform=ax_left_hist.transAxes, fontsize=35)

ax_right_hist.text(-0.20, 0.97, 'R',
                  transform=ax_right_hist.transAxes, fontsize=35)

#ax_left_hist.set_title("Left hemisphere")
#ax_right_hist.set_title("Right hemisphere")

ax_left_hist.set_ylabel("Firing rate (spks/s)", fontweight='bold')
ax_right_hist.set_ylabel("Firing rate (spks/s)", fontweight='bold')
ax_right_hist.set_xlabel("Time (s)", fontweight='bold')



# Filter the trace rot_z with low pass filter to make it easier to find change in directions
B, A = sciSig.butter(3, 0.1, btype='low', analog=False)
rotFilt = sciSig.lfilter(B, A, rot_z)

left_turn_inds = np.where(rot_z < -0.005)
left_turn_filt_inds = np.where(rotFilt < -0.005)
left_turn_chunks_ids = np.where(np.diff(left_turn_filt_inds) > 1)[1]
# np.append(0,left_turn_chunks_ids)
left_turn_chunks = []
for i, x in enumerate(left_turn_chunks_ids):
    if i == 0:
        left_turn_chunks.append(left_turn_filt_inds[0][0:x + 1])
    else:
        left_turn_chunks.append(
            left_turn_filt_inds[0][left_turn_chunks_ids[i - 1] + 1:x + 1])

left_turn_chunk_times = [np.array(time[x]) / (time[-1] / 20.)
                         for x in left_turn_chunks]

right_turn_inds = np.where(rot_z > 0.005)
right_turn_filt_inds = np.where(rotFilt > 0.005)
right_turn_chunks_ids = np.where(np.diff(right_turn_filt_inds) > 1)[1]
# np.append(0,left_turn_chunks_ids)
right_turn_chunks = []
for i, x in enumerate(right_turn_chunks_ids):
    if i == 0:
        right_turn_chunks.append(right_turn_filt_inds[0][0:x + 1])
    else:
        right_turn_chunks.append(
            right_turn_filt_inds[0][right_turn_chunks_ids[i - 1] + 1:x + 1])


right_turn_chunk_times = [
    np.array(time[x]) / (time[-1] / 20.) for x in right_turn_chunks]


ax = pl.axes([.80, .80, .15, .15])

ax.plot(np.linspace(0, 20, len(rot_z)), rot_z)

stim_start = 0  # Flag to detect when stimulation starts for the first stim
stim_starttime=[]
stim_stoptime=[]

for t, _ in enumerate(xp):
    try:
        if isDuringStimulation(t):
            if stim_start == 0:
                stim_starttime.append(t)
                stim_start = 1
            ax.plot(t, rot_z_p[t], '.r', markersize=10., linewidth=1., color=colors.colors[2])
        else:
            ax.plot(t, rot_z_p[t], '.g', markersize=10., color=colors.colors[4])
            # First time it comes here after stim_start = 1
            if stim_start == 1:
                stim_stoptime.append(t)
                stim_start = 0 
    except:
        pass
print "stim_starttime",stim_starttime
print "stim_stoptime",stim_stoptime

for x,y in zip(stim_starttime,stim_stoptime):
    ax.axvspan(x,y,ymin=abs(min(rot_z))/(abs(max(rot_z))+abs(min(rot_z))),ymax=1.0,alpha=0.3,color=colors.colors[5])
    ax.axvspan(x,y,ymin=0,ymax=abs(min(rot_z))/(abs(max(rot_z))+abs(min(rot_z))),alpha=0.3,color=colors.colors[3])
ax.set_xticklabels([])
ax.set_yticks([min(rot_z), 0.0, max(rot_z)])

#if "no_stim.yaml" in experiment_fn:
ax.set_yticklabels(['R', '0', 'L'],fontsize=30)
#else:
#    ax.set_yticklabels([])

ax.set_ylim([min(rot_z), max(rot_z)])
ax.yaxis.tick_right()


#ax.vlines(left_turn_inds[0], -0.04, 0.04, colors.colors[1], alpha=0.2)
#ax.vlines(right_turn_inds[0], -0.04, 0.04, colors.colors[2], alpha=0.2)

#fig.savefig(turn_out_fn)
pl.savefig(traj_out_fn)
