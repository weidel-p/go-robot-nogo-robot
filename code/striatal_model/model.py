#!/usr/bin/python
import numpy as np
import time
import params as p
import nest
import sys
import yaml
import json
import pdb

from mpi4py import MPI
comm = MPI.COMM_WORLD


with open("cfg.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

with open("scale.json", 'r') as f:
    scale = float(json.load(f)["scale"])

if scale == 1.:
    folder_ = "data"
else:
    folder_ = "data_long"

seed = [np.random.randint(0, 9999999)] * p.num_threads
print "seed", seed
nest.ResetKernel()
nest.SetKernelStatus({"resolution": p.timestep, "overwrite_files": True,
                      "rng_seeds": seed, "print_time": True, "local_num_threads": p.num_threads})

nest.set_verbosity("M_FATAL")

# copy models

nest.CopyModel("iaf_cond_alpha", "d1", p.d1_params)
nest.CopyModel("iaf_cond_alpha", "d2", p.d2_params)


# Background noise
noise_d1 = nest.Create("poisson_generator", 1, {"rate": p.bg_noise_d1})
noise_d2 = nest.Create("poisson_generator", 1, {"rate": p.bg_noise_d2})


def connWithinChannels(ch1d1, ch1d2):
    nest.Connect(ch1d1, ch1d1, p.conn_params_d1d1_within_chan,
                 p.syn_params_d1d1_within_chan)
    nest.Connect(ch1d1, ch1d2, p.conn_params_d1d2_within_chan,
                 p.syn_params_d1d2_within_chan)
    nest.Connect(ch1d2, ch1d1, p.conn_params_d2d1_within_chan,
                 p.syn_params_d2d1_within_chan)
    nest.Connect(ch1d2, ch1d2, p.conn_params_d2d2_within_chan,
                 p.syn_params_d2d2_within_chan)


def connBetweenChannels(ch1d1, ch1d2, ch2d1, ch2d2, which):
    if which == "Near":  # Is the channel far or near ?
        nest.Connect(ch1d1, ch2d1, p.conn_params_d1d1_between_near_chan,
                     p.syn_params_d1d1_between_near_chan)
        nest.Connect(ch1d1, ch2d2, p.conn_params_d1d2_between_near_chan,
                     p.syn_params_d1d2_between_near_chan)
        nest.Connect(ch1d2, ch2d1, p.conn_params_d2d1_between_near_chan,
                     p.syn_params_d2d1_between_near_chan)
        nest.Connect(ch1d2, ch2d2, p.conn_params_d2d2_between_near_chan,
                     p.syn_params_d2d2_between_near_chan)
    else:
        nest.Connect(ch1d1, ch2d1, p.conn_params_d1d1_between_far_chan,
                     p.syn_params_d1d1_between_far_chan)
        nest.Connect(ch1d1, ch2d2, p.conn_params_d1d2_between_far_chan,
                     p.syn_params_d1d2_between_far_chan)
        nest.Connect(ch1d2, ch2d1, p.conn_params_d2d1_between_far_chan,
                     p.syn_params_d2d1_between_far_chan)
        nest.Connect(ch1d2, ch2d2, p.conn_params_d2d2_between_far_chan,
                     p.syn_params_d2d2_between_far_chan)


def saveGIDs(data, label):

    with open(label, 'w+') as f:
        json.dump(data, f)


def create_hemisphere(label):

    # create spike detector
    sd = nest.Create("spike_detector", 1, {"to_file": True, 'label': label, "use_gid_in_filename": False})

    # create all channel neurons
    channels = []
    for i in range(p.num_channels):
        # position in the grid
        row = i / p.grid_size[0][0]
        col = i % p.grid_size[0][1]
        channels.append({'d1': nest.Create('d1', p.num_neurons_per_channel),
                         'd2': nest.Create('d2', p.num_neurons_per_channel), 'row': row, 'col': col})

    for i, c0 in enumerate(channels):

        # connect background noise
        nest.Connect(noise_d1, c0['d1'], syn_spec={"weight": p.bg_weight_d1, "delay": 1.})
        nest.Connect(noise_d2, c0['d2'], syn_spec={"weight": p.bg_weight_d2, "delay": 1.})

        # connect to sd
        nest.Connect(c0['d1'] + c0['d2'], sd)

        flags = np.zeros((p.grid_size[0]))  # To prevent connecting the two channels twice

        # connect within channel
        connWithinChannels(c0['d1'], c0['d2'])

        # connect between channels - first near ones, which are all of those for which row and cols differ by 1
        srcRow = c0['row']
        srcCol = c0['col']
        flags[srcRow][srcCol] = 1  # Because of the connWithinChannels
        for j in xrange(8):  # There are 8 near connections around every channel
            if j == 0:
                destRow = srcRow + 1
                destCol = srcCol
            if j == 1:
                destRow = srcRow
                destCol = srcCol + 1
            if j == 2:
                destRow = srcRow + 1
                destCol = srcCol + 1
            if j == 3:
                destRow = srcRow - 1
                destCol = srcCol
            if j == 4:
                destRow = srcRow
                destCol = srcCol - 1
            if j == 5:
                destRow = srcRow - 1
                destCol = srcCol - 1
            if j == 6:
                destRow = srcRow - 1
                destCol = srcCol + 1
            if j == 7:
                destRow = srcRow + 1
                destCol = srcCol - 1

            # Trying to wrap up the grid
            if destRow < 0:
                destRow = p.grid_size[0][0] - 1
            if destCol < 0:
                destCol = p.grid_size[0][1] - 1
            if destRow > p.grid_size[0][0] - 1:
                destRow = 0
            if destCol > p.grid_size[0][1] - 1:
                destCol = 0
            for dest in channels:
                if dest['row'] == destRow and dest['col'] == destCol and flags[destRow][destCol] == 0:
                    connBetweenChannels(c0['d1'], c0['d2'], dest['d1'], dest['d2'], 'Near')
                    flags[destRow][destCol] = 1

        # Now connecting the farther channels
        for j, c1 in enumerate(channels):
            if np.abs(c1['row'] - c0['row']) > 1 or np.abs(c1['col'] - c0['col']) > 1 and flags[c1['row']][c1['col']] == 0:
                # far channels
                connBetweenChannels(c0['d1'], c0['d2'], c1['d1'], c1['d2'], 'Far')

                flags[c1['row']][c1['col']] = 1

    return {'channels': channels, 'spike_detector': sd}


# CREATE HEMISPHERES

left_hemisphere = create_hemisphere("../../{}/left_hemisphere".format(folder_))
right_hemisphere = create_hemisphere("../../{}/right_hemisphere".format(folder_))


try:
    if cfg["inter-hemisphere-connection"]:
        # Connect D1s of both hemisperes - refer to Cui at al , D1 of ipsilateral hemisphere decreases during contralateral movement
        d1sLeft = [y for x in left_hemisphere['channels'] for y in x['d1']]
        d1sRight = [y for x in right_hemisphere['channels'] for y in x['d1']]

        nest.Connect(d1sLeft, d1sRight, conn_spec={'rule': 'fixed_outdegree', 'outdegree': int(
            p.c_inter_hemis * len(d1sRight))}, syn_spec={"weight": p.j_inter_hemis, "delay": 10})
        nest.Connect(d1sRight, d1sLeft, conn_spec={'rule': 'fixed_outdegree', 'outdegree': int(
            p.c_inter_hemis * len(d1sLeft))}, syn_spec={"weight": p.j_inter_hemis, "delay": 10})

        # Connect D2s of both hemisperes - refer to Cui at al , D2 of ipsilateral hemisphere decreases during contralateral movement
        d2sLeft = [y for x in left_hemisphere['channels'] for y in x['d2']]
        d2sRight = [y for x in right_hemisphere['channels'] for y in x['d2']]

        nest.Connect(d2sLeft, d2sRight, conn_spec={'rule': 'fixed_outdegree', 'outdegree': int(
            p.c_inter_hemis * len(d2sRight))}, syn_spec={"weight": p.j_inter_hemis, "delay": 10})
        nest.Connect(d2sRight, d2sLeft, conn_spec={'rule': 'fixed_outdegree', 'outdegree': int(
            p.c_inter_hemis * len(d2sLeft))}, syn_spec={"weight": p.j_inter_hemis, "delay": 10})
except:
    pass

try:
    if cfg["cut-D2D2-connections"]:
        # Connect D1s of both hemisperes - refer to Cui at al , D1 of ipsilateral hemisphere decreases during contralateral movement
        for chan in left_hemisphere['channels']:
            if chan['col'] == 4 and chan['row'] == 3:
                chan_go_right_d2 = chan['d2']
            if chan['col'] == 3 and chan['row'] == 3:
                chan_go_left_d2 = chan['d2']

        nest.SetStatus(nest.GetConnections(chan_go_left_d2, chan_go_right_d2), {'weight': 0.})
        nest.SetStatus(nest.GetConnections(chan_go_right_d2, chan_go_left_d2), {'weight': 0.})

        for chan in right_hemisphere['channels']:
            if chan['col'] == 4 and chan['row'] == 3:
                chan_go_right_d2 = chan['d2']
            if chan['col'] == 3 and chan['row'] == 3:
                chan_go_left_d2 = chan['d2']

        nest.SetStatus(nest.GetConnections(chan_go_left_d2, chan_go_right_d2), {'weight': 0.})
        nest.SetStatus(nest.GetConnections(chan_go_right_d2, chan_go_left_d2), {'weight': 0.})
except:
    pass


# Save GIDs for later
saveGIDs(left_hemisphere, "../../{}/neuron_ids_left_hemisphere.json".format(folder_))
saveGIDs(right_hemisphere, "../../{}/neuron_ids_right_hemisphere.json".format(folder_))


# CREATE STIM

for s in cfg["stim-params"]:
    if s['hemisphere'] == 'none':
        # no stimulation, only background input
        pass
    else:
        num_stimulations = len(s['start_times'])
        for i in range(num_stimulations):
            start_time = s['start_times'][i]
            stop_time = s['stop_times'][i]
            amplitude = s['amplitude']
            hemisphere = s['hemisphere']
            targets = s['targets'][i]
            cell_type = s['cell-type']

            stim = nest.Create("dc_generator", 1, {
                "start": start_time * scale, "stop": stop_time * scale, "amplitude": amplitude})

            if hemisphere == 'left':
                channels_in_hemi = left_hemisphere['channels']
            elif hemisphere == 'right':
                channels_in_hemi = right_hemisphere['channels']

            if targets[0][0] == 'all':

                # stimulate the whole hemisphere
                if cell_type == 'D1':
                    d1_neurons = list(np.ravel([channels_in_hemi[i]['d1'] for i in range(p.num_channels)]))
                    nest.Connect(stim, d1_neurons)
                elif cell_type == 'D2':
                    d2_neurons = list(np.ravel([channels_in_hemi[i]['d2'] for i in range(p.num_channels)]))
                    nest.Connect(stim, d2_neurons)

            else:

                # stimulate only specific channels
                for t in targets:
                    for chan in channels_in_hemi:
                        if (chan['row'] == t[0] and chan['col'] == t[1]):
                            # this channel is targeted

                            if cell_type == 'D1':
                                nest.Connect(stim, chan['d1'])
                            elif cell_type == 'D2':
                                nest.Connect(stim, chan['d2'])


proxy_out = nest.Create('music_event_out_proxy')
nest.SetStatus(proxy_out, {'port_name': 'out'})

for i in range(p.num_neurons_per_channel):
    nest.Connect([right_hemisphere['channels'][3 * p.grid_size[0][1] + 3]['d1'][i]], proxy_out, 'one_to_one', {     # Hardcoded for now, turn left is (3,3)
        'music_channel': i, 'delay': 1.0})

for i in range(p.num_neurons_per_channel):
    nest.Connect([right_hemisphere['channels'][3 * p.grid_size[0][1] + 3]['d2'][i]], proxy_out, 'one_to_one', {     # Hardcoded for now, turn left is (3,3)
        'music_channel': p.num_neurons_per_channel + i, 'delay': 1.0})

for i in range(p.num_neurons_per_channel):
    nest.Connect([left_hemisphere['channels'][3 * p.grid_size[0][1] + 4]['d1'][i]], proxy_out, 'one_to_one', {
                 'music_channel': 2 * p.num_neurons_per_channel + i, 'delay': 1.0})

for i in range(p.num_neurons_per_channel):
    nest.Connect([left_hemisphere['channels'][3 * p.grid_size[0][1] + 4]['d2'][i]], proxy_out, 'one_to_one', {
                 'music_channel': 3 * p.num_neurons_per_channel + i, 'delay': 1.0})

#####################################
##### JUST FOR TECHNICAL RESONS #####
#####################################

proxy_in = nest.Create('music_event_in_proxy', 1)
nest.SetStatus(proxy_in, [{'port_name': 'in',
                           'music_channel': c} for c in range(1)])
nest.SetAcceptableLatency('in', 1.0)  # useless?


#####################################
#####################################
#####################################

comm.Barrier()

t0 = time.time()
nest.Simulate(p.runtime * scale)

print "TIME ELAPSED: ", time.time() - t0, "rtf", p.runtime * scale / (1000 * (time.time() - t0))
