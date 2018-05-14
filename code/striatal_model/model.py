#!/usr/bin/python
import numpy as np
import time
from params import *
import nest
import sys
import yaml

from mpi4py import MPI
comm = MPI.COMM_WORLD

with open("cfg.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


seed = [np.random.randint(0, 9999999)] * num_threads
print "seed", seed
nest.ResetKernel()
nest.SetKernelStatus({"resolution": timestep, "overwrite_files": True,
                      "rng_seeds": seed, "print_time": False, "local_num_threads": num_threads})

nest.set_verbosity("M_FATAL")

# copy models

nest.CopyModel("iaf_cond_alpha", "d1", d1_params)
nest.CopyModel("iaf_cond_alpha", "d2", d2_params)

# nest.CopyModel("dc_generator", "stimulus", params={
#               "start": start, "stop": stop, "amplitude": stim_amplitude})


# Background noise
noise_d1 = nest.Create("poisson_generator", 1, {"rate": bg_noise_d1})
noise_d2 = nest.Create("poisson_generator", 1, {"rate": bg_noise_d2})


def create_hemisphere(label):
    d1_turnleft = nest.Create('d1', num_neurons_per_channel)
    d2_turnleft = nest.Create('d2', num_neurons_per_channel)
    d1_turnright = nest.Create('d1', num_neurons_per_channel)
    d2_turnright = nest.Create('d2', num_neurons_per_channel)

    # Connect within channels
    nest.Connect(d1_turnleft, d1_turnleft,
                 conn_params_d1d1_within_chan, syn_params_d1d1_within_chan)
    nest.Connect(d1_turnleft, d2_turnleft,
                 conn_params_d1d2_within_chan, syn_params_d1d2_within_chan)
    nest.Connect(d2_turnleft, d1_turnleft,
                 conn_params_d2d1_within_chan, syn_params_d2d1_within_chan)
    nest.Connect(d2_turnleft, d2_turnleft,
                 conn_params_d2d2_within_chan, syn_params_d2d2_within_chan)

    nest.Connect(d1_turnright, d1_turnright,
                 conn_params_d1d1_within_chan, syn_params_d1d1_within_chan)
    nest.Connect(d1_turnright, d2_turnright,
                 conn_params_d1d2_within_chan, syn_params_d1d2_within_chan)
    nest.Connect(d2_turnright, d1_turnright,
                 conn_params_d2d1_within_chan, syn_params_d2d1_within_chan)
    nest.Connect(d2_turnright, d2_turnright,
                 conn_params_d2d2_within_chan, syn_params_d2d2_within_chan)

    # Connection between channels
    nest.Connect(d1_turnleft, d1_turnright,
                 conn_params_d1d1_between_chan, syn_params_d1d1_between_chan)
    nest.Connect(d1_turnleft, d2_turnright,
                 conn_params_d1d2_between_chan, syn_params_d1d2_between_chan)
    nest.Connect(d2_turnleft, d1_turnright,
                 conn_params_d2d1_between_chan, syn_params_d2d1_between_chan)
    nest.Connect(d2_turnleft, d2_turnright,
                 conn_params_d2d2_between_chan, syn_params_d2d2_between_chan)

    nest.Connect(d1_turnright, d1_turnleft,
                 conn_params_d1d1_between_chan, syn_params_d1d1_between_chan)
    nest.Connect(d1_turnright, d2_turnleft,
                 conn_params_d1d2_between_chan, syn_params_d1d2_between_chan)
    nest.Connect(d2_turnright, d1_turnleft,
                 conn_params_d2d1_between_chan, syn_params_d2d1_between_chan)
    nest.Connect(d2_turnright, d2_turnleft,
                 conn_params_d2d2_between_chan, syn_params_d2d2_between_chan)

    nest.Connect(noise_d1, d1_turnleft + d1_turnright, syn_spec={
                 "weight": bg_weight_d1, "delay": 1.})
    nest.Connect(noise_d2, d2_turnleft + d2_turnright, syn_spec={
                 "weight": bg_weight_d2, "delay": 1.})

    sd = nest.Create("spike_detector", 1, {
                     "to_file": True, 'label': label, "use_gid_in_filename": False})

    nest.Connect(d1_turnleft + d2_turnleft + d1_turnright + d2_turnright, sd)

    return {"d1_turnleft": d1_turnleft, "d2_turnleft": d2_turnleft, "d1_turnright": d1_turnright, "d2_turnright": d2_turnright, "spike_detector": sd}


# CREATE HEMISPHERES

left_hemisphere = create_hemisphere("../../data/left_hemisphere")
right_hemisphere = create_hemisphere("../../data/right_hemisphere")


for s in cfg["stim-params"]:
    stim = nest.Create("dc_generator", 1, {
                       "start": start, "stop": stop, "amplitude": s["amplitude"]})
    if s["hemisphere"] == "left":
        if s["cell-type"] == "D1":
            nest.Connect(stim, left_hemisphere["d1_turnleft"] +
                         left_hemisphere["d1_turnright"], syn_spec={"weight": bg_weight_d1})
        if s["cell-type"] == "D2":
            nest.Connect(stim, left_hemisphere["d2_turnleft"] +
                         left_hemisphere["d2_turnright"], syn_spec={"weight": bg_weight_d2})
    if s["hemisphere"] == "right":
        if s["cell-type"] == "D1":
            nest.Connect(stim, right_hemisphere["d1_turnleft"] +
                         right_hemisphere["d1_turnright"], syn_spec={"weight": bg_weight_d1})
        if s["cell-type"] == "D2":
            nest.Connect(stim, right_hemisphere["d2_turnleft"] +
                         right_hemisphere["d2_turnright"], syn_spec={"weight": bg_weight_d2})


proxy_out = nest.Create('music_event_out_proxy')
nest.SetStatus(proxy_out, {'port_name': 'out'})

for i in range(100):
    nest.Connect([right_hemisphere["d1_turnleft"][i]], proxy_out, 'all_to_all', {
                 'music_channel': i, 'delay': 1.0})  # to_ms(options.music_timestep)})
for i in range(100):
    nest.Connect([left_hemisphere["d1_turnright"][i]], proxy_out, 'all_to_all', {
                 'music_channel': 100 + i, 'delay': 1.0})  # to_ms(options.music_timestep)})


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
nest.Simulate(runtime)

print "TIME ELAPSED: ", time.time() - t0, "rtf", runtime / (1000 * (time.time() - t0))
