import numpy as np


#########################
### SIMULATION PARAMS ###
#########################

timestep = 1.
runtime = 20000.

num_threads = 9


######################
### NETWORK PARAMS ###
######################

bg_noise_d1 = 95.
bg_noise_d2 = 57.


bg_weight_d1 = 2.5
bg_weight_d2 = 2.5


## NEURONS ##

num_neurons_per_channel = 40

# Channels are now aranged on an imaginary grid. The size of the grid determines the number of channels

grid_size = [[6, 6]]  # 0th element = row, 1st element = column
num_channels = grid_size[0][0] * grid_size[0][1]

d1_params = {"V_m": -87.2,
             "E_L": -87.2,
             "V_reset": -87.2,  # Not specified, but check bottom left figure in Gertler Fig 2. vreset > V_rest
             "V_th": -50.,
             "g_L": 9.,
             "C_m": 195.,
             "tau_syn_in": 10.0,
             "tau_syn_ex": 5.0
             }

d2_params = {"V_m": -85.4,
             "E_L": -85.4,
             "V_reset": -85.4,
             "V_th": -50.,
             "g_L": 4.5,
             "C_m": 159.,
             "tau_syn_in": 10.0,
             "tau_syn_ex": 5.0
             }


## CONNECTIVITY ##

# Mimicing a distance dependent kernel: TL ---> gen---> TR; TR---> gen; TR-> TL; gen---> TL, TL -> TR ; ---> strong inhibition, -> weak inhibition
# Since now the number of neurons / channel have reduced to scale the connectivity, since now far connectivity is completely ineffective:
scaleConn = 2.0
withinChanConnPropScaling = 0.6 * scaleConn  # These scaling numbers for within, far and near channels are calculated from Lope-Huerta 2013, Fig 6, on the basis of number of connected neurons depending on size. Within ~ 5, hence the average of connected D1/D2 neurons within channel is ~ 5, similarly,  near channel ~ 15, far channel ~ 1

betweenNearChanConnPropScaling = 1.7 * scaleConn
betweenFarChanConnPropScaling = 0.15 * scaleConn

withinChanDelay = 1.0
betweenNearChanDelay = 2.5
betweenFarChanDelay = 4.5

cd1d1 = 0.07 * num_neurons_per_channel  # Planert
cd1d2 = 0.05 * num_neurons_per_channel
cd2d1 = 0.13 * num_neurons_per_channel
cd2d2 = 0.23 * num_neurons_per_channel  # In planert
cd1fsi = 0.89
cd2fsi = 0.67
weightScale = 0.9   # Re-tuned below according to iaf_cond_alpha and Table 1 in Planert, see script neuorn_model_tuning.py
jd1d1 = -0.75 * weightScale  # Had to be tuned so that mV match the planert data, we did forget that these were tuned for aeif_exp
jd1d2 = -0.85 * weightScale
jd2d1 = -1.7 * weightScale  # also depends on neuron properties
jd2d2 = -1.35 * weightScale

c_inter_hemis = 0.1
j_inter_hemis = -0.5
conn_params_d1d1_within_chan = {'rule': 'fixed_outdegree', 'outdegree': int(cd1d1 * withinChanConnPropScaling)}
conn_params_d1d2_within_chan = {'rule': 'fixed_outdegree', 'outdegree': int(cd1d2 * withinChanConnPropScaling)}
conn_params_d2d1_within_chan = {'rule': 'fixed_outdegree', 'outdegree': int(cd2d1 * withinChanConnPropScaling)}
conn_params_d2d2_within_chan = {'rule': 'fixed_outdegree', 'outdegree': int(cd2d2 * withinChanConnPropScaling)}

conn_params_d1d1_between_near_chan = {'rule': 'fixed_outdegree',
                                      'outdegree': int(cd1d1 * betweenNearChanConnPropScaling)}
conn_params_d1d2_between_near_chan = {'rule': 'fixed_outdegree',
                                      'outdegree': int(cd1d2 * betweenNearChanConnPropScaling)}
conn_params_d2d1_between_near_chan = {'rule': 'fixed_outdegree',
                                      'outdegree': int(cd2d1 * betweenNearChanConnPropScaling)}
conn_params_d2d2_between_near_chan = {'rule': 'fixed_outdegree',
                                      'outdegree': int(cd2d2 * betweenNearChanConnPropScaling)}

conn_params_d1d1_between_far_chan = {'rule': 'fixed_outdegree', 'outdegree': int(cd1d1 * betweenFarChanConnPropScaling)}
conn_params_d1d2_between_far_chan = {'rule': 'fixed_outdegree', 'outdegree': int(cd1d2 * betweenFarChanConnPropScaling)}
conn_params_d2d1_between_far_chan = {'rule': 'fixed_outdegree', 'outdegree': int(cd2d1 * betweenFarChanConnPropScaling)}
conn_params_d2d2_between_far_chan = {'rule': 'fixed_outdegree', 'outdegree': int(cd2d2 * betweenFarChanConnPropScaling)}

syn_params_d1d1_within_chan = {"weight": jd1d1, "delay": withinChanDelay}
syn_params_d1d2_within_chan = {"weight": jd1d2, "delay": withinChanDelay}
syn_params_d2d1_within_chan = {"weight": jd2d1, "delay": withinChanDelay}
syn_params_d2d2_within_chan = {"weight": jd2d2, "delay": withinChanDelay}

syn_params_d1d1_between_near_chan = {"weight": jd1d1, "delay": betweenNearChanDelay}
syn_params_d1d2_between_near_chan = {"weight": jd1d2, "delay": betweenNearChanDelay}
syn_params_d2d1_between_near_chan = {"weight": jd2d1, "delay": betweenNearChanDelay}
syn_params_d2d2_between_near_chan = {"weight": jd2d2, "delay": betweenNearChanDelay}

syn_params_d1d1_between_far_chan = {"weight": jd1d1, "delay": betweenFarChanDelay}
syn_params_d1d2_between_far_chan = {"weight": jd1d2, "delay": betweenFarChanDelay}
syn_params_d2d1_between_far_chan = {"weight": jd2d1, "delay": betweenFarChanDelay}
syn_params_d2d2_between_far_chan = {"weight": jd2d2, "delay": betweenFarChanDelay}
