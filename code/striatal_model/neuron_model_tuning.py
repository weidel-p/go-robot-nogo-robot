import nest
import pylab as pl
import pickle
from nest import voltage_trace
from nest import raster_plot as rplt
import numpy as np

from params import *

seed = [np.random.randint(0, 9999999)] * num_threads


def calcFI():

    #amplitudesList = np.arange(3.5,4.5,0.1)
    amplitudesList = np.arange(100, 500, 50.)

    listD1 = []
    listD2 = []

    for amp in amplitudesList:

        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": timestep, "overwrite_files": True, "rng_seeds": seed,
                              "print_time": True, "local_num_threads": num_threads})

        nest.CopyModel("iaf_cond_alpha", "d1", d1_params)
        #nest.CopyModel("izhikevich", "d1", d1_params_iz)
        nest.CopyModel("iaf_cond_alpha", "d2", d2_params)
        #nest.CopyModel("izhikevich", "d2", d2_params_iz)

        d1 = nest.Create("d1", 1)
        d2 = nest.Create("d2", 1)
        dc = nest.Create("dc_generator", 1)
        sd = nest.Create("spike_detector", 2)
        mult = nest.Create("multimeter", 1, params={
                           "withgid": True, "withtime": True, "record_from": ["V_m"]})

        nest.Connect(d1, [sd[0]])
        nest.Connect(d2, [sd[1]])
        nest.Connect(dc, d1)
        nest.Connect(dc, d2)
        nest.Connect(mult, d1)
        nest.Connect(mult, d2)

        nest.SetStatus(dc, params={"amplitude": amp})
        nest.Simulate(10000.)
        evs_d1 = nest.GetStatus([sd[0]])[0]["events"]["senders"]
        ts_d1 = nest.GetStatus([sd[0]])[0]["events"]["times"]

        evs_d2 = nest.GetStatus([sd[1]])[0]["events"]["senders"]
        ts_d2 = nest.GetStatus([sd[1]])[0]["events"]["times"]

        listD1.append(len(ts_d1) / 10.0)
        listD2.append(len(ts_d2) / 10.0)

        # voltage_trace.from_device(mult)
        # pl.show()

    FI = dict()
    FI["d1"] = listD1
    FI["d2"] = listD2
    pickle.dump(FI, open("../../data/FI.pickle", "w"))
    pl.figure()
    pl.text(70, 62, "A", fontweight='bold', fontsize=15)
    pl.plot(amplitudesList, listD1, 'bo-', label='D1', linewidth=1.5)
    pl.plot(amplitudesList, listD2, 'go-', label='D2', linewidth=1.5)
    pl.legend(loc='best')
    pl.xlabel("Amplitude(pA)", fontweight='bold', fontsize=14)
    pl.ylabel("Firing rate (sps)", fontweight='bold', fontsize=14)
    for x in pl.gca().get_xticklabels():
        x.set_fontweight('bold')
        x.set_fontsize(10)
    for x in pl.gca().get_yticklabels():
        x.set_fontweight('bold')
        x.set_fontsize(10)

    pl.savefig("../../data/FI.pdf")

    print "d1", FI["d1"], "d2", FI["d2"], amplitudesList
    pl.figure()
    voltage_trace.from_device(mult)
    pl.show()


def checkConninMV():
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": timestep, "overwrite_files": True, "rng_seeds": seed,
                          "print_time": True, "local_num_threads": num_threads})

    nest.CopyModel("iaf_cond_alpha", "d21", d2_params)
    #nest.CopyModel("izhikevich", "d1", d1_params_iz)
    nest.CopyModel("iaf_cond_alpha", "d22", d2_params)
    #nest.CopyModel("izhikevich", "d2", d2_params_iz)

    d21 = nest.Create("d21", 1)
    d22 = nest.Create("d22", 1)
    nest.SetStatus(d22, {'I_e': 27.})  # Has to be tuned so that d2 is at -80
    # nest.SetStatus(d1,{'I_e':69.}) # Has to be tuned so that d1 is at -80
    dc = nest.Create("dc_generator", 1)
    sd = nest.Create("spike_detector", 2)
    mult = nest.Create("multimeter", 1, params={
                       "withgid": True, "withtime": True, "record_from": ["V_m"]})

    nest.Connect(d21, [sd[0]])
    nest.Connect(d22, [sd[1]])
    nest.Connect(dc, d21)
    # nest.Connect(dc,d2)
    # nest.Connect(mult,d1)
    nest.Connect(mult, d22)
    nest.Connect(d21, d22, syn_spec={'weight': jd2d2})
    nest.SetStatus(dc, params={"amplitude": 250.})

    nest.Simulate(1000.)
    evs_d1 = nest.GetStatus([sd[0]])[0]["events"]["senders"]
    ts_d1 = nest.GetStatus([sd[0]])[0]["events"]["times"]

    V_m = nest.GetStatus(mult)[0]["events"]["V_m"]
    ts = nest.GetStatus(mult)[0]["events"]["times"]
    inds = np.where(ts > 400.)
    Vmmin = np.min(V_m[inds])
    print "conn_strength", Vmmin + 80.
    # pl.figure(1)
    # rplt.from_device(sd)
    pl.figure(2)
    voltage_trace.from_device(mult)
    pl.plot(ts_d1, np.ones(len(ts_d1)) * -80., 'r|', markersize=10)
    pl.show()


calcFI()
# checkConninMV()
