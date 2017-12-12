import nest
import pylab as pl
import pickle
from nest import voltage_trace
from nest import raster_plot as rplt
import numpy as np
import sys
sys.path.append("../analysis")
import colors

from params import *

seed = [np.random.randint(0, 9999999)] * num_threads


def calcFI():

    amplitudesList = np.arange(100, 500, 50.)

    listD1 = []
    listD2 = []

    for amp in amplitudesList:

        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": timestep, "overwrite_files": True, "rng_seeds": seed,
                              "print_time": True, "local_num_threads": num_threads})

        nest.CopyModel("iaf_cond_alpha", "d1", d1_params)
        nest.CopyModel("iaf_cond_alpha", "d2", d2_params)

        d1 = nest.Create("d1", 1)
        d2 = nest.Create("d2", 1)
        dc = nest.Create("dc_generator", 1)
        sd = nest.Create("spike_detector", 2)
        mult = nest.Create("multimeter", 1, params={"withgid": True, "withtime": True, "record_from": ["V_m"]})

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


    FI = dict()
    FI["d1"] = listD1
    FI["d2"] = listD2
    pl.figure(figsize=[16, 10])
    pl.text(70, 62, "A", fontweight='bold', fontsize=60)
    pl.plot(amplitudesList, listD1, 'o-', color=colors.colors[1], label='D1')
    pl.plot(amplitudesList, listD2, 'o-', color=colors.colors[2], label='D2')
    pl.legend(loc='best')
    pl.xlabel("Amplitude(pA)", fontweight='bold')
    pl.ylabel("Firing rate (spks/s)", fontweight='bold')
    pl.ylim([-1, 60])
    for x in pl.gca().get_xticklabels():
        x.set_fontweight('bold')
    for x in pl.gca().get_yticklabels():
        x.set_fontweight('bold')

    pl.savefig("FI.pdf")

    print "d1", FI["d1"], "d2", FI["d2"], amplitudesList
    pl.figure()
    voltage_trace.from_device(mult)
    pl.show()


calcFI()
