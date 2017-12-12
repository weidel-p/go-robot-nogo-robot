import numpy as np
import pylab as pl
import matplotlib.cm as cm
import os
import pickle
import pdb
from nest import raster_plot as rplt
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), '..', 'striatal_model/'))
import params as p
import json
import matplotlib.pyplot as plt
from itertools import combinations
import itertools
import yaml
from pylab import *
import seaborn


colors = ["#1A2017", "#540D6E", "#992A44", "#336492", "#D9B43D"]
seaborn.set_palette(colors)
seaborn.set_context('paper', font_scale=1.0, rc={"lines.linewidth": 1.5})
seaborn.set_style('whitegrid', {"axes.linewidth": 0.5})


def plotDistVsAct(DistData, SpikesData):
    print "layer1", DistData["name1"]
    print "layer2", DistData["name2"]
    DistVsRate = []
    for i, x in enumerate(DistData["layer1_ids"]):
        for j, y in enumerate(DistData["layer2_ids"]):
            ind = np.where(SpikesData["evs"] == y)
            ts = SpikesData["ts"][ind]
            tsIp = np.logical_and(ts >= start, ts <= stop)
            DistVsRate.append(
                [DistData["Dist"][i][j], len(tsIp) / ((stop - start) * 0.001)])
    pl.figure()
    pl.title(DistData["name1"] + " vs " + DistData["name2"])
    pl.xlabel("Distance")
    pl.ylabel("Firing rate, Spikes/sec")
    pl.plot(np.array(DistVsRate)[:, 0], np.array(DistVsRate)[:, 1], 'b.')
    pl.savefig("DistVsRate.png")


def findDist(layer1, layer2, name1, name2):
    nodesL1 = nest.GetLeaves(layer1)[0]
    nodesL2 = nest.GetLeaves(layer2)[0]
    DistL1L2 = np.zeros((len(nodesL1), len(nodesL2)))
    for i, x in enumerate(nodesL1):
        for j, y in enumerate(nodesL2):
            # node1 and node2 are connected
            if len(nest.GetConnections([x], [y])) > 0:
                p1 = topp.GetPosition([x])[0]
                p2 = topp.GetPosition([y])[0]
                DistL1L2[i][j] = findEuclidean(p1, p2)

    ids_Dist = dict()
    ids_Dist["layer1_ids"] = nodesL1
    ids_Dist["name1"] = name1
    ids_Dist["layer2_ids"] = nodesL2
    ids_Dist["name2"] = name2
    ids_Dist["Dist"] = DistL1L2
    return ids_Dist


def findEuclidean(p1, p2):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def plotTimeDepAct(senders, times, pos, path):
    activity = []
    time = np.arange(0, 20, 1.)
    window = np.exp(- time / 2.)

    u_ids = pos[:, 0]

    # Extract activity neuron wise
    for j, y in enumerate(u_ids):
        # Rows == neurons, columns == time series
        activity.append(times[np.where(senders == y)[0]])

    # For each neuron, find the PSTH with bin size == 100msec, reduce the simtime to 1000msec
    spike_count = []
    binsize = 20.
    binning = np.arange(0, runtime, binsize)

    for x in activity:
        height, edges = np.histogram(x, bins=binning)
        spike_count.append(np.convolve(height, window, mode='same'))

    spike_count = np.array(spike_count)
    # Because we want to take mean of each neurons across the time points 0:start/binsize. So the shape of this vector is 1600
    mean_activity = np.mean(spike_count[:, :int(start / binsize)], 1)

    for i in xrange(np.shape(spike_count)[1]):
        spike_count[:, i] -= mean_activity

    # Now get positions for each neuron and plot the activity as heat map
    # 3D vector, 1st d = time, 2,3 grid, 20x20, 4thd==activity
    timeDepAct = np.zeros((len(height), rows_striatum, columns_striatum))

    if os.path.exists(path) == False:
        os.mkdir(path)

    files = os.listdir(path)
    for f in files:
        if os.path.exists(path + f):
            os.remove(path + f)

    for i in xrange(np.shape(timeDepAct)[0]):  # Iter over all time points
        # Because filtAct has rows == neuron, cols = time points
        timeDepAct[i] = np.reshape(
            spike_count[:, i], [rows_striatum, columns_striatum])

        pl.figure()
        pl.pcolor(timeDepAct[i], cmap=cm.RdBu_r, vmin=-2, vmax=2.)
        # cbar =
        pl.colorbar()
        pl.savefig(path + "/timeDepAct" + str(i) + ".jpg")
    pickle.dump(timeDepAct, open(path + "/timeDepAct.pickle", "w"))

    pl.close('all')


def plotAllData(Alldata, params):
    # Raster

    fig1 = pl.figure(figsize=(12, 6))
    restData = dict()
    subfigs = []
    #import pdb
    # pdb.set_trace()
    cols = ["r", "b", "g", "orange", "y", "m", "c",
            "k", "brown", "darkred", "gray", "darkgreen"]
    for i in xrange(len(Alldata)):
        subfigs.append(fig1.add_subplot(1, 3, i + 1))
        subfigs[-1].plot(Alldata[i]["ts"][::1], Alldata[i]["evs"][::1],
                         '.', label=Alldata[i]["nuc"], color=cols[i], markersize=4)
        subfigs[-1].legend()
        subfigs[-1].set_xlim(0, runtime)
        restData[Alldata[i]["nuc"]] = dict()
        restData[Alldata[i]["nuc"]]["ts"] = Alldata[i]["ts"]
        restData[Alldata[i]["nuc"]]["evs"] = Alldata[i]["evs"]

    # for i in xrange(len(Alldata)):
    fig1.savefig("Raster.png")

    return restData


def rasterPlot(fn):
    rplt.from_file(fn)
    # pl.savefig("{dir}/{fn}.pdf".format(dir=dirname, fn=fn)) #dirname+keys+"_"+hemis+".pdf")
    # rplt.show()


def calcCorrCoef(gids, ch_sp_s1, ch_sp_t1, ch_sp_s2, ch_sp_t2, ch1_mean, ch2_mean, start, stop):
    binsize = 200.
    binning = []
    for st1, st2 in zip(start, stop):
        temp = np.arange(st1, st2, binsize)
        for x in temp:
            binning.append(x)
#    mean1signal = np.histogram(ch1_mean,bins = binning)[0]/float(p.num_neurons_per_channel*p.num_channels) # Since it is subtracted from spike count not in sps units, but spike count per neuron
#    mean2signal = np.histogram(ch2_mean,bins = binning)[0]/float(p.num_neurons_per_channel*p.num_channels)
#    time = np.arange(0,10,1.)
#    expKern = np.exp(-time/10.)
#    mean1signal = np.convolve(mean1signal,expKern)
#    mean2signal = np.convolve(mean2signal,expKern)

    # print "mean1signal",mean1signal
    # print "mean2signal",mean2signal
    # print "binning",binning
    corrcoefs = []
    for n in gids:
        st11 = ch_sp_t1[np.where(n[0] == ch_sp_s1)[0]]  # Spike train 1
        st21 = ch_sp_t2[np.where(n[1] == ch_sp_s2)[0]]  # Spike train 2
        a1, b1 = np.histogram(st11, bins=binning)
        a2, b2 = np.histogram(st21, bins=binning)
        # print "a1",a1
        # print "a2",a2
        # print "a1-mean1signal",a1-mean1signal
#         a1 = np.convolve(a1,expKern)
#         a2 = np.convolve(a2,expKern)
#         a1Norm = (a1-mean1signal)/mean1signal
#         a1Norm[np.where(np.isnan(a1Norm)==True)] = 0
#         a2Norm = (a2-mean2signal)/mean2signal
#         a2Norm[np.where(np.isnan(a2Norm)==True)] = 0
        # Delta F / F, here dividing by mean1signal for even a singke point makes the whole corrcoef nan,maybe this can be fixed with filtering the spikes, for now removing dividing by mean1signal
        corrcoefs.append(np.corrcoef(a1, a2)[0][1])
    # print "corrcoefs",corrcoefs
    # print "mean1signal",mean1signal
    # print "a1",a1
    # print "a1-mean1signal/mean1signal",(a1-mean1signal)/mean1signal
    corrcoefs = np.array(corrcoefs)[np.where(np.isnan(np.array(corrcoefs)) == False)]

    return corrcoefs


def getCorrAtDist(allCorrs, alldist, dist):
    inds = np.where(np.logical_and(np.array(alldist) >= dist, np.array(alldist) < dist + 1) == True)[0]
    temp = []
    #inds1 = allCorrs[0][inds]
    for y in inds:
        for z in allCorrs[y]:
            temp.append(z)

    return temp


def subFigsFormat(i1, subfigHands):
    if i1 % 6 != 0:
        for x in subfigHands.get_yticklabels():
            x.set_visible(False)
    else:
        for x in subfigHands.get_yticklabels():
            x.set_fontsize(6)
            x.set_fontweight('bold')
        for x in subfigHands.get_yticklabels()[1::2]:
            x.set_visible(False)
    if i1 <= 29:
        for x in subfigHands.get_xticklabels():
            x.set_visible(False)
    else:
        for x in subfigHands.get_xticklabels():
            x.set_fontsize(6)
            x.set_fontweight('bold')
        for x in subfigHands.get_xticklabels()[0::2]:
            x.set_visible(False)

    subfigHands.set_xlim(-0.6, 1.)
    # subfigHands.set_ylim(0,8.5)
    # subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
    subfigHands.legend(prop={'size': 4, "weight": 'bold'})


def stdTicklabelsFormat(ticklabels):    # Set the fontsize, fontweight as bold and visible only every second ticklabel
    for x in ticklabels():
        x.set_fontweight('bold')
        x.set_fontsize(10)

    for x in ticklabels()[::2]:
        x.set_visible(False)


def channelCorrelations(fn, dirname, hemis, expName):  # calculates the pair wise correlations within and between channels

    #rand_channelCorrelations(fn, dirname, hemis, expName)

    spike_data = np.loadtxt(fn)
    all_senders = spike_data[:, 0]
    all_spike_times = spike_data[:, 1]
    print os.getcwd()
    filename = "../striatal_model/experiments/" + expName
    # This is to separate the times of activity with and without external input
    if expName == 'sequences.yaml' or expName == 'sequencesd1d2.yaml' or expName == 'competingActions.yaml':
        with open(filename, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        if hemis == 'left_hemisphere':
            start_times = cfg['stim-params'][1]['start_times']
            stop_times = cfg['stim-params'][1]['stop_times']
        else:
            start_times = cfg['stim-params'][2]['start_times']  # Right hemisphere has different stimulation times
            stop_times = cfg['stim-params'][2]['stop_times']

    else:  # If any other experiment, there is only 1 start and stop time as specified in params.py
        start_times = [p.start]
        stop_times = [p.stop]
    if expName == 'sequencesMultTrials.yaml':
        return  # Just returns for sequencesMultTrials

    if hemis == 'left_hemisphere':
        with open(dirname + "neuron_ids_left_hemisphere.json", "r+") as f:
            hemisphere_neuron_ids = json.load(f)
    else:
        with open(dirname + "neuron_ids_right_hemisphere.json", "r+") as f:
            hemisphere_neuron_ids = json.load(f)

    fhand = file(dirname + "allD1Spikes.gdf", 'r')
    all_d1_spikes = np.loadtxt(fhand)
    fhand.close()

    fhand = file(dirname + "allD2Spikes.gdf", 'r')
    all_d2_spikes = np.loadtxt(fhand)
    fhand.close()

    hists = []
    binsize = 200.  # Correlations in Klaus et. al. calculated with binsize 100-150ms
    binsizeInSecs = binsize / 1000.
    binning = np.arange(0, p.runtime, binsize)

    binCorr = np.arange(-1.0, 1.2, 0.1)
    channel_id = 0
    # To simulate a sense of distance each channel stretches over a distance of ~ 40mu metre
    # 50 pair wise correlations are considered

    # Plot D1-D1, D2-D2 and D1-D2 corrcoefs within each channel and between each channels
    fig1ip = pl.figure()  # Within channels
    fig1woip = pl.figure()  # Within channels
    fig2ip = pl.figure()  # Between channels during ip
    fig2woip = pl.figure()  # Between channels without ip
    fig1ip.suptitle("Corrcoef distributions within Channels during stimulation")
    fig1woip.suptitle("Corrcoef distributions within Channels without stimulation")
    fig2ip.suptitle("Corrcoef distributions between Channels during stimulation")
    fig2woip.suptitle("Corrcoef distributions between Channels without stimulation")
    subfigHands1ip = []
    subfigHands1woip = []
    subfigHands2ip = []
    subfigHands2woip = []
    Correlations = dict()
    for i, channel in enumerate(hemisphere_neuron_ids['channels']):
        Correlations[i] = dict()
        Correlations[i]['row'] = channel['row']
        Correlations[i]['col'] = channel['col']
        channel_n_ids1 = channel['d1']
        channel_n_ids2 = channel['d2']
        channel_spike_data1 = np.array([])
        channel_spike_data2 = np.array([])

        # get spikes for this channel
        mask1 = np.hstack([np.where(n_id == all_senders)[0] for n_id in channel_n_ids1])
        mask2 = np.hstack([np.where(n_id == all_senders)[0] for n_id in channel_n_ids2])
        channel_spike_senders1 = all_senders[mask1]
        channel_spike_senders2 = all_senders[mask2]
        channel_spike_times1 = all_spike_times[mask1]
        channel_spike_times2 = all_spike_times[mask2]
        # Pick 50 neuron paiirs for calculating correlation coefficients
        numPairs = 100
        gids1 = np.array([comb for comb in combinations(channel_n_ids1, 2)])
        np.random.shuffle(gids1)
        gids1 = gids1[:numPairs]     # To prevent picking up the same pair again
        gids2 = np.array([comb for comb in combinations(channel_n_ids2, 2)])
        np.random.shuffle(gids2)
        gids2 = gids2[:numPairs]
        gids12 = np.array([comb for comb in itertools.product(channel_n_ids1, channel_n_ids2)])
        np.random.shuffle(gids12)
        gids12 = gids12[:numPairs]
        # print gids12

        inds1ip = []
        inds1woip = []
        mean1ip = []
        mean1woip = []
        inds2ip = []
        mean2ip = []
        inds2woip = []
        mean2woip = []
        # for start,stop in zip(start_times,stop_times):
        timeswoIpstart = []
        timeswoIpstop = []
        if len(start_times) == 1 and len(stop_times) == 1:
            start = start_times[0]
            stop = stop_times[0]
            inds1ip.append(np.where(np.logical_and(channel_spike_times1 >= start, channel_spike_times1 <= stop) == True)[
                           0])  # When input is present for D1
            inds2ip.append(np.where(np.logical_and(channel_spike_times2 >= start, channel_spike_times2 <= stop) == True)[
                           0])  # When input is present for D2
            #mean1ip.append(np.where(np.logical_and(all_d1_spikes >=start,all_d1_spikes <= stop) == True)[0])
            #mean2ip.append(np.where(np.logical_and(all_d2_spikes >=start,all_d2_spikes <= stop) == True)[0])
            inds1woip.append(np.where(np.logical_or(channel_spike_times1 < start,
                                                    channel_spike_times1 > stop) == True)[0])  # When input is absent
            # mean1woip.append(np.where(np.logical_or(all_d1_spikes <start,all_d1_spikes > stop) == True)[0]) # When input is absent
            inds2woip.append(np.where(np.logical_or(channel_spike_times2 <
                                                    start, channel_spike_times2 > stop) == True)[0])
            timeswoIpstart = [0, stop]
            timeswoIpstop = [start, p.runtime]
            # mean2woip.append(np.where(np.logical_or(all_d2_spikes <start,all_d2_spikes > stop) == True)[0]) # When input is absent
        else:
            start1 = start_times[0]
            start2 = start_times[1]
            stop1 = stop_times[0]
            stop2 = stop_times[1]
            inds1ip.append(np.where(np.logical_or(np.logical_and(channel_spike_times1 >= start1, channel_spike_times1 <= stop1),
                                                  np.logical_and(channel_spike_times1 >= start2, channel_spike_times1 <= stop2)) == True)[0])  # When input is present
            inds2ip.append(np.where(np.logical_or(np.logical_and(channel_spike_times2 >= start1, channel_spike_times2 <= stop1),
                                                  np.logical_and(channel_spike_times2 >= start2, channel_spike_times2 <= stop2)) == True)[0])  # When input is present
            #mean1ip.append(np.where(np.logical_or(np.logical_and(all_d1_spikes >=start1,all_d1_spikes <= stop1),np.logical_and(all_d1_spikes >=start2,all_d1_spikes <= stop2)) == True)[0])
            #mean2ip.append(np.where(np.logical_or(np.logical_and(all_d2_spikes >=start1,all_d2_spikes <= stop1),np.logical_and(all_d2_spikes >=start2,all_d2_spikes <= stop2)) == True)[0])

            inds1woip.append(np.where(np.logical_or(np.logical_or(channel_spike_times1 < start1, channel_spike_times1 > stop2),
                                                    np.logical_and(channel_spike_times1 > stop1, channel_spike_times1 < start2)) == True)[0])  # When input is absent
            inds2woip.append(np.where(np.logical_or(np.logical_or(channel_spike_times2 < start1, channel_spike_times2 > stop2),
                                                    np.logical_and(channel_spike_times2 > stop1, channel_spike_times2 < start2)) == True)[0])  # When input is absent
            timeswoIpstart = [0, stop1, stop2]
            timeswoIpstop = [start1, start2, p.runtime]
            #mean1woip.append(np.where(np.logical_or(np.logical_or(all_d1_spikes <start1,all_d1_spikes > stop2),np.logical_and(all_d1_spikes >stop1,all_d1_spikes<start2))    == True)[0])
            #mean2woip.append(np.where(np.logical_or(np.logical_or(all_d2_spikes <start1,all_d2_spikes > stop2),np.logical_and(all_d2_spikes >stop1,all_d2_spikes<start2))    == True)[0])
        # print "mean1ip",mean1ip
        # This mean signal should be subtracted from all neurons for corrcoef calculation
        # meanD1signalip = np.histogram(all_d1_spikes[mean1ip],bins = binning)[0]/float(p.num_neurons_per_channel*p.num_channels) # Since it is subtracted from spike count not in sps units, but spike count per neuron
        # meanD1signalwoip = np.histogram(all_d1_spikes[mean1woip],bins = binning)[0]/float(p.num_neurons_per_channel*p.num_channels) # Since it is subtracted from spike count not in sps units
        #meanD2signalip = np.histogram(all_d2_spikes[mean2ip],bins = binning)[0]/float(p.num_neurons_per_channel*p.num_channels)
        #meanD2signalwoip = np.histogram(all_d2_spikes[mean2woip],bins = binning)[0]/float(p.num_neurons_per_channel*p.num_channels)

        # print "meanD1signalip",meanD1signalip
        # print "meanD1signalwoip",meanD1signalwoip
        corrcoef11ip = np.array(calcCorrCoef(gids1, channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip], channel_spike_senders1[inds1ip],
                                             channel_spike_times1[inds1ip], all_d1_spikes[inds1ip], all_d1_spikes[inds1ip], start_times, stop_times))
        corrcoef11woip = np.array(calcCorrCoef(gids1, channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip], channel_spike_senders1[inds1woip],
                                               channel_spike_times1[inds1woip], all_d1_spikes[inds1woip], all_d1_spikes[inds1woip], timeswoIpstart, timeswoIpstop))

        #corrcoef11ip = np.array(calcCorrCoef(gids1,channel_spike_senders1[inds1ip],channel_spike_times1[inds1ip],channel_spike_senders1[inds1ip],channel_spike_times1[inds1ip],meanD1signalip,meanD1signalip))
        #corrcoef11woip = np.array(calcCorrCoef(gids1,channel_spike_senders1[inds1woip],channel_spike_times1[inds1woip],channel_spike_senders1[inds1woip],channel_spike_times1[inds1woip],meanD1signalwoip,meanD1signalwoip))

        #corrcoef22ip = np.array(calcCorrCoef(gids2,channel_spike_senders2[inds2ip],channel_spike_times2[inds2ip],channel_spike_senders2[inds2ip],channel_spike_times2[inds2ip],meanD2signalip,meanD2signalip))
        #corrcoef22woip = np.array(calcCorrCoef(gids2,channel_spike_senders2[inds2woip],channel_spike_times2[inds2woip],channel_spike_senders2[inds2woip],channel_spike_times2[inds2woip],meanD2signalwoip,meanD2signalwoip))

        corrcoef22ip = np.array(calcCorrCoef(gids2, channel_spike_senders2[inds2ip], channel_spike_times2[inds2ip], channel_spike_senders2[inds2ip],
                                             channel_spike_times2[inds2ip], all_d2_spikes[inds2ip], all_d2_spikes[inds2ip], start_times, stop_times))
        corrcoef22woip = np.array(calcCorrCoef(gids2, channel_spike_senders2[inds2woip], channel_spike_times2[inds2woip], channel_spike_senders2[inds2woip],
                                               channel_spike_times2[inds2woip], all_d2_spikes[inds2woip], all_d2_spikes[inds2woip], timeswoIpstart, timeswoIpstop))

        #corrcoef12ip = np.array(calcCorrCoef(gids12,channel_spike_senders1[inds1ip],channel_spike_times1[inds1ip],channel_spike_senders2[inds2ip],channel_spike_times2[inds2ip],meanD1signalip,meanD2signalip))
        #corrcoef12woip = np.array(calcCorrCoef(gids12,channel_spike_senders1[inds1woip],channel_spike_times1[inds1woip],channel_spike_senders2[inds2woip],channel_spike_times2[inds2woip],meanD1signalwoip,meanD2signalwoip))
        corrcoef12ip = np.array(calcCorrCoef(gids12, channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip], channel_spike_senders2[inds2ip],
                                             channel_spike_times2[inds2ip], all_d1_spikes[inds1ip], all_d2_spikes[inds2ip], start_times, stop_times))
        corrcoef12woip = np.array(calcCorrCoef(gids12, channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip], channel_spike_senders2[inds2woip],
                                               channel_spike_times2[inds2woip], all_d1_spikes[inds1woip], all_d2_spikes[inds2woip], timeswoIpstart, timeswoIpstop))

        # print corrcoef12
        a3, b3 = np.histogram(corrcoef11ip, bins=binCorr, normed=True)
        a4, b4 = np.histogram(corrcoef22ip, bins=binCorr, normed=True)
        a5, b5 = np.histogram(corrcoef12ip, bins=binCorr, normed=True)

        a3woip, b3woip = np.histogram(corrcoef11woip, bins=binCorr, normed=True)
        a4woip, b4woip = np.histogram(corrcoef22woip, bins=binCorr, normed=True)
        a5woip, b5woip = np.histogram(corrcoef12woip, bins=binCorr, normed=True)

        if hemis == 'left_hemisphere' and channel['row'] == 3 and channel['col'] == 4:
            subfigHands2ip.append(fig2ip.add_subplot(
                p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * 3 + 4 + 1))
            subfigHands2woip.append(fig2woip.add_subplot(
                p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * 3 + 4 + 1))
            subfigHands2ip[-1].plot(b3[:-1], a3, 'b-', linewidth=1.0, label='D1-D1')
            subfigHands2ip[-1].plot(b4[:-1], a4, 'g-', linewidth=1.0, label='D2-D2')
            subfigHands2ip[-1].plot(b5[:-1], a5, 'r-', linewidth=1.0, label='D1-D2')
            subfigHands2woip[-1].plot(b3woip[:-1], a3woip, 'b-', linewidth=1.0)
            # subfigHands2[-1].text(0.6,4,str(np.round(np.mean(corrcoef11_p),2)),fontsize=4,fontweight='bold',color='b')
            subfigHands2woip[-1].plot(b4woip[:-1], a4woip, 'g-', linewidth=1.0)
            # subfigHands2[-1].text(0.6,3.5,str(np.round(np.mean(corrcoef22_p),2)),fontsize=4,fontweight='bold',color='g')
            subfigHands2woip[-1].plot(b5woip[:-1], a5woip, 'r-', linewidth=1.0)

            # subfigHands2[-1].text(0.6,3,str(np.round(np.mean(corrcoef12_p),2)),fontsize=4,fontweight='bold',color='r')
            '''
            if i %6 != 0:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_yticklabels()[1::2]:
                    x.set_visible(False)               
            if i <=29:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_xticklabels()[0::2]:
                    x.set_visible(False)               
                   
            subfigHands2ip[-1].set_xlim(-0.6,1.)
            subfigHands2woip[-1].set_xlim(-0.6,1.)
            subfigHands2ip[-1].set_ylim(0,8.)
            subfigHands2woip[-1].set_ylim(0,8.)
            #subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
            subfigHands2ip[-1].legend(prop={'size':4,"weight":'bold'})
            subfigHands2woip[-1].legend(prop={'size':4,"weight":'bold'})
            '''
            subFigsFormat(i, subfigHands2ip[-1])
            subFigsFormat(i, subfigHands2woip[-1])

            # subfigHands2woip[-1].set_yscale('log')
            # subfigHands2ip[-1].set_yscale('log')
            # This channel is the one that gets input in sequences
        if hemis == 'right_hemisphere' and channel['row'] == 3 and channel['col'] == 3:
            subfigHands2ip.append(fig2ip.add_subplot(
                p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * 3 + 3 + 1))
            subfigHands2woip.append(fig2woip.add_subplot(
                p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * 3 + 3 + 1))
            subfigHands2ip[-1].plot(b3[:-1], a3, 'b-', linewidth=1.0, label='D1-D1')
            subfigHands2ip[-1].plot(b4[:-1], a4, 'g-', linewidth=1.0, label='D2-D2')
            subfigHands2ip[-1].plot(b5[:-1], a5, 'r-', linewidth=1.0, label='D1-D2')

            subfigHands2woip[-1].plot(b3woip[:-1], a3woip, 'b-', linewidth=1.0)
            # subfigHands2[-1].text(0.6,4,str(np.round(np.mean(corrcoef11_p),2)),fontsize=4,fontweight='bold',color='b')
            subfigHands2woip[-1].plot(b4woip[:-1], a4woip, 'g-', linewidth=1.0)
            # subfigHands2[-1].text(0.6,3.5,str(np.round(np.mean(corrcoef22_p),2)),fontsize=4,fontweight='bold',color='g')
            subfigHands2woip[-1].plot(b5woip[:-1], a5woip, 'r-', linewidth=1.0)

            # subfigHands2[-1].text(0.6,3,str(np.round(np.mean(corrcoef12_p),2)),fontsize=4,fontweight='bold',color='r')
            '''
            if i %6 != 0:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_yticklabels()[1::2]:
                    x.set_visible(False)               
            if i <=29:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_xticklabels()[0::2]:
                    x.set_visible(False)               
                   
            subfigHands2[-1].set_xlim(-1.0,1.)
            subfigHands2[-1].set_ylim(0,8.)
            subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
            subfigHands2[-1].legend(prop={'size':4,"weight":'bold'})
            '''
            subFigsFormat(i, subfigHands2ip[-1])
            # subfigHands2ip[-1].set_yscale('log')
            subFigsFormat(i, subfigHands2woip[-1])
            # subfigHands2woip[-1].set_yscale('log')

        # Making grid plot instead of all rows plot
        subfigHands1ip.append(fig1ip.add_subplot(p.grid_size[0][0] + 1, p.grid_size[0][1], i + 1))
        # Making grid plot instead of all rows plot
        subfigHands1woip.append(fig1woip.add_subplot(p.grid_size[0][0] + 1, p.grid_size[0][1], i + 1))
        subfigHands1ip[-1].plot(b3[:-1], a3, 'b-', linewidth=1.0, label='D1-D1')
        subfigHands1ip[-1].plot(b4[:-1], a4, 'g-', linewidth=1.0, label='D2-D2')
        subfigHands1ip[-1].plot(b5[:-1], a5, 'r-', linewidth=1.0, label='D1-D2')

        subfigHands1woip[-1].plot(b3woip[:-1], a3woip, 'b-', linewidth=1.0)
        # subfigHands1[-1].text(0.6,3,str(np.round(np.mean(corrcoef11_p),2)),fontsize=4,fontweight='bold',color='b')
        subfigHands1woip[-1].plot(b4woip[:-1], a4woip, 'g-', linewidth=1.0)
        # subfigHands1[-1].text(0.6,2.5,str(np.round(np.mean(corrcoef22_p),2)),fontsize=4,fontweight='bold',color='g')
        subfigHands1woip[-1].plot(b5woip[:-1], a5woip, 'r-', linewidth=1.0)

        # subfigHands1[-1].text(0.6,2.0,str(np.round(np.mean(corrcoef12_p),2)),fontsize=4,fontweight='bold',color='r')
        '''
        if i %6 != 0:
            for x in subfigHands1[-1].get_yticklabels():
                x.set_visible(False)
        else:
            for x in subfigHands1[-1].get_yticklabels():
                x.set_fontsize(6)
                x.set_fontweight('bold')
            for x in subfigHands1[-1].get_yticklabels()[1::2]:
                x.set_visible(False)               
        if i <=29:
            for x in subfigHands1[-1].get_xticklabels():
                x.set_visible(False)
        else:
            for x in subfigHands1[-1].get_xticklabels():
                x.set_fontsize(6)
                x.set_fontweight('bold')
            for x in subfigHands1[-1].get_xticklabels()[0::2]:
                x.set_visible(False)               
               
        subfigHands1[-1].set_xlim(-1.0,1.)
        subfigHands1[-1].set_ylim(0,8.)
        subfigHands1[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
        subfigHands1[-1].legend(prop={'size':4,"weight":'bold'})
        '''
        subFigsFormat(i, subfigHands1ip[-1])
        # subfigHands1ip[-1].set_yscale('log')

        subFigsFormat(i, subfigHands1woip[-1])
        # subfigHands1woip[-1].set_yscale('log')

        Correlations[i]['withinCorr_d1d1_ip'] = corrcoef11ip
        Correlations[i]['withinCorr_d1d1_woip'] = corrcoef11woip
        Correlations[i]['withinCorr_d2d2_ip'] = corrcoef22ip
        Correlations[i]['withinCorr_d2d2_woip'] = corrcoef22woip
        Correlations[i]['withinCorr_d1d2_ip'] = corrcoef12ip
        Correlations[i]['withinCorr_d1d2_woip'] = corrcoef12woip

        allBwCorrsd1d1ip = []
        allBwCorrsd1d1woip = []
        allBwCorrsd2d2ip = []
        allBwCorrsd2d2woip = []
        allBwCorrsd1d2ip = []
        allBwCorrsd1d2woip = []

        for j, ch1 in enumerate(hemisphere_neuron_ids['channels']):
            if ch1 == channel:
                continue    # We calculated within correlation above
            ch1_ids = ch1['d1']
            ch1_spike_data = np.array([])
            ch2_ids = ch1['d2']
            ch2_spike_data = np.array([])

            # get spikes for this channel
            mask1 = np.hstack([np.where(n_id == all_senders)[0] for n_id in ch1_ids])
            ch1_spike_senders = all_senders[mask1]
            ch1_spike_times = all_spike_times[mask1]
            mask2 = np.hstack([np.where(n_id == all_senders)[0] for n_id in ch2_ids])
            ch2_spike_senders = all_senders[mask2]
            ch2_spike_times = all_spike_times[mask2]

            # Pick 50 neuron pairs for calculating correlation coefficients
            gids11 = np.array([x for x in itertools.product(list(channel_n_ids1), list(ch1_ids))])
            np.random.shuffle(gids11)
            gids11 = gids11[:numPairs]
            gids22 = np.array([x for x in itertools.product(list(channel_n_ids2), list(ch2_ids))])
            np.random.shuffle(gids22)
            gids22 = gids22[:numPairs]
            gids33 = np.array([x for x in itertools.product(list(channel_n_ids1), list(ch2_ids))])
            np.random.shuffle(gids33)
            gids33 = gids33[:numPairs]
            # For channel 2 , convention, first number 1 is for D1/D2 and second number 2 is for channel
            inds12ip = []
            inds12woip = []
            inds22ip = []
            inds22woip = []
            # for start,stop in zip(start_times,stop_times):
            if len(start_times) == 1 and len(stop_times) == 1:
                start = start_times[0]
                stop = stop_times[0]
                inds12ip.append(np.where(np.logical_and(ch1_spike_times >= start, ch1_spike_times <= stop) == True)[
                                0])  # When input is present for D1
                inds22ip.append(np.where(np.logical_and(ch2_spike_times >= start, ch2_spike_times <= stop) == True)[
                                0])  # When input is present for D2
                inds12woip.append(np.where(np.logical_or(ch1_spike_times < start,
                                                         ch1_spike_times > stop) == True)[0])  # When input is absent
                inds22woip.append(np.where(np.logical_or(ch2_spike_times < start, ch2_spike_times > stop) == True)[0])
                # No need to recalculate mean here, since that remains same
            else:
                start1 = start_times[0]
                start2 = start_times[1]
                stop1 = stop_times[0]
                stop2 = stop_times[1]
                inds12ip.append(np.where(np.logical_or(np.logical_and(ch1_spike_times >= start1, ch1_spike_times <= stop1),
                                                       np.logical_and(ch1_spike_times >= start2, ch1_spike_times <= stop2)) == True)[0])  # When input is present
                inds22ip.append(np.where(np.logical_or(np.logical_and(ch2_spike_times >= start1, ch2_spike_times <= stop1),
                                                       np.logical_and(ch2_spike_times >= start2, ch2_spike_times <= stop2)) == True)[0])  # When input is present
                inds12woip.append(np.where(np.logical_or(np.logical_or(ch1_spike_times < start1, ch1_spike_times > stop2),
                                                         np.logical_and(ch1_spike_times > stop1, ch1_spike_times < start2)) == True)[0])  # When input is absent
                inds22woip.append(np.where(np.logical_or(np.logical_or(ch2_spike_times < start1, ch2_spike_times > stop2),
                                                         np.logical_and(ch2_spike_times > stop1, ch2_spike_times < start2)) == True)[0])  # When input is absent

            # bwcorrcoef11ip = np.array(calcCorrCoef(gids11,channel_spike_senders1[inds1ip],channel_spike_times1[inds1ip],ch1_spike_senders[inds12ip],ch1_spike_times[inds12ip],meanD1signalip,meanD1signalip)) #Corrcoef between D1 of "channel"(index i) with D1 of "ch1" (index j) when input is present
            # bwcorrcoef11woip = np.array(calcCorrCoef(gids11,channel_spike_senders1[inds1woip],channel_spike_times1[inds1woip],ch1_spike_senders[inds12woip],ch1_spike_times[inds12woip],meanD1signalwoip,meanD1signalwoip)) #Corrcoef between D1 of "channel"(index i) with D1 of "ch1" (index j) when input is absent

            # bwcorrcoef22ip = np.array(calcCorrCoef(gids22,channel_spike_senders2[inds2ip],channel_spike_times2[inds2ip],ch2_spike_senders[inds22ip],ch2_spike_times[inds22ip],meanD2signalip,meanD2signalip))#Corrcoef between D2 of "channel"(index i) with D2 of "ch1"(index j) when input is present
            # bwcorrcoef22woip = np.array(calcCorrCoef(gids22,channel_spike_senders2[inds2woip],channel_spike_times2[inds2woip],ch2_spike_senders[inds22woip],ch2_spike_times[inds22woip],meanD2signalwoip,meanD2signalwoip)) #Corrcoef between D2 of "channel"(index i) with D2 of "ch1" when input is absent

            # bwcorrcoef33ip = np.array(calcCorrCoef(gids33,channel_spike_senders1[inds1ip],channel_spike_times1[inds1ip],ch2_spike_senders[inds22ip],ch2_spike_times[inds22ip],meanD1signalip,meanD2signalip))#Corrcoef between D1 of "channel" with D2 of "ch1" when input is present
            # bwcorrcoef33woip = np.array(calcCorrCoef(gids33,channel_spike_senders1[inds1woip],channel_spike_times1[inds1woip],ch2_spike_senders[inds22woip],ch2_spike_times[inds22woip],meanD1signalwoip,meanD2signalwoip))#Corrcoef between D1 of "channel" with D2 of "ch1" when input is absent

            bwcorrcoef11ip = np.array(calcCorrCoef(gids11, channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip], ch1_spike_senders[inds12ip], ch1_spike_times[inds12ip],
                                                   all_d1_spikes[inds1ip], all_d1_spikes[inds1ip], start_times, stop_times))  # Corrcoef between D1 of "channel"(index i) with D1 of "ch1" (index j) when input is present
            bwcorrcoef11woip = np.array(calcCorrCoef(gids11, channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip], ch1_spike_senders[inds12woip], ch1_spike_times[inds12woip], all_d1_spikes[
                                        inds1woip], all_d1_spikes[inds1woip], timeswoIpstart, timeswoIpstop))  # Corrcoef between D1 of "channel"(index i) with D1 of "ch1" (index j) when input is absent

            bwcorrcoef22ip = np.array(calcCorrCoef(gids22, channel_spike_senders2[inds2ip], channel_spike_times2[inds2ip], ch2_spike_senders[inds22ip], ch2_spike_times[inds22ip],
                                                   all_d2_spikes[inds2ip], all_d2_spikes[inds2ip], start_times, stop_times))  # Corrcoef between D2 of "channel"(index i) with D2 of "ch1"(index j) when input is present
            bwcorrcoef22woip = np.array(calcCorrCoef(gids22, channel_spike_senders2[inds2woip], channel_spike_times2[inds2woip], ch2_spike_senders[inds22woip], ch2_spike_times[inds22woip],
                                                     all_d2_spikes[inds2woip], all_d2_spikes[inds2woip], timeswoIpstart, timeswoIpstop))  # Corrcoef between D2 of "channel"(index i) with D2 of "ch1" when input is absent

            bwcorrcoef33ip = np.array(calcCorrCoef(gids33, channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip], ch2_spike_senders[inds22ip], ch2_spike_times[
                                      inds22ip], all_d1_spikes[inds1ip], all_d2_spikes[inds2ip], start_times, stop_times))  # Corrcoef between D1 of "channel" with D2 of "ch1" when input is present
            bwcorrcoef33woip = np.array(calcCorrCoef(gids33, channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip], ch2_spike_senders[inds22woip], ch2_spike_times[
                                        inds22woip], all_d1_spikes[inds1woip], all_d2_spikes[inds2woip], timeswoIpstart, timeswoIpstop))  # Corrcoef between D1 of "channel" with D2 of "ch1" when input is absent

            if (hemis == 'left_hemisphere' and channel['row'] == 3 and channel['col'] == 4) or (hemis == 'right_hemisphere' and channel['row'] == 3 and channel['col'] == 3):

                subfigHands2ip.append(fig2ip.add_subplot(
                    p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * ch1['row'] + ch1['col'] + 1))
                subfigHands2woip.append(fig2woip.add_subplot(
                    p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * ch1['row'] + ch1['col'] + 1))
                # Correlation between D1-D1, D1-D2 and D2-D2 between other channels and the one that got input in seqeuences paradigm
                a3, b3 = np.histogram(bwcorrcoef11ip, bins=binCorr, normed=True)
                a4, b4 = np.histogram(bwcorrcoef22ip, bins=binCorr, normed=True)
                a5, b5 = np.histogram(bwcorrcoef33ip, bins=binCorr, normed=True)

                a3woip, b3woip = np.histogram(bwcorrcoef11woip, bins=binCorr, normed=True)
                a4woip, b4woip = np.histogram(bwcorrcoef22woip, bins=binCorr, normed=True)
                a5woip, b5woip = np.histogram(bwcorrcoef33woip, bins=binCorr, normed=True)

                subfigHands2ip[-1].plot(b3[:-1], a3, 'b-', linewidth=1.0, label='D1-D1', alpha=0.5)
                subfigHands2ip[-1].plot(b4[:-1], a4, 'g-', linewidth=1.0, label='D2-D2', alpha=0.5)
                subfigHands2ip[-1].plot(b5[:-1], a5, 'r-', linewidth=1.0, label='D1-D2', alpha=0.5)

                subfigHands2woip[-1].plot(b3woip[:-1], a3woip, 'b-', linewidth=1.0, alpha=0.5)
                # subfigHands2[-1].text(0.6,4,str(np.round(np.mean(bwcorrcoef11_p),2)),fontsize=4,fontweight='bold',color='b')
                subfigHands2woip[-1].plot(b4woip[:-1], a4woip, 'g-', linewidth=1.0, alpha=0.5)
                # subfigHands2[-1].text(0.6,3.5,str(np.round(np.mean(bwcorrcoef22_p),2)),fontsize=4,fontweight='bold',color='g')
                subfigHands2woip[-1].plot(b5woip[:-1], a5woip, 'r-', linewidth=1.0, alpha=0.5)
                # subfigHands2[-1].text(0.6,3,str(np.round(np.mean(bwcorrcoef33_p),2)),fontsize=4,fontweight='bold',color='r')
                '''
                if i %6 != 0:
                    for x in subfigHands2[-1].get_yticklabels():
                        x.set_visible(False)
                else:
                    for x in subfigHands2[-1].get_yticklabels():
                        x.set_fontsize(6)
                        x.set_fontweight('bold')
                    for x in subfigHands2[-1].get_yticklabels()[1::2]:
                        x.set_visible(False)               
                if i <=29:
                    for x in subfigHands2[-1].get_xticklabels():
                        x.set_visible(False)
                else:
                    for x in subfigHands2[-1].get_xticklabels():
                        x.set_fontsize(6)
                        x.set_fontweight('bold')
                    for x in subfigHands2[-1].get_xticklabels()[0::2]:
                        x.set_visible(False)               
                       
                subfigHands2[-1].set_xlim(-1.0,1.)
                subfigHands2[-1].set_ylim(0,8.)
                subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
                subfigHands2[-1].legend(prop={'size':4,"weight":'bold'})
                '''
                subFigsFormat(j, subfigHands2ip[-1])
                # subfigHands2ip[-1].set_yscale('log')
                subFigsFormat(j, subfigHands2woip[-1])
                # subfigHands2woip[-1].set_yscale('log')

                print "j", j

            # 0th element in temp is distance, 1st = d1-d1,2nd = d2-d2, 3rd d1-d2
            # First column is distance, the other 50 columns are corrcoefs
            temp11ip = np.zeros((1, len(bwcorrcoef11ip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp11woip = np.zeros((1, len(bwcorrcoef11woip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp22ip = np.zeros((1, len(bwcorrcoef22ip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp22woip = np.zeros((1, len(bwcorrcoef22woip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp12ip = np.zeros((1, len(bwcorrcoef33ip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp12woip = np.zeros((1, len(bwcorrcoef33woip) + 1))

            temp11ip[0][0] = temp11woip[0][0] = temp22ip[0][0] = temp22woip[0][0] = temp12ip[0][0] = temp12woip[0][0] = np.sqrt(
                (channel['row'] - ch1['row'])**2 + (channel['col'] - ch1['col'])**2)   # Storing the distance of this channel combination
            # Consider wrapping in calculating distance
            if (np.abs(channel['row'] - ch1['row']) == 5) or (np.abs(channel['col'] - ch1['col']) == 5):
                temp11ip[0][0] = temp11woip[0][0] = temp22ip[0][0] = temp22woip[0][0] = temp12ip[0][0] = temp12woip[0][0] = 1.

            temp11ip[0][1:] = bwcorrcoef11ip
            temp11woip[0][1:] = bwcorrcoef11woip
            allBwCorrsd1d1ip.append(temp11ip)
            allBwCorrsd1d1woip.append(temp11woip)

            temp22ip[0][1:] = bwcorrcoef22ip
            temp22woip[0][1:] = bwcorrcoef22woip
            allBwCorrsd2d2ip.append(temp22ip)
            allBwCorrsd2d2woip.append(temp22woip)

            temp12ip[0][1:] = bwcorrcoef33ip
            temp12woip[0][1:] = bwcorrcoef33woip
            allBwCorrsd1d2ip.append(temp12ip)
            allBwCorrsd1d2woip.append(temp12woip)

            '''
            a3,b3 = np.histogram(bwcorrcoef11,bins=binCorr,normed=True)    
            a4,b4 = np.histogram(bwcorrcoef22,bins=binCorr,normed=True)    
            a5,b5 = np.histogram(bwcorrcoef33,bins=binCorr,normed=True)    

            subfigHands2.append(fig2.add_subplot(p.grid_size[0][0]+1, p.grid_size[0][1], i + 1)) # Making grid plot instead of all rows plot
            subfigHands2[-1].plot(b3[:-1],a3,'b-',linewidth=1.5,label='D1-D1')
            subfigHands2[-1].plot(b4[:-1],a4,'g-',linewidth=1.5,label='D2-D2')
            subfigHands2[-1].plot(b5[:-1],a5,'r-',linewidth=1.5,label='D1-D2')

            if i %6 != 0:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_yticklabels()[1::2]:
                    x.set_visible(False)               
            if i <=29:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_xticklabels()[0::2]:
                    x.set_visible(False)               
                   
            subfigHands2[-1].set_xlim(-0.6,1.)
            subfigHands2[-1].set_ylim(0,6.)
            subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
            subfigHands2[-1].legend(prop={'size':4,"weight":'bold'})
            '''

        Correlations[i]['bwCorr_d1d1_ip'] = allBwCorrsd1d1ip
        Correlations[i]['bwCorr_d1d1_woip'] = allBwCorrsd1d1woip
        Correlations[i]['bwCorr_d2d2_ip'] = allBwCorrsd2d2ip
        Correlations[i]['bwCorr_d2d2_woip'] = allBwCorrsd2d2woip
        Correlations[i]['bwCorr_d1d2_ip'] = allBwCorrsd1d2ip
        Correlations[i]['bwCorr_d1d2_woip'] = allBwCorrsd1d2woip

    # Save the data first
    # with open(dirname + "corrcoefs.json", "w+") as f: # json gave an error figure out later , save as pickle for now
    '''
    with open(dirname + "corrcoefs.pickle", "w") as f:
        pickle.dump(Correlations,f) 
    '''
    # Now plot
    fig = pl.figure()
    corrsVsDist = []
    # Numpy.histogram does not work on different size arrays in a list, hence converting them to list
    allWithinCorrsd1d1ip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                            for x in Correlations[i]['withinCorr_d1d1_ip']]
    allWithinCorrsd1d1woip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                              for x in Correlations[i]['withinCorr_d1d1_woip']]
    allWithinCorrsd2d2ip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                            for x in Correlations[i]['withinCorr_d2d2_ip']]
    allWithinCorrsd2d2woip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                              for x in Correlations[i]['withinCorr_d2d2_woip']]
    allWithinCorrsd1d2ip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                            for x in Correlations[i]['withinCorr_d1d2_ip']]
    allWithinCorrsd1d2woip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                              for x in Correlations[i]['withinCorr_d1d2_woip']]
    allBetCorrsd1d1ip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                         for x in Correlations[i]['bwCorr_d1d1_ip']]  # 1st column is the distance)
    distd1d1ip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                  for x in Correlations[i]['bwCorr_d1d1_ip']]  # 1st column is the distance)
    # for i in xrange(len(hemisphere_neuron_ids['channels'])):
    #   print "bwCorr_d1d1_woip",Correlations[i]['bwCorr_d1d1_woip']
    #   print "bwCorr_d1d1_ip",Correlations[i]['bwCorr_d1d1_ip']

    allBetCorrsd1d1woip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                           for x in Correlations[i]['bwCorr_d1d1_woip']]  # 1st column is the distance)
    distd1d1woip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                    for x in Correlations[i]['bwCorr_d1d1_woip']]  # 1st column is the distance)

    allBetCorrsd2d2ip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                         for x in Correlations[i]['bwCorr_d2d2_ip']]  # 1st column is the distance)
    distd2d2ip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                  for x in Correlations[i]['bwCorr_d2d2_ip']]  # 1st column is the distance)

    allBetCorrsd2d2woip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                           for x in Correlations[i]['bwCorr_d2d2_woip']]  # 1st column is the distance)
    distd2d2woip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                    for x in Correlations[i]['bwCorr_d2d2_woip']]  # 1st column is the distance)

    allBetCorrsd1d2ip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                         for x in Correlations[i]['bwCorr_d1d2_ip']]  # 1st column is the distance)
    distd1d2ip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                  for x in Correlations[i]['bwCorr_d1d2_ip']]  # 1st column is the distance)

    allBetCorrsd1d2woip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                           for x in Correlations[i]['bwCorr_d1d2_woip']]  # 1st column is the distance)
    distd1d2woip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                    for x in Correlations[i]['bwCorr_d1d2_woip']]  # 1st column is the distance)

    # corrsVsDist.append(allWithinCorrs) # 0 distance
    a11ip, b11ip = np.histogram(allWithinCorrsd1d1ip, bins=binCorr, normed=True)
    a11woip, b11woip = np.histogram(allWithinCorrsd1d1woip, bins=binCorr, normed=True)
    a22ip, b22ip = np.histogram(allWithinCorrsd2d2ip, bins=binCorr, normed=True)
    a22woip, b22woip = np.histogram(allWithinCorrsd2d2woip, bins=binCorr, normed=True)
    a12ip, b12ip = np.histogram(allWithinCorrsd1d2ip, bins=binCorr, normed=True)
    a12woip, b12woip = np.histogram(allWithinCorrsd1d2woip, bins=binCorr, normed=True)
    ax11 = fig.add_subplot(321)
    ax12 = fig.add_subplot(322)
    ax11.set_title("D1-D1 (Stimulation)", fontsize=10, fontweight='bold')
    ax12.set_title("D1-D1 (Background)", fontsize=10, fontweight='bold')
    ax21 = fig.add_subplot(323)
    ax22 = fig.add_subplot(324)
    ax21.set_title("D2-D2 (Stimulation)", fontsize=10, fontweight='bold')
    ax22.set_title("D2-D2 (Background)", fontsize=10, fontweight='bold')
    ax31 = fig.add_subplot(325)
    ax32 = fig.add_subplot(326)
    ax31.set_title("D1-D2 (Stimulation)", fontsize=10, fontweight='bold')
    ax32.set_title("D1-D2 (Background)", fontsize=10, fontweight='bold')

    ax11.plot(b11ip[:-1], a11ip, 'm-', label='0-40', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax12.plot(b11woip[:-1], a11woip, 'm-', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax21.plot(b22ip[:-1], a22ip, 'm-', label='0-40', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax22.plot(b22woip[:-1], a22woip, 'm-', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax31.plot(b12ip[:-1], a12ip, 'm-', label='0-40', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax32.plot(b12woip[:-1], a12woip, 'm-', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    # Find all unique distances
    uniqDist = np.unique(distd1d1ip)  # Any one of the dist arrays will do
    # print uniqDist
    maxDist = 4  # Because maximum distance is sqrt(32)
    for i in xrange(1, maxDist + 1, 1):       # Too crowded, so sampling every 2 distance

        if i % 1 == 0:
            tempd1d1ip = getCorrAtDist(allBetCorrsd1d1ip, distd1d1ip, i)
            tempd2d2ip = getCorrAtDist(allBetCorrsd2d2ip, distd2d2ip, i)
            tempd1d2ip = getCorrAtDist(allBetCorrsd1d2ip, distd1d2ip, i)
            tempd1d1woip = getCorrAtDist(allBetCorrsd1d1woip, distd1d1woip, i)
            tempd2d2woip = getCorrAtDist(allBetCorrsd2d2woip, distd2d2woip, i)
            tempd1d2woip = getCorrAtDist(allBetCorrsd1d2woip, distd1d2woip, i)
            #temp = np.array(temp)[np.where(np.isnan(np.array(temp))==False)]
            # corrsVsDist.append(temp)    # Next dist
            ad1d1ip, bd1d1ip = np.histogram(tempd1d1ip, bins=binCorr, normed=True)
            ad2d2ip, bd2d2ip = np.histogram(tempd2d2ip, bins=binCorr, normed=True)
            ad1d2ip, bd1d2ip = np.histogram(tempd1d2ip, bins=binCorr, normed=True)
            ad1d1woip, bd1d1woip = np.histogram(tempd1d1woip, bins=binCorr, normed=True)
            ad2d2woip, bd2d2woip = np.histogram(tempd2d2woip, bins=binCorr, normed=True)
            ad1d2woip, bd1d2woip = np.histogram(tempd1d2woip, bins=binCorr, normed=True)

            ax11.plot(bd1d1ip[:-1], ad1d1ip, '-', label=str(i * 40) + "-" + str((i + 1) * 40),
                      linewidth=1.5, color=cm.copper_r((i * 1) / float(maxDist)), alpha=0.5)

            ax12.plot(bd1d1woip[:-1], ad1d1woip, '-', linewidth=1.5, color=cm.copper_r(i / float(maxDist)), alpha=0.5)

            ax21.plot(bd2d2ip[:-1], ad2d2ip, '-', label=str(i * 40) + "-" + str((i + 1) * 40),
                      linewidth=1.5, color=cm.copper_r((i * 1) / float(maxDist)), alpha=0.5)

            ax22.plot(bd2d2woip[:-1], ad2d2woip, '-', linewidth=1.5, color=cm.copper_r(i / float(maxDist)), alpha=0.5)

            ax31.plot(bd1d2ip[:-1], ad1d2ip, '-', label=str(i * 40) + "-" + str((i + 1) * 40),
                      linewidth=1.5, color=cm.copper_r((i * 1) / float(maxDist)), alpha=0.5)

            ax32.plot(bd1d2woip[:-1], ad1d2woip, '-', linewidth=1.5, color=cm.copper_r(i / float(maxDist)), alpha=0.5)

    ax31.set_xlabel("CorrCoef")
    ax31.set_ylabel("PDF")

    ax11.set_xlim(-0.6, 1.0)
    ax12.set_xlim(-0.6, 1.0)
    ax21.set_xlim(-0.6, 1.0)
    ax22.set_xlim(-0.6, 1.0)
    ax31.set_xlim(-0.6, 1.0)
    ax32.set_xlim(-0.6, 1.0)
    '''
    ax11.set_ylim(0,12.0)
    ax12.set_ylim(0,12.0)
    ax21.set_ylim(0,12.0)
    ax22.set_ylim(0,12.0)
    ax31.set_ylim(0,12.0)
    ax32.set_ylim(0,12.0)
    '''
    ax11.legend(prop={'size': 6, 'weight': 'bold'})
    ax21.legend(prop={'size': 6, 'weight': 'bold'})
    ax31.legend(prop={'size': 6, 'weight': 'bold'})
    for x in ax11.get_xticklabels():
        x.set_visible(False)
    stdTicklabelsFormat(ax11.get_yticklabels)

    for x in ax12.get_xticklabels():
        x.set_visible(False)
    stdTicklabelsFormat(ax12.get_yticklabels)
    stdTicklabelsFormat(ax21.get_yticklabels)
    stdTicklabelsFormat(ax22.get_yticklabels)
    stdTicklabelsFormat(ax31.get_yticklabels)
    stdTicklabelsFormat(ax32.get_yticklabels)
    stdTicklabelsFormat(ax31.get_xticklabels)
    stdTicklabelsFormat(ax32.get_xticklabels)

    for x in ax21.get_xticklabels():
        x.set_visible(False)
    for x in ax22.get_xticklabels():
        x.set_visible(False)
    '''
    ax11.set_yscale('log')
    ax12.set_yscale('log')

    ax21.set_yscale('log')
    ax22.set_yscale('log')

    ax31.set_yscale('log')
    ax32.set_yscale('log')
    '''

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle("Pairwise corr coef Vs dist")
    fig.savefig(dirname + "D1CorrCoefs_" + hemis + ".pdf")
    fig1ip.savefig(dirname + "CorrCoefs_Within_Channels_Stim_" + hemis + ".pdf")
    fig1woip.savefig(dirname + "CorrCoefs_Within_Channels_WoStim_" + hemis + ".pdf")
    fig2ip.savefig(dirname + "CorrCoefs_Between_Channels_Stim_" + hemis + ".pdf")
    fig2woip.savefig(dirname + "CorrCoefs_Between_Channels_WoStim_" + hemis + ".pdf")


# calculates the pair wise correlations within and between channels
def rand_channelCorrelations(fn, dirname, hemis, expName):
    spike_data = np.loadtxt(fn)
    all_senders = spike_data[:, 0]
    all_spike_times = spike_data[:, 1]
    print os.getcwd()
    filename = "../striatal_model/experiments/" + expName
    # This is to separate the times of activity with and without external input
    if expName == 'sequences.yaml' or expName == 'sequencesd1d2.yaml' or expName == 'competingActions.yaml':
        with open(filename, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        start_times = cfg['stim-params'][1]['start_times']
        stop_times = cfg['stim-params'][1]['stop_times']
    else:  # If any other experiment, there is only 1 start and stop time as specified in params.py
        start_times = [p.start]
        stop_times = [p.stop]
    if expName == 'sequencesMultTrials.yaml':
        return  # Just returns for sequencesMultTrials

    if hemis == 'left_hemisphere':
        with open(dirname + "neuron_ids_left_hemisphere.json", "r+") as f:
            hemisphere_neuron_ids = json.load(f)
    else:
        with open(dirname + "neuron_ids_right_hemisphere.json", "r+") as f:
            hemisphere_neuron_ids = json.load(f)

    fhand = file(dirname + "allD1Spikes.gdf", 'r')
    all_d1_spikes = np.loadtxt(fhand)
    fhand.close()

    fhand = file(dirname + "allD2Spikes.gdf", 'r')
    all_d2_spikes = np.loadtxt(fhand)
    fhand.close()

    hists = []
    binsize = 100.  # Correlations in Klaus et. al. calculated with binsize 100-150ms
    binsizeInSecs = binsize / 1000.
    binning = np.arange(0, p.runtime, binsize)

    binCorr = np.arange(-1.0, 1.2, 0.1)
    channel_id = 0
    # To simulate a sense of distance each channel stretches over a distance of ~ 40mu metre
    # 50 pair wise correlations are considered

    # Plot D1-D1, D2-D2 and D1-D2 corrcoefs within each channel and between each channels
    fig1ip = pl.figure()  # Within channels
    fig1woip = pl.figure()  # Within channels
    fig2ip = pl.figure()  # Between channels during ip
    fig2woip = pl.figure()  # Between channels without ip
    fig1ip.suptitle("Corrcoef distributions within Channels during stimulation")
    fig1woip.suptitle("Corrcoef distributions within Channels without stimulation")
    fig2ip.suptitle("Corrcoef distributions between Channels during stimulation")
    fig2woip.suptitle("Corrcoef distributions between Channels without stimulation")
    subfigHands1ip = []
    subfigHands1woip = []
    subfigHands2ip = []
    subfigHands2woip = []
    Correlations = dict()
    for i, channel in enumerate(hemisphere_neuron_ids['channels']):
        Correlations[i] = dict()
        Correlations[i]['row'] = channel['row']
        Correlations[i]['col'] = channel['col']
        channel_n_ids1 = channel['d1']
        channel_n_ids2 = channel['d2']
        channel_spike_data1 = np.array([])
        channel_spike_data2 = np.array([])

        # get spikes for this channel
        mask1 = np.hstack([np.where(n_id == all_senders)[0] for n_id in channel_n_ids1])
        mask2 = np.hstack([np.where(n_id == all_senders)[0] for n_id in channel_n_ids2])
        channel_spike_senders1 = all_senders[mask1]
        channel_spike_senders2 = all_senders[mask2]
        channel_spike_times1 = all_spike_times[mask1]
        channel_spike_times2 = all_spike_times[mask2]
        # Pick 50 neuron paiirs for calculating correlation coefficients
        numPairs = 100
        gids1 = np.array([comb for comb in combinations(channel_n_ids1, 2)])
        np.random.shuffle(gids1)
        gids1 = gids1[:numPairs]     # To prevent picking up the same pair again
        gids2 = np.array([comb for comb in combinations(channel_n_ids2, 2)])
        np.random.shuffle(gids2)
        gids2 = gids2[:numPairs]
        gids12 = np.array([comb for comb in itertools.product(channel_n_ids1, channel_n_ids2)])
        np.random.shuffle(gids12)
        gids12 = gids12[:numPairs]
        # print gids12

        inds1ip = []
        inds1woip = []
        mean1ip = []
        mean1woip = []
        inds2ip = []
        mean2ip = []
        inds2woip = []
        mean2woip = []
        # for start,stop in zip(start_times,stop_times):
        if len(start_times) == 1 and len(stop_times) == 1:
            start = start_times[0]
            stop = stop_times[0]
            inds1ip.append(np.where(np.logical_and(channel_spike_times1 >= start, channel_spike_times1 <= stop) == True)[
                           0])  # When input is present for D1
            inds2ip.append(np.where(np.logical_and(channel_spike_times2 >= start, channel_spike_times2 <= stop) == True)[
                           0])  # When input is present for D2
            #mean1ip.append(np.where(np.logical_and(all_d1_spikes >=start,all_d1_spikes <= stop) == True)[0])
            #mean2ip.append(np.where(np.logical_and(all_d2_spikes >=start,all_d2_spikes <= stop) == True)[0])
            inds1woip.append(np.where(np.logical_or(channel_spike_times1 < start,
                                                    channel_spike_times1 > stop) == True)[0])  # When input is absent
            # mean1woip.append(np.where(np.logical_or(all_d1_spikes <start,all_d1_spikes > stop) == True)[0]) # When input is absent
            inds2woip.append(np.where(np.logical_or(channel_spike_times2 <
                                                    start, channel_spike_times2 > stop) == True)[0])
            # mean2woip.append(np.where(np.logical_or(all_d2_spikes <start,all_d2_spikes > stop) == True)[0]) # When input is absent
        else:
            start1 = start_times[0]
            start2 = start_times[1]
            stop1 = stop_times[0]
            stop2 = stop_times[1]
            inds1ip.append(np.where(np.logical_or(np.logical_and(channel_spike_times1 >= start1, channel_spike_times1 <= stop1),
                                                  np.logical_and(channel_spike_times1 >= start2, channel_spike_times1 <= stop2)) == True)[0])  # When input is present
            inds2ip.append(np.where(np.logical_or(np.logical_and(channel_spike_times2 >= start1, channel_spike_times2 <= stop1),
                                                  np.logical_and(channel_spike_times2 >= start2, channel_spike_times2 <= stop2)) == True)[0])  # When input is present
            #mean1ip.append(np.where(np.logical_or(np.logical_and(all_d1_spikes >=start1,all_d1_spikes <= stop1),np.logical_and(all_d1_spikes >=start2,all_d1_spikes <= stop2)) == True)[0])
            #mean2ip.append(np.where(np.logical_or(np.logical_and(all_d2_spikes >=start1,all_d2_spikes <= stop1),np.logical_and(all_d2_spikes >=start2,all_d2_spikes <= stop2)) == True)[0])

            inds1woip.append(np.where(np.logical_or(np.logical_or(channel_spike_times1 < start1, channel_spike_times1 > stop2),
                                                    np.logical_and(channel_spike_times1 > stop1, channel_spike_times1 < start2)) == True)[0])  # When input is absent
            inds2woip.append(np.where(np.logical_or(np.logical_or(channel_spike_times2 < start1, channel_spike_times2 > stop2),
                                                    np.logical_and(channel_spike_times2 > stop1, channel_spike_times2 < start2)) == True)[0])  # When input is absent
            #mean1woip.append(np.where(np.logical_or(np.logical_or(all_d1_spikes <start1,all_d1_spikes > stop2),np.logical_and(all_d1_spikes >stop1,all_d1_spikes<start2))    == True)[0])
            #mean2woip.append(np.where(np.logical_or(np.logical_or(all_d2_spikes <start1,all_d2_spikes > stop2),np.logical_and(all_d2_spikes >stop1,all_d2_spikes<start2))    == True)[0])
        # print "mean1ip",mean1ip
        # This mean signal should be subtracted from all neurons for corrcoef calculation
        # Since it is subtracted from spike count not in sps units, but spike count per neuron
        meanD1signalip = np.histogram(all_d1_spikes[mean1ip], bins=binning)[
            0] / float(p.num_neurons_per_channel * p.num_channels)
        meanD1signalwoip = np.histogram(all_d1_spikes[mean1woip], bins=binning)[0] / float(p.num_neurons_per_channel *
                                                                                           p.num_channels)  # Since it is subtracted from spike count not in sps units
        meanD2signalip = np.histogram(all_d2_spikes[mean2ip], bins=binning)[
            0] / float(p.num_neurons_per_channel * p.num_channels)
        meanD2signalwoip = np.histogram(all_d2_spikes[mean2woip], bins=binning)[
            0] / float(p.num_neurons_per_channel * p.num_channels)

        print "meanD1signalip", meanD1signalip
        print "meanD1signalwoip", meanD1signalwoip
        corrcoef11ip = np.array(calcCorrCoef(gids1, channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip],
                                             channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip], meanD1signalip, meanD1signalip))
        corrcoef11woip = np.array(calcCorrCoef(gids1, channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip],
                                               channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip], meanD1signalwoip, meanD1signalwoip))

        corrcoef22ip = np.array(calcCorrCoef(gids2, channel_spike_senders2[inds2ip], channel_spike_times2[inds2ip],
                                             channel_spike_senders2[inds2ip], channel_spike_times2[inds2ip], meanD2signalip, meanD2signalip))
        corrcoef22woip = np.array(calcCorrCoef(gids2, channel_spike_senders2[inds2woip], channel_spike_times2[inds2woip],
                                               channel_spike_senders2[inds2woip], channel_spike_times2[inds2woip], meanD2signalwoip, meanD2signalwoip))

        corrcoef12ip = np.array(calcCorrCoef(gids12, channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip],
                                             channel_spike_senders2[inds2ip], channel_spike_times2[inds2ip], meanD1signalip, meanD2signalip))
        corrcoef12woip = np.array(calcCorrCoef(gids12, channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip],
                                               channel_spike_senders2[inds2woip], channel_spike_times2[inds2woip], meanD1signalwoip, meanD2signalwoip))

        # print corrcoef12
        a3, b3 = np.histogram(corrcoef11ip, bins=binCorr, normed=True)
        a4, b4 = np.histogram(corrcoef22ip, bins=binCorr, normed=True)
        a5, b5 = np.histogram(corrcoef12ip, bins=binCorr, normed=True)

        a3woip, b3woip = np.histogram(corrcoef11woip, bins=binCorr, normed=True)
        a4woip, b4woip = np.histogram(corrcoef22woip, bins=binCorr, normed=True)
        a5woip, b5woip = np.histogram(corrcoef12woip, bins=binCorr, normed=True)

        if hemis == 'left_hemisphere' and channel['row'] == 3 and channel['col'] == 4:
            subfigHands2ip.append(fig2ip.add_subplot(
                p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * 3 + 4 + 1))
            subfigHands2woip.append(fig2woip.add_subplot(
                p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * 3 + 4 + 1))
            subfigHands2ip[-1].plot(b3[:-1], a3, 'b-', linewidth=1.0, label='D1-D1')
            subfigHands2ip[-1].plot(b4[:-1], a4, 'g-', linewidth=1.0, label='D2-D2')
            subfigHands2ip[-1].plot(b5[:-1], a5, 'r-', linewidth=1.0, label='D1-D2')
            subfigHands2woip[-1].plot(b3woip[:-1], a3woip, 'b-', linewidth=1.0)
            # subfigHands2[-1].text(0.6,4,str(np.round(np.mean(corrcoef11_p),2)),fontsize=4,fontweight='bold',color='b')
            subfigHands2woip[-1].plot(b4woip[:-1], a4woip, 'g-', linewidth=1.0)
            # subfigHands2[-1].text(0.6,3.5,str(np.round(np.mean(corrcoef22_p),2)),fontsize=4,fontweight='bold',color='g')
            subfigHands2woip[-1].plot(b5woip[:-1], a5woip, 'r-', linewidth=1.0)

            # subfigHands2[-1].text(0.6,3,str(np.round(np.mean(corrcoef12_p),2)),fontsize=4,fontweight='bold',color='r')
            '''
            if i %6 != 0:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_yticklabels()[1::2]:
                    x.set_visible(False)               
            if i <=29:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_xticklabels()[0::2]:
                    x.set_visible(False)               
                   
            subfigHands2ip[-1].set_xlim(-0.6,1.)
            subfigHands2woip[-1].set_xlim(-0.6,1.)
            subfigHands2ip[-1].set_ylim(0,8.)
            subfigHands2woip[-1].set_ylim(0,8.)
            #subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
            subfigHands2ip[-1].legend(prop={'size':4,"weight":'bold'})
            subfigHands2woip[-1].legend(prop={'size':4,"weight":'bold'})
            '''
            subFigsFormat(i, subfigHands2ip[-1])
            subFigsFormat(i, subfigHands2woip[-1])

            # subfigHands2woip[-1].set_yscale('log')
            # subfigHands2ip[-1].set_yscale('log')
            # This channel is the one that gets input in sequences
        if hemis == 'right_hemisphere' and channel['row'] == 3 and channel['col'] == 3:
            subfigHands2ip.append(fig2ip.add_subplot(
                p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * 3 + 3 + 1))
            subfigHands2woip.append(fig2woip.add_subplot(
                p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * 3 + 3 + 1))
            subfigHands2ip[-1].plot(b3[:-1], a3, 'b-', linewidth=1.0, label='D1-D1')
            subfigHands2ip[-1].plot(b4[:-1], a4, 'g-', linewidth=1.0, label='D2-D2')
            subfigHands2ip[-1].plot(b5[:-1], a5, 'r-', linewidth=1.0, label='D1-D2')

            subfigHands2woip[-1].plot(b3woip[:-1], a3woip, 'b-', linewidth=1.0)
            # subfigHands2[-1].text(0.6,4,str(np.round(np.mean(corrcoef11_p),2)),fontsize=4,fontweight='bold',color='b')
            subfigHands2woip[-1].plot(b4woip[:-1], a4woip, 'g-', linewidth=1.0)
            # subfigHands2[-1].text(0.6,3.5,str(np.round(np.mean(corrcoef22_p),2)),fontsize=4,fontweight='bold',color='g')
            subfigHands2woip[-1].plot(b5woip[:-1], a5woip, 'r-', linewidth=1.0)

            # subfigHands2[-1].text(0.6,3,str(np.round(np.mean(corrcoef12_p),2)),fontsize=4,fontweight='bold',color='r')
            '''
            if i %6 != 0:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_yticklabels()[1::2]:
                    x.set_visible(False)               
            if i <=29:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_xticklabels()[0::2]:
                    x.set_visible(False)               
                   
            subfigHands2[-1].set_xlim(-1.0,1.)
            subfigHands2[-1].set_ylim(0,8.)
            subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
            subfigHands2[-1].legend(prop={'size':4,"weight":'bold'})
            '''
            subFigsFormat(i, subfigHands2ip[-1])
            # subfigHands2ip[-1].set_yscale('log')
            subFigsFormat(i, subfigHands2woip[-1])
            # subfigHands2woip[-1].set_yscale('log')

        # Making grid plot instead of all rows plot
        subfigHands1ip.append(fig1ip.add_subplot(p.grid_size[0][0] + 1, p.grid_size[0][1], i + 1))
        # Making grid plot instead of all rows plot
        subfigHands1woip.append(fig1woip.add_subplot(p.grid_size[0][0] + 1, p.grid_size[0][1], i + 1))
        subfigHands1ip[-1].plot(b3[:-1], a3, 'b-', linewidth=1.0, label='D1-D1')
        subfigHands1ip[-1].plot(b4[:-1], a4, 'g-', linewidth=1.0, label='D2-D2')
        subfigHands1ip[-1].plot(b5[:-1], a5, 'r-', linewidth=1.0, label='D1-D2')

        subfigHands1woip[-1].plot(b3woip[:-1], a3woip, 'b-', linewidth=1.0)
        # subfigHands1[-1].text(0.6,3,str(np.round(np.mean(corrcoef11_p),2)),fontsize=4,fontweight='bold',color='b')
        subfigHands1woip[-1].plot(b4woip[:-1], a4woip, 'g-', linewidth=1.0)
        # subfigHands1[-1].text(0.6,2.5,str(np.round(np.mean(corrcoef22_p),2)),fontsize=4,fontweight='bold',color='g')
        subfigHands1woip[-1].plot(b5woip[:-1], a5woip, 'r-', linewidth=1.0)

        # subfigHands1[-1].text(0.6,2.0,str(np.round(np.mean(corrcoef12_p),2)),fontsize=4,fontweight='bold',color='r')
        '''
        if i %6 != 0:
            for x in subfigHands1[-1].get_yticklabels():
                x.set_visible(False)
        else:
            for x in subfigHands1[-1].get_yticklabels():
                x.set_fontsize(6)
                x.set_fontweight('bold')
            for x in subfigHands1[-1].get_yticklabels()[1::2]:
                x.set_visible(False)               
        if i <=29:
            for x in subfigHands1[-1].get_xticklabels():
                x.set_visible(False)
        else:
            for x in subfigHands1[-1].get_xticklabels():
                x.set_fontsize(6)
                x.set_fontweight('bold')
            for x in subfigHands1[-1].get_xticklabels()[0::2]:
                x.set_visible(False)               
               
        subfigHands1[-1].set_xlim(-1.0,1.)
        subfigHands1[-1].set_ylim(0,8.)
        subfigHands1[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
        subfigHands1[-1].legend(prop={'size':4,"weight":'bold'})
        '''
        subFigsFormat(i, subfigHands1ip[-1])
        # subfigHands1ip[-1].set_yscale('log')

        subFigsFormat(i, subfigHands1woip[-1])
        # subfigHands1woip[-1].set_yscale('log')

        Correlations[i]['withinCorr_d1d1_ip'] = corrcoef11ip
        Correlations[i]['withinCorr_d1d1_woip'] = corrcoef11woip
        Correlations[i]['withinCorr_d2d2_ip'] = corrcoef22ip
        Correlations[i]['withinCorr_d2d2_woip'] = corrcoef22woip
        Correlations[i]['withinCorr_d1d2_ip'] = corrcoef12ip
        Correlations[i]['withinCorr_d1d2_woip'] = corrcoef12woip

        allBwCorrsd1d1ip = []
        allBwCorrsd1d1woip = []
        allBwCorrsd2d2ip = []
        allBwCorrsd2d2woip = []
        allBwCorrsd1d2ip = []
        allBwCorrsd1d2woip = []

        for j, ch1 in enumerate(hemisphere_neuron_ids['channels']):
            if ch1 == channel:
                continue    # We calculated within correlation above
            ch1_ids = ch1['d1']
            ch1_spike_data = np.array([])
            ch2_ids = ch1['d2']
            ch2_spike_data = np.array([])

            # get spikes for this channel
            mask1 = np.hstack([np.where(n_id == all_senders)[0] for n_id in ch1_ids])
            ch1_spike_senders = all_senders[mask1]
            ch1_spike_times = all_spike_times[mask1]
            mask2 = np.hstack([np.where(n_id == all_senders)[0] for n_id in ch2_ids])
            ch2_spike_senders = all_senders[mask2]
            ch2_spike_times = all_spike_times[mask2]

            # Pick 50 neuron pairs for calculating correlation coefficients
            gids11 = np.array([x for x in itertools.product(list(channel_n_ids1), list(ch1_ids))])
            np.random.shuffle(gids11)
            gids11 = gids11[:numPairs]
            gids22 = np.array([x for x in itertools.product(list(channel_n_ids2), list(ch2_ids))])
            np.random.shuffle(gids22)
            gids22 = gids22[:numPairs]
            gids33 = np.array([x for x in itertools.product(list(channel_n_ids1), list(ch2_ids))])
            np.random.shuffle(gids33)
            gids33 = gids33[:numPairs]
            # For channel 2 , convention, first number 1 is for D1/D2 and second number 2 is for channel
            inds12ip = []
            inds12woip = []
            inds22ip = []
            inds22woip = []
            # for start,stop in zip(start_times,stop_times):
            if len(start_times) == 1 and len(stop_times) == 1:
                start = start_times[0]
                stop = stop_times[0]
                inds12ip.append(np.where(np.logical_and(ch1_spike_times >= start, ch1_spike_times <= stop) == True)[
                                0])  # When input is present for D1
                inds22ip.append(np.where(np.logical_and(ch2_spike_times >= start, ch2_spike_times <= stop) == True)[
                                0])  # When input is present for D2
                inds12woip.append(np.where(np.logical_or(ch1_spike_times < start,
                                                         ch1_spike_times > stop) == True)[0])  # When input is absent
                inds22woip.append(np.where(np.logical_or(ch2_spike_times < start, ch2_spike_times > stop) == True)[0])
                # No need to recalculate mean here, since that remains same
            else:
                start1 = start_times[0]
                start2 = start_times[1]
                stop1 = stop_times[0]
                stop2 = stop_times[1]
                inds12ip.append(np.where(np.logical_or(np.logical_and(ch1_spike_times >= start1, ch1_spike_times <= stop1),
                                                       np.logical_and(ch1_spike_times >= start2, ch1_spike_times <= stop2)) == True)[0])  # When input is present
                inds22ip.append(np.where(np.logical_or(np.logical_and(ch2_spike_times >= start1, ch2_spike_times <= stop1),
                                                       np.logical_and(ch2_spike_times >= start2, ch2_spike_times <= stop2)) == True)[0])  # When input is present
                inds12woip.append(np.where(np.logical_or(np.logical_or(ch1_spike_times < start1, ch1_spike_times > stop2),
                                                         np.logical_and(ch1_spike_times > stop1, ch1_spike_times < start2)) == True)[0])  # When input is absent
                inds22woip.append(np.where(np.logical_or(np.logical_or(ch2_spike_times < start1, ch2_spike_times > stop2),
                                                         np.logical_and(ch2_spike_times > stop1, ch2_spike_times < start2)) == True)[0])  # When input is absent

            # Corrcoef between D1 of "channel"(index i) with D1 of "ch1" (index j) when input is present
            bwcorrcoef11ip = np.array(calcCorrCoef(
                gids11, channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip], ch1_spike_senders[inds12ip], ch1_spike_times[inds12ip], meanD1signalip, meanD1signalip))
            # Corrcoef between D1 of "channel"(index i) with D1 of "ch1" (index j) when input is absent
            bwcorrcoef11woip = np.array(calcCorrCoef(
                gids11, channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip], ch1_spike_senders[inds12woip], ch1_spike_times[inds12woip], meanD1signalwoip, meanD1signalwoip))

            # Corrcoef between D2 of "channel"(index i) with D2 of "ch1"(index j) when input is present
            bwcorrcoef22ip = np.array(calcCorrCoef(
                gids22, channel_spike_senders2[inds2ip], channel_spike_times2[inds2ip], ch2_spike_senders[inds22ip], ch2_spike_times[inds22ip], meanD2signalip, meanD2signalip))
            # Corrcoef between D2 of "channel"(index i) with D2 of "ch1" when input is absent
            bwcorrcoef22woip = np.array(calcCorrCoef(
                gids22, channel_spike_senders2[inds2woip], channel_spike_times2[inds2woip], ch2_spike_senders[inds22woip], ch2_spike_times[inds22woip], meanD2signalwoip, meanD2signalwoip))

            # Corrcoef between D1 of "channel" with D2 of "ch1" when input is present
            bwcorrcoef33ip = np.array(calcCorrCoef(
                gids33, channel_spike_senders1[inds1ip], channel_spike_times1[inds1ip], ch2_spike_senders[inds22ip], ch2_spike_times[inds22ip], meanD1signalip, meanD2signalip))
            bwcorrcoef33woip = np.array(calcCorrCoef(gids33, channel_spike_senders1[inds1woip], channel_spike_times1[inds1woip], ch2_spike_senders[inds22woip],
                                                     ch2_spike_times[inds22woip], meanD1signalwoip, meanD2signalwoip))  # Corrcoef between D1 of "channel" with D2 of "ch1" when input is absent

            if (hemis == 'left_hemisphere' and channel['row'] == 3 and channel['col'] == 4) or (hemis == 'right_hemisphere' and channel['row'] == 3 and channel['col'] == 3):

                subfigHands2ip.append(fig2ip.add_subplot(
                    p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * ch1['row'] + ch1['col'] + 1))
                subfigHands2woip.append(fig2woip.add_subplot(
                    p.grid_size[0][0], p.grid_size[0][1], p.grid_size[0][1] * ch1['row'] + ch1['col'] + 1))
                # Correlation between D1-D1, D1-D2 and D2-D2 between other channels and the one that got input in seqeuences paradigm
                a3, b3 = np.histogram(bwcorrcoef11ip, bins=binCorr, normed=True)
                a4, b4 = np.histogram(bwcorrcoef22ip, bins=binCorr, normed=True)
                a5, b5 = np.histogram(bwcorrcoef33ip, bins=binCorr, normed=True)

                a3woip, b3woip = np.histogram(bwcorrcoef11woip, bins=binCorr, normed=True)
                a4woip, b4woip = np.histogram(bwcorrcoef22woip, bins=binCorr, normed=True)
                a5woip, b5woip = np.histogram(bwcorrcoef33woip, bins=binCorr, normed=True)

                subfigHands2ip[-1].plot(b3[:-1], a3, 'b-', linewidth=1.0, label='D1-D1', alpha=0.5)
                subfigHands2ip[-1].plot(b4[:-1], a4, 'g-', linewidth=1.0, label='D2-D2', alpha=0.5)
                subfigHands2ip[-1].plot(b5[:-1], a5, 'r-', linewidth=1.0, label='D1-D2', alpha=0.5)

                subfigHands2woip[-1].plot(b3woip[:-1], a3woip, 'b-', linewidth=1.0, alpha=0.5)
                # subfigHands2[-1].text(0.6,4,str(np.round(np.mean(bwcorrcoef11_p),2)),fontsize=4,fontweight='bold',color='b')
                subfigHands2woip[-1].plot(b4woip[:-1], a4woip, 'g-', linewidth=1.0, alpha=0.5)
                # subfigHands2[-1].text(0.6,3.5,str(np.round(np.mean(bwcorrcoef22_p),2)),fontsize=4,fontweight='bold',color='g')
                subfigHands2woip[-1].plot(b5woip[:-1], a5woip, 'r-', linewidth=1.0, alpha=0.5)
                # subfigHands2[-1].text(0.6,3,str(np.round(np.mean(bwcorrcoef33_p),2)),fontsize=4,fontweight='bold',color='r')
                '''
                if i %6 != 0:
                    for x in subfigHands2[-1].get_yticklabels():
                        x.set_visible(False)
                else:
                    for x in subfigHands2[-1].get_yticklabels():
                        x.set_fontsize(6)
                        x.set_fontweight('bold')
                    for x in subfigHands2[-1].get_yticklabels()[1::2]:
                        x.set_visible(False)               
                if i <=29:
                    for x in subfigHands2[-1].get_xticklabels():
                        x.set_visible(False)
                else:
                    for x in subfigHands2[-1].get_xticklabels():
                        x.set_fontsize(6)
                        x.set_fontweight('bold')
                    for x in subfigHands2[-1].get_xticklabels()[0::2]:
                        x.set_visible(False)               
                       
                subfigHands2[-1].set_xlim(-1.0,1.)
                subfigHands2[-1].set_ylim(0,8.)
                subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
                subfigHands2[-1].legend(prop={'size':4,"weight":'bold'})
                '''
                subFigsFormat(j, subfigHands2ip[-1])
                # subfigHands2ip[-1].set_yscale('log')
                subFigsFormat(j, subfigHands2woip[-1])
                # subfigHands2woip[-1].set_yscale('log')

                print "j", j

            # 0th element in temp is distance, 1st = d1-d1,2nd = d2-d2, 3rd d1-d2
            # First column is distance, the other 50 columns are corrcoefs
            temp11ip = np.zeros((1, len(bwcorrcoef11ip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp11woip = np.zeros((1, len(bwcorrcoef11woip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp22ip = np.zeros((1, len(bwcorrcoef22ip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp22woip = np.zeros((1, len(bwcorrcoef22woip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp12ip = np.zeros((1, len(bwcorrcoef33ip) + 1))
            # First column is distance, the other 50 columns are corrcoefs
            temp12woip = np.zeros((1, len(bwcorrcoef33woip) + 1))

            rand_row_0 = np.random.randint(6)
            rand_row_1 = np.random.randint(6)
            rand_col_0 = np.random.randint(6)
            rand_col_1 = np.random.randint(6)

            temp11ip[0][0] = temp11woip[0][0] = temp22ip[0][0] = temp22woip[0][0] = temp12ip[0][0] = temp12woip[0][0] = np.sqrt(
                (rand_row_0 - rand_row_1)**2 + (rand_col_0 - rand_col_1)**2)   # Storing the distance of this channel combination

            temp11ip[0][1:] = bwcorrcoef11ip
            temp11woip[0][1:] = bwcorrcoef11woip
            allBwCorrsd1d1ip.append(temp11ip)
            allBwCorrsd1d1woip.append(temp11woip)

            temp22ip[0][1:] = bwcorrcoef22ip
            temp22woip[0][1:] = bwcorrcoef22woip
            allBwCorrsd2d2ip.append(temp22ip)
            allBwCorrsd2d2woip.append(temp22woip)

            temp12ip[0][1:] = bwcorrcoef33ip
            temp12woip[0][1:] = bwcorrcoef33woip
            allBwCorrsd1d2ip.append(temp12ip)
            allBwCorrsd1d2woip.append(temp12woip)

            '''
            a3,b3 = np.histogram(bwcorrcoef11,bins=binCorr,normed=True)    
            a4,b4 = np.histogram(bwcorrcoef22,bins=binCorr,normed=True)    
            a5,b5 = np.histogram(bwcorrcoef33,bins=binCorr,normed=True)    

            subfigHands2.append(fig2.add_subplot(p.grid_size[0][0]+1, p.grid_size[0][1], i + 1)) # Making grid plot instead of all rows plot
            subfigHands2[-1].plot(b3[:-1],a3,'b-',linewidth=1.5,label='D1-D1')
            subfigHands2[-1].plot(b4[:-1],a4,'g-',linewidth=1.5,label='D2-D2')
            subfigHands2[-1].plot(b5[:-1],a5,'r-',linewidth=1.5,label='D1-D2')

            if i %6 != 0:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_yticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_yticklabels()[1::2]:
                    x.set_visible(False)               
            if i <=29:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_visible(False)
            else:
                for x in subfigHands2[-1].get_xticklabels():
                    x.set_fontsize(6)
                    x.set_fontweight('bold')
                for x in subfigHands2[-1].get_xticklabels()[0::2]:
                    x.set_visible(False)               
                   
            subfigHands2[-1].set_xlim(-0.6,1.)
            subfigHands2[-1].set_ylim(0,6.)
            subfigHands2[-1].vlines(x=0,ymin=0,ymax=6,colors='k',linestyles='dashed')
            subfigHands2[-1].legend(prop={'size':4,"weight":'bold'})
            '''

        Correlations[i]['bwCorr_d1d1_ip'] = allBwCorrsd1d1ip
        Correlations[i]['bwCorr_d1d1_woip'] = allBwCorrsd1d1woip
        Correlations[i]['bwCorr_d2d2_ip'] = allBwCorrsd2d2ip
        Correlations[i]['bwCorr_d2d2_woip'] = allBwCorrsd2d2woip
        Correlations[i]['bwCorr_d1d2_ip'] = allBwCorrsd1d2ip
        Correlations[i]['bwCorr_d1d2_woip'] = allBwCorrsd1d2woip

    # Save the data first
    # with open(dirname + "corrcoefs.json", "w+") as f: # json gave an error figure out later , save as pickle for now
    '''
    with open(dirname + "corrcoefs.pickle", "w") as f:
        pickle.dump(Correlations,f) 
    '''
    # Now plot
    fig = pl.figure()
    corrsVsDist = []
    # Numpy.histogram does not work on different size arrays in a list, hence converting them to list
    allWithinCorrsd1d1ip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                            for x in Correlations[i]['withinCorr_d1d1_ip']]
    allWithinCorrsd1d1woip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                              for x in Correlations[i]['withinCorr_d1d1_woip']]
    allWithinCorrsd2d2ip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                            for x in Correlations[i]['withinCorr_d2d2_ip']]
    allWithinCorrsd2d2woip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                              for x in Correlations[i]['withinCorr_d2d2_woip']]
    allWithinCorrsd1d2ip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                            for x in Correlations[i]['withinCorr_d1d2_ip']]
    allWithinCorrsd1d2woip = [x.tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                              for x in Correlations[i]['withinCorr_d1d2_woip']]
    allBetCorrsd1d1ip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                         for x in Correlations[i]['bwCorr_d1d1_ip']]  # 1st column is the distance)
    distd1d1ip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                  for x in Correlations[i]['bwCorr_d1d1_ip']]  # 1st column is the distance)
    # for i in xrange(len(hemisphere_neuron_ids['channels'])):
    #   print "bwCorr_d1d1_woip",Correlations[i]['bwCorr_d1d1_woip']
    #   print "bwCorr_d1d1_ip",Correlations[i]['bwCorr_d1d1_ip']

    allBetCorrsd1d1woip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                           for x in Correlations[i]['bwCorr_d1d1_woip']]  # 1st column is the distance)
    distd1d1woip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                    for x in Correlations[i]['bwCorr_d1d1_woip']]  # 1st column is the distance)

    allBetCorrsd2d2ip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                         for x in Correlations[i]['bwCorr_d2d2_ip']]  # 1st column is the distance)
    distd2d2ip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                  for x in Correlations[i]['bwCorr_d2d2_ip']]  # 1st column is the distance)

    allBetCorrsd2d2woip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                           for x in Correlations[i]['bwCorr_d2d2_woip']]  # 1st column is the distance)
    distd2d2woip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                    for x in Correlations[i]['bwCorr_d2d2_woip']]  # 1st column is the distance)

    allBetCorrsd1d2ip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                         for x in Correlations[i]['bwCorr_d1d2_ip']]  # 1st column is the distance)
    distd1d2ip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                  for x in Correlations[i]['bwCorr_d1d2_ip']]  # 1st column is the distance)

    allBetCorrsd1d2woip = [x[0][1:].tolist() for i in xrange(len(hemisphere_neuron_ids['channels']))
                           for x in Correlations[i]['bwCorr_d1d2_woip']]  # 1st column is the distance)
    distd1d2woip = [x[0][0] for i in xrange(len(hemisphere_neuron_ids['channels']))
                    for x in Correlations[i]['bwCorr_d1d2_woip']]  # 1st column is the distance)

    # corrsVsDist.append(allWithinCorrs) # 0 distance
    a11ip, b11ip = np.histogram(allWithinCorrsd1d1ip, bins=binCorr, normed=True)
    a11woip, b11woip = np.histogram(allWithinCorrsd1d1woip, bins=binCorr, normed=True)
    a22ip, b22ip = np.histogram(allWithinCorrsd2d2ip, bins=binCorr, normed=True)
    a22woip, b22woip = np.histogram(allWithinCorrsd2d2woip, bins=binCorr, normed=True)
    a12ip, b12ip = np.histogram(allWithinCorrsd1d2ip, bins=binCorr, normed=True)
    a12woip, b12woip = np.histogram(allWithinCorrsd1d2woip, bins=binCorr, normed=True)
    ax11 = fig.add_subplot(321)
    ax12 = fig.add_subplot(322)
    ax11.set_title("D1-D1 (Stimulation)", fontsize=10, fontweight='bold')
    ax12.set_title("D1-D1 (Background)", fontsize=10, fontweight='bold')
    ax21 = fig.add_subplot(323)
    ax22 = fig.add_subplot(324)
    ax21.set_title("D2-D2 (Stimulation)", fontsize=10, fontweight='bold')
    ax22.set_title("D2-D2 (Background)", fontsize=10, fontweight='bold')
    ax31 = fig.add_subplot(325)
    ax32 = fig.add_subplot(326)
    ax31.set_title("D1-D2 (Stimulation)", fontsize=10, fontweight='bold')
    ax32.set_title("D1-D2 (Background)", fontsize=10, fontweight='bold')

    ax11.plot(b11ip[:-1], a11ip, 'm-', label='0-40', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax12.plot(b11woip[:-1], a11woip, 'm-', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax21.plot(b22ip[:-1], a22ip, 'm-', label='0-40', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax22.plot(b22woip[:-1], a22woip, 'm-', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax31.plot(b12ip[:-1], a12ip, 'm-', label='0-40', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    ax32.plot(b12woip[:-1], a12woip, 'm-', linewidth=1.5, alpha=0.5)

#    setp(gca(),yscale='log')
    # Find all unique distances
    uniqDist = np.unique(distd1d1ip)  # Any one of the dist arrays will do
    maxDist = 7
    for i in xrange(1, maxDist + 1, 1):       # Too crowded, so sampling every 2 distance

        if i % 2 == 0:
            tempd1d1ip = getCorrAtDist(allBetCorrsd1d1ip, distd1d1ip, i)
            tempd2d2ip = getCorrAtDist(allBetCorrsd2d2ip, distd2d2ip, i)
            tempd1d2ip = getCorrAtDist(allBetCorrsd1d2ip, distd1d2ip, i)
            tempd1d1woip = getCorrAtDist(allBetCorrsd1d1woip, distd1d1woip, i)
            tempd2d2woip = getCorrAtDist(allBetCorrsd2d2woip, distd2d2woip, i)
            tempd1d2woip = getCorrAtDist(allBetCorrsd1d2woip, distd1d2woip, i)
            #temp = np.array(temp)[np.where(np.isnan(np.array(temp))==False)]
            # corrsVsDist.append(temp)    # Next dist
            ad1d1ip, bd1d1ip = np.histogram(tempd1d1ip, bins=binCorr, normed=True)
            ad2d2ip, bd2d2ip = np.histogram(tempd2d2ip, bins=binCorr, normed=True)
            ad1d2ip, bd1d2ip = np.histogram(tempd1d2ip, bins=binCorr, normed=True)
            ad1d1woip, bd1d1woip = np.histogram(tempd1d1woip, bins=binCorr, normed=True)
            ad2d2woip, bd2d2woip = np.histogram(tempd2d2woip, bins=binCorr, normed=True)
            ad1d2woip, bd1d2woip = np.histogram(tempd1d2woip, bins=binCorr, normed=True)

            ax11.plot(bd1d1ip[:-1], ad1d1ip, '-', label=str(i * 40) + "-" + str((i + 1) * 40),
                      linewidth=1.5, color=cm.copper_r((i * 2) / float(maxDist)), alpha=0.5)

            ax12.plot(bd1d1woip[:-1], ad1d1woip, '-', linewidth=1.5, color=cm.copper_r(i / float(maxDist)), alpha=0.5)

            ax21.plot(bd2d2ip[:-1], ad2d2ip, '-', label=str(i * 40) + "-" + str((i + 1) * 40),
                      linewidth=1.5, color=cm.copper_r((i * 2) / float(maxDist)), alpha=0.5)

            ax22.plot(bd2d2woip[:-1], ad2d2woip, '-', linewidth=1.5, color=cm.copper_r(i / float(maxDist)), alpha=0.5)

            ax31.plot(bd1d2ip[:-1], ad1d2ip, '-', label=str(i * 40) + "-" + str((i + 1) * 40),
                      linewidth=1.5, color=cm.copper_r((i * 2) / float(maxDist)), alpha=0.5)

            ax32.plot(bd1d2woip[:-1], ad1d2woip, '-', linewidth=1.5, color=cm.copper_r(i / float(maxDist)), alpha=0.5)

    ax31.set_xlabel("CorrCoef")
    ax31.set_ylabel("PDF")

    ax11.set_xlim(-0.6, 1.0)
    ax12.set_xlim(-0.6, 1.0)
    ax21.set_xlim(-0.6, 1.0)
    ax22.set_xlim(-0.6, 1.0)
    ax31.set_xlim(-0.6, 1.0)
    ax32.set_xlim(-0.6, 1.0)
    '''
    ax11.set_ylim(0,12.0)
    ax12.set_ylim(0,12.0)
    ax21.set_ylim(0,12.0)
    ax22.set_ylim(0,12.0)
    ax31.set_ylim(0,12.0)
    ax32.set_ylim(0,12.0)
    '''
    ax11.legend(prop={'size': 6, 'weight': 'bold'})
    ax21.legend(prop={'size': 6, 'weight': 'bold'})
    ax31.legend(prop={'size': 6, 'weight': 'bold'})
    for x in ax11.get_xticklabels():
        x.set_visible(False)
    stdTicklabelsFormat(ax11.get_yticklabels)

    for x in ax12.get_xticklabels():
        x.set_visible(False)
    stdTicklabelsFormat(ax12.get_yticklabels)
    stdTicklabelsFormat(ax21.get_yticklabels)
    stdTicklabelsFormat(ax22.get_yticklabels)
    stdTicklabelsFormat(ax31.get_yticklabels)
    stdTicklabelsFormat(ax32.get_yticklabels)
    stdTicklabelsFormat(ax31.get_xticklabels)
    stdTicklabelsFormat(ax32.get_xticklabels)

    for x in ax21.get_xticklabels():
        x.set_visible(False)
    for x in ax22.get_xticklabels():
        x.set_visible(False)
    '''
    ax11.set_yscale('log')
    ax12.set_yscale('log')

    ax21.set_yscale('log')
    ax22.set_yscale('log')

    ax31.set_yscale('log')
    ax32.set_yscale('log')
    '''

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle("Pairwise corr coef Vs dist")
    fig.savefig(dirname + "random_D1CorrCoefs_" + hemis + ".pdf")
    fig1ip.savefig(dirname + "random_CorrCoefs_Within_Channels_Stim_" + hemis + ".pdf")
    fig1woip.savefig(dirname + "random_CorrCoefs_Within_Channels_WoStim_" + hemis + ".pdf")
    fig2ip.savefig(dirname + "random_CorrCoefs_Between_Channels_Stim_" + hemis + ".pdf")
    fig2woip.savefig(dirname + "random_CorrCoefs_Between_Channels_WoStim_" + hemis + ".pdf")


def channelHistogram(fn, dirname, hemis):       # Also calculate the within and between channel signal-noise-ratio
    spike_data = np.loadtxt(fn)
    all_senders = spike_data[:, 0]
    all_spike_times = spike_data[:, 1]

    if hemis == 'left_hemisphere':
        with open(dirname + "neuron_ids_left_hemisphere.json", "r+") as f:
            hemisphere_neuron_ids = json.load(f)
    else:
        with open(dirname + "neuron_ids_right_hemisphere.json", "r+") as f:
            hemisphere_neuron_ids = json.load(f)
    expName = fn.split("/")[3]
    filename = "../striatal_model/experiments/" + expName
    # This is to separate the times of activity with and without external input
    if expName == 'sequencesMultTrials.yaml' or expName == 'sequencesMultTrialsd2.yaml':
        with open(filename, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        start_times = cfg['stim-params'][1]['start_times']
        stop_times = cfg['stim-params'][1]['stop_times']
        d1Actip = []
        d1Actwoip = []
        d2Actip = []
        d2Actwoip = []
    hists = []
    binsize = 200.
    binning = np.arange(0, p.runtime, binsize)

    channel_id = 0
    all_d1_spikes = np.array([])
    all_d2_spikes = np.array([])

    fig = pl.figure()

    signalD1 = []  # D1 activity of channel receiving the input
    signalD2 = []  # D2 activity of channel receiving the input
    noiseD1 = []    # D1 activity of all other channels
    noiseD2 = []     # D2 activity of all other channels
    neighborD1 = []  # D1 activity of immediately neighbouring channels
    neighborD2 = []  # D2 activity of immediately neighbouring channels

    for i, channel in enumerate(hemisphere_neuron_ids['channels']):

        #ax = fig.add_subplot(p.num_channels + 1, 1, i + 1)
        ax = fig.add_subplot(p.grid_size[0][0] + 1, p.grid_size[0][1], i +
                             1)  # Making grid plot instead of all rows plot

        binsize = 200.          # Different bin size for individual channel histograms, to see better
        binning = np.arange(0, p.runtime, binsize)

        channel_n_ids = channel['d1']
        channel_spike_data = np.array([])

        # get spikes for this channel
        mask = np.hstack([np.where(n_id == all_senders)[0] for n_id in channel_n_ids])
        channel_spike_senders = all_senders[mask]
        channel_spike_times = all_spike_times[mask]

        all_d1_spikes = np.append(all_d1_spikes, channel_spike_times)

        hist = np.histogram(channel_spike_times, bins=binning)
        binsizeInSecs = binsize / 1000.
        if channel['row'] == 3 and channel['col'] == 3:
            ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs),
                    label="TurnLeft_D1")  # Normalize by channel size to get rate per neuron
            if hemis == 'right_hemisphere':
                signalD1.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
                if expName == 'sequencesMultTrials.yaml' or expName == 'sequencesMultTrialsd2.yaml':
                    currStart = 0
                    for start, stop in zip(start_times, stop_times):
                        currStop = start
                        rated1ip = len(channel_spike_times[np.logical_and(channel_spike_times >= start,
                                                                          channel_spike_times <= stop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                        rated1woip = len(channel_spike_times[np.logical_and(channel_spike_times >= currStart,
                                                                            channel_spike_times <= currStop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                        d1Actip.append(rated1ip)
                        d1Actwoip.append(rated1woip)
                        currStart = stop
        elif channel['row'] == 3 and channel['col'] == 4:
            ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs), label="TurnRight_D1")
            if hemis == 'left_hemisphere':
                signalD1.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
                if expName == 'sequencesMultTrials.yaml' or expName == 'sequencesMultTrialsd2.yaml':
                    currStart = 0
                    for start, stop in zip(start_times, stop_times):
                        currStop = start
                        rated1ip = len(channel_spike_times[np.logical_and(channel_spike_times >= start,
                                                                          channel_spike_times <= stop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                        rated1woip = len(channel_spike_times[np.logical_and(channel_spike_times >= currStart,
                                                                            channel_spike_times <= currStop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                        d1Actip.append(rated1ip)
                        d1Actwoip.append(rated1woip)
                        currStart = stop

        else:
            ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs), label=str(channel_id) + "_D1")
            if hemis == 'right_hemisphere' and (channel['row'] == 2 or channel['col'] == 2):
                neighborD1.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
            elif hemis == 'left_hemisphere' and (channel['row'] == 2 or channel['col'] == 3):
                neighborD1.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
            else:
                noiseD1.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))

        channel_n_ids = channel['d2']
        channel_spike_data = np.array([])

        # get spikes for this channel
        mask = np.hstack([np.where(n_id == all_senders)[0] for n_id in channel_n_ids])
        channel_spike_senders = all_senders[mask]
        channel_spike_times = all_spike_times[mask]

        all_d2_spikes = np.append(all_d2_spikes, channel_spike_times)

        hist = np.histogram(channel_spike_times, bins=binning)
        if channel['row'] == 3 and channel['col'] == 4:
            ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs), label="TurnRight_D2")
            if hemis == 'left_hemisphere':
                signalD2.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
                if expName == 'sequencesMultTrials.yaml' or expName == 'sequencesMultTrialsd2.yaml':
                    currStart = 0
                    for start, stop in zip(start_times, stop_times):
                        currStop = start
                        rated2ip = len(channel_spike_times[np.logical_and(channel_spike_times >= start,
                                                                          channel_spike_times <= stop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                        rated2woip = len(channel_spike_times[np.logical_and(channel_spike_times >= currStart,
                                                                            channel_spike_times <= currStop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                        d2Actip.append(rated2ip)
                        d2Actwoip.append(rated2woip)
                        currStart = stop

        elif channel['row'] == 3 and channel['col'] == 3:
            ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs), label="TurnLeft_D2")
            if hemis == 'right_hemisphere':
                signalD2.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
                if expName == 'sequencesMultTrials.yaml' or expName == 'sequencesMultTrialsd2.yaml':
                    currStart = 0
                    for start, stop in zip(start_times, stop_times):
                        currStop = start
                        rated2ip = len(channel_spike_times[np.logical_and(channel_spike_times >= start,
                                                                          channel_spike_times <= stop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                        rated2woip = len(channel_spike_times[np.logical_and(channel_spike_times >= currStart,
                                                                            channel_spike_times <= currStop)]) / (((stop - start) / 1000.) * len(channel_n_ids))
                        d2Actip.append(rated2ip)
                        d2Actwoip.append(rated2woip)
                        currStart = stop

        else:
            ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * binsizeInSecs), label=str(channel_id) + "_D2")
            if hemis == 'right_hemisphere' and (channel['row'] == 2 or channel['col'] == 2):
                neighborD2.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
            elif hemis == 'left_hemisphere' and (channel['row'] == 2 or channel['col'] == 3):
                neighborD2.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))
            else:
                noiseD2.append(hist[0] / float(len(channel_n_ids) * binsizeInSecs))

        ax.legend(prop={'size': 4, 'weight': 'bold'})
        # Ideally thsi should be for all sequences, since stimulation times is still set to old values, even if simulation times are increased to 200 seconds
        if expName == 'sequencesd1d2.yaml' or expName == 'competingActions.yaml':
            ax.set_xlim(0, 20000.)
        channel_id += 1
        for x in ax.get_xticklabels():
            x.set_visible(False)
        for x in ax.get_yticklabels()[1::2]:
            x.set_visible(False)
        for x in ax.get_yticklabels():
            x.set_fontsize(6)

    binsize = 200.
    binning = np.arange(0, p.runtime, binsize)

    #ax = fig.add_subplot(p.num_channels + 1, 1, p.num_channels + 1)
    ax = plt.subplot2grid((p.grid_size[0][0] + 1, p.grid_size[0][1]), (p.grid_size[0][0], 0), colspan=p.grid_size[0][1])
    hist = np.histogram(all_d1_spikes, bins=binning)
    ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) * p.num_channels * binsizeInSecs),
            label="all_D1", color='b')  # Assuming all channels are equal size
    ax.hlines(y=np.mean(hist[0] / float(len(channel_n_ids) * p.num_channels * binsizeInSecs)),
              xmin=np.min(hist[1][:-1]), xmax=np.max(hist[1][:-1]), linestyles='dashed', colors='b', linewidth=1.5)
    hist = np.histogram(all_d2_spikes, bins=binning)
    ax.plot(hist[1][:-1], hist[0] / float(len(channel_n_ids) *
                                          p.num_channels * binsizeInSecs), label="all_D2", color='g')
    ax.hlines(y=np.mean(hist[0] / float(len(channel_n_ids) * p.num_channels * binsizeInSecs)),
              xmin=np.min(hist[1][:-1]), xmax=np.max(hist[1][:-1]), linestyles='dashed', colors='g', linewidth=1.5)
    ax.legend(prop={'size': 6, 'weight': 'bold'})
    for x in ax.get_yticklabels()[1::2]:
        x.set_visible(False)
    for x in ax.get_yticklabels():
        x.set_fontsize(6)
    # Ideally thsi should be for all sequences, since stimulation times is still set to old values, even if simulation times are increased to 200 seconds
    if expName == 'sequencesd1d2.yaml' or expName == 'competingActions.yaml':
        ax.set_xlim(0, 20000.)

    fhand = file(dirname + "allD1Spikes.gdf", 'w')
    np.savetxt(fhand, all_d1_spikes)
    fhand.close()

    fhand = file(dirname + "allD2Spikes.gdf", 'w')
    np.savetxt(fhand, all_d2_spikes)
    fhand.close()

    fig.savefig(dirname + "D1D2_Hists_" + hemis + ".pdf")

    fig1 = pl.figure()
    ax1 = fig1.add_subplot(111)
    print np.shape(hist[1])
    print np.shape(np.array(signalD1))
    print np.shape(np.array(signalD1) / (np.array(signalD1) + np.array(signalD2)))
    ax1.plot(hist[1][:-1], (np.array(signalD1) / (np.array(signalD1) + np.array(signalD2)))
             [0], 'b-', linewidth=1.5, label='within channel S-N-R D1')
    ax1.plot(hist[1][:-1], (np.array(signalD1) / (np.sum(np.array(noiseD1), axis=0) + np.array(signalD1)))
             [0], 'g-', linewidth=1.5, label='between channel S-N-R D1')
    ax1.plot(hist[1][:-1], (np.array(signalD1) / (np.sum(np.array(neighborD1), axis=0) + np.array(signalD1)))
             [0], 'r-', linewidth=1.5, label='S-N-R w.r.t immediate neighbours -D1')
    ax1.set_ylim(0, 1.5)
    ax1.legend(prop={'size': 10, 'weight': 'bold'})
    fig1.savefig(dirname + "Signal-Noise-Ratios_" + hemis + ".pdf")

    if expName == 'sequencesMultTrials.yaml' or expName == 'sequencesMultTrialsd2.yaml':
        fig2 = pl.figure()
        t1 = fig2.add_subplot(121)
        t1.set_title('D1 Activity')
        t1.plot(np.ones(len(d1Actwoip)) * 2, d1Actwoip, 'o', color='gray')
        t1.plot(np.ones(len(d1Actip)) * 5, d1Actip, 'o', color='blue')
        t1.plot(2, np.mean(d1Actwoip), '*', color='gray', markersize=10)
        t1.plot(5, np.mean(d1Actip), '*', color='blue', markersize=10)
        for woip, ip in zip(d1Actwoip, d1Actip):
            t1.plot([2, 5], [woip, ip], 'b-')
        t1.set_xticks([2, 5])
        t1.set_xticklabels(['Background', 'Stimulation'])
        t1.set_xlim(0, 7)

        t2 = fig2.add_subplot(122)
        t2.set_title('D2 Activity')
        t2.plot(np.ones(len(d2Actwoip)) * 2, d2Actwoip, 'o', color='gray')
        t2.plot(np.ones(len(d2Actip)) * 5, d2Actip, 'o', color='green')
        t2.plot(2, np.mean(d2Actwoip), '*', color='gray', markersize=10)
        t2.plot(5, np.mean(d2Actip), '*', color='green', markersize=10)
        for woip, ip in zip(d2Actwoip, d2Actip):
            t2.plot([2, 5], [woip, ip], 'g-')

        t2.set_xticks([2, 5])
        t2.set_xticklabels(['Background', 'Stimulation'])
        t2.set_xlim(0, 7)
        fig2.savefig(dirname + "MultipleTrialsAct_" + hemis + ".pdf")


def postProcess(ts):

    binning = np.arange(0, runtime, 50.)
    a, b = np.histogram(ts, bins=binning)
    return a / 0.05, b
