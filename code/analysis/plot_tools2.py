import numpy as np
from scipy.stats import pearsonr
import params
import pylab as pl
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

grid_dimensions = [6, 6]
transient_duration = 500  # (ms)


def get_stim_times(cfg, hemisphere, params, mask=False, scale=10):
    scale = int(scale)

    stim_times_start = np.array([])
    stim_times_stop = np.array([])

    for c in cfg["stim-params"]:
        if c['hemisphere'] == hemisphere:
            # runtimes for correlations is 10 times longer
            stim_times_start = np.array(c['start_times']) * scale
            # runtimes for correlations is 10 times longer
            stim_times_stop = np.array(c['stop_times']) * scale
        else:
            pass

    if len(stim_times_start) == 0:
        # no stimulation in this hemisphere or no stim experiment
        if hemisphere == 'left':
            hemisphere = 'right'
        elif hemisphere == 'right':
            hemisphere = 'left'

        # we get the stimulation times for the other hemisphere, otherwise the plots break.
        # careful, these plots don't make sense and are just for comparison
        for c in cfg["stim-params"]:
            if c['hemisphere'] == hemisphere:
                # runtimes for correlations is 10 times longer
                stim_times_start = np.array(c['start_times']) * scale
                # runtimes for correlations is 10 times longer
                stim_times_stop = np.array(c['stop_times']) * scale
            else:
                pass

        if len(stim_times_start) == 0:
            # no stim experiment. Just return artificial stimulation times
            stim_times_start = [5000. * scale]
            stim_times_stop = [15000. * scale]

    stim_times_start = np.array(stim_times_start)
    stim_times_stop = np.array(stim_times_stop)

    if mask:
        # the times before and after stimulus contain strong transients, this
        # destroys the correlations, here we split stim from background time while removing the transients

        # runtimes for correlations is 10 times longer
        stim_times = np.zeros(int(params.runtime) * scale)
        for t, _ in enumerate(stim_times):
            for boundaries in zip(stim_times_start, stim_times_stop):
                if t >= boundaries[0] + transient_duration and t < boundaries[1]:
                    stim_times[t] = 1

        # runtimes for correlations is 10 times longer
        bckgrnd_times = np.zeros(int(params.runtime) * scale)
        for t, _ in enumerate(bckgrnd_times):
            for boundaries in zip(stim_times_start, stim_times_stop):
                if t < boundaries[0] or t >= boundaries[1] + transient_duration:
                    bckgrnd_times[t] += 1

        # in order to be labeled background, the times must lie outside of ALL stimulations
        bckgrnd_times[np.where(bckgrnd_times < len(stim_times_start))[0]] = 0
        bckgrnd_times[np.where(bckgrnd_times != 0)[0]] = 1

        return [stim_times, bckgrnd_times]

    else:
        return [stim_times_start, stim_times_stop]


def get_grid_pos(nid, channels):
    for cell in channels:
        if nid in cell['d1']:
            return ['d1', [cell['row'], cell['col']]]
        elif nid in cell['d2']:
            return ['d2', [cell['row'], cell['col']]]


def get_grid_pos_fromChanNum(chanNo):
    rows = params.grid_size[0][0]
    cols = params.grid_size[0][1]

    r = chanNo / rows
    c = chanNo % rows

    return (r, c)


def get_ChanNum_gridPos(r, c):
    rows = params.grid_size[0][0]
    cols = params.grid_size[0][1]

    return r * rows + c


def get_NearNeighbors(r, c):
    neighbors = [[r - 1, c], [r, c - 1], [r - 1, c - 1], [r + 1, c + 1],
                 [r + 1, c], [r, c + 1], [r + 1, c - 1], [r - 1, c + 1]]
    # Checking for wrapping up grids
    rows = params.grid_size[0][0]
    cols = params.grid_size[0][1]

    for x in neighbors:
        if x[0] < 0:
            x[0] = rows - 1
        if x[1] < 0:
            x[1] = cols - 1
        if x[0] > rows - 1:
            x[0] = 0
        if x[1] > cols - 1:
            x[1] = 0

    return neighbors


def get_FarNeigbours(r, c):
    rows = params.grid_size[0][0]
    cols = params.grid_size[0][1]
    neighbors = []
    for r1 in np.arange(-2, 3, 1):  # -2 to 2
        if r1 == -2 or r1 == 2:
            cList = np.arange(-2, 3, 1)
        else:
            cList = [-2, 2]
        for c1 in cList:
            nwR = r + r1
            nwC = c + c1
            if nwR < 0:
                nwR = rows - 1
            if nwC < 0:
                nwC = cols - 1
            if nwR > (rows - 1):
                nwR = 0
            if nwC > (cols - 1):
                nwC = 0
            neighbors.append((nwR, nwC))

    return neighbors


def get_dist(x, y):
    # euklidean distance with wrapped edges
    tmp = [min(abs(x[0] - y[0]), abs(x[0] + grid_dimensions[0] - y[0])),
           min(abs(x[1] - y[1]), abs(x[1] + grid_dimensions[1] - y[1]))]
    return np.sqrt(tmp[0]**2 + tmp[1]**2)


def get_spikes_mask(senders, times, nids, scale=10):
    spike_times = [times[np.where(senders == nid)[0]] for nid in nids]

    spike_masks = []
    for st in spike_times:
        tmp = np.zeros(int(params.runtime) * scale)  # simtime
        tmp[st.astype(int)] = 1
        spike_masks.append(tmp)

    return spike_masks


def filter_spikes(spikes, kernel, tau_exp=0.3, scale=10):
    spikes = np.array(spikes) / tau_exp

    fst = [np.convolve(st, kernel, 'full')[:int(
        params.runtime) * scale] for st in spikes]
    if any([any(np.isnan(x)) for x in fst]):
        print "spikes nan", spikes
    return fst


def correlate(filtered_spikes):
    cc = np.eye(len(filtered_spikes))

    for i, st0 in enumerate(filtered_spikes):
        for j, st1 in enumerate(filtered_spikes):
            if i == j:
                continue

            cc_, p_ = pearsonr(st0, st1)
            if p_ > 0.05:
                # setting insignificant correlations to 0 could be misleading
                # but removing them also removes many weakly correlated signals.
                cc_ = 0

            cc[i, j] = cc_
            cc[j, i] = cc_

    return cc


def correlate2(filtered_spikes_A, filtered_spikes_B, deno=1):  # deno - denominator
    cc = np.corrcoef(filtered_spikes_A, filtered_spikes_B)
    cc[np.where(np.isnan(cc))] = 0.0
    return cc


def plot_filt(fst):
    for ft in fst:
        pl.plot(ft)
    pl.show()


def shiftedColorMap(cmap, min_val, max_val, name):
    '''Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero. Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered.
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.'''
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val)  # Edit #2
    midpoint = 1.0 - max_val / (max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False),
                             np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5)  # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap


# calculate all possible distances
all_possible_distances = []
for x in range(grid_dimensions[0]):
    for y in range(grid_dimensions[1]):
        all_possible_distances.append(get_dist([0, 0], [x, y]))

all_possible_distances = np.unique(all_possible_distances)
all_possible_distances = sorted(all_possible_distances)
