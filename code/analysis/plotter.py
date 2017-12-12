import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.cm as cm
import os
import pickle
import json
import plot_tools as ptools
import sys

#layer_D1 = np.loadtxt("../../data/pos_D1.csv")

num_threads = 9

# print "sys.argv", sys.argv
# if os.path.exists('../../data/{}/left_hemisphere.gdf'.format(sys.argv[3])):
#    os.remove('../../data/{}/left_hemisphere.gdf'.format(sys.argv[3]))
#fhand = file('../../data/{}/left_hemisphere.gdf'.format(sys.argv[3]), 'a+')
# for x in xrange(num_threads):
#    filename = "../../data/{}/left_hemisphere-{}.gdf".format(sys.argv[3], x)
#    if os.stat(filename).st_size == 0:
#        continue
##    temp = np.genfromtxt(filename, skip_footer=1)
#    temp = np.loadtxt(filename)
#    np.savetxt(fhand, temp)
# fhand.close()
#
#
# if os.path.exists('../../data/{}/right_hemisphere.gdf'.format(sys.argv[3])):
#    os.remove('../../data/{}/right_hemisphere.gdf'.format(sys.argv[3]))
#fhand = file('../../data/{}/right_hemisphere.gdf'.format(sys.argv[3]), 'a+')
# for x in xrange(num_threads):
#    filename = "../../data/{}/right_hemisphere-{}.gdf".format(sys.argv[3], x)
#    if os.stat(filename).st_size == 0:
#        continue
#    #temp = np.genfromtxt(filename, skip_footer=1)
#    temp = np.loadtxt(filename)
#    np.savetxt(fhand, temp)
# fhand.close()

# if os.stat("../../data/{}/left_hemisphere.gdf".format(sys.argv[3])).st_size > 0:
#    ptools.rasterPlot("../../data/{}/left_hemisphere.gdf".format(sys.argv[3]))
#    pl.savefig(
#        "../../data/{}/raster_plot_left_hemisphere.pdf".format(sys.argv[3]))

ptools.channelHistogram("../../data/{}/left_hemisphere.gdf".format(
    sys.argv[3]), "../../data/{}/".format(sys.argv[3]), "left_hemisphere")

# ptools.channelCorrelations("../../data/{}/left_hemisphere.gdf".format(
#    sys.argv[3]), "../../data/{}/".format(sys.argv[3]), "left_hemisphere", sys.argv[3])

# if os.stat("../../data/{}/right_hemisphere.gdf".format(sys.argv[3])).st_size > 0:
#    ptools.rasterPlot("../../data/{}/right_hemisphere.gdf".format(sys.argv[3]))
#    pl.savefig(
#        "../../data/{}/raster_plot_right_hemisphere.pdf".format(sys.argv[3]))

ptools.channelHistogram("../../data/{}/right_hemisphere.gdf".format(
    sys.argv[3]), "../../data/{}/".format(sys.argv[3]), "right_hemisphere")

# ptools.channelCorrelations("../../data/{}/right_hemisphere.gdf".format(
#    sys.argv[3]), "../../data/{}/".format(sys.argv[3]), "right_hemisphere", sys.argv[3])
