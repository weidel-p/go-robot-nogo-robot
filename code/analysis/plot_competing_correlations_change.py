import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.stats import pearsonr
import pylab as pl
import sys
import json
import yaml
sys.path.append("code/striatal_model")
import colors
from plot_tools2 import *
import pandas
import seaborn as sbn

corr_in_fn = sys.argv[1]
corr_NoD2_in_fn = sys.argv[2]
out_within_go_left_fn = sys.argv[3]
out_within_go_right_fn = sys.argv[4]
out_between_D1D1_fn = sys.argv[5]
out_between_D2D2_fn = sys.argv[6]


df = pandas.read_json(corr_in_fn)
df_NoD2 = pandas.read_json(corr_NoD2_in_fn)
df_NoD2 = df_NoD2.rename(columns={'CC': 'CC No D2'})
combined = pandas.concat([df, df_NoD2['CC No D2']], axis=1)

go_left_d1d2 = combined.loc[(combined['channel'] == 'go left') & (combined['type'] == 'd1d2')][['CC', 'CC No D2']]
go_right_d1d2 = combined.loc[(combined['channel'] == 'go right') & (combined['type'] == 'd1d2')][['CC', 'CC No D2']]
between_d1d1 = combined.loc[(combined['channel'] == 'between') & (combined['type'] == 'd1d1')][['CC', 'CC No D2']]
between_d2d2 = combined.loc[(combined['channel'] == 'between') & (combined['type'] == 'd2d2')][['CC', 'CC No D2']]


sbn.set_context('paper', font_scale=3., rc={"lines.linewidth": 1.5})
sbn.set_style('whitegrid', {"axes.linewidth": 1.5})


def plot_change(data, label, fn):
    fig = pl.figure(figsize=[4, 8])
    fig.set_tight_layout(True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-1, 1])
    ax.set_ylabel(label, fontweight='bold')
    sbn.violinplot(data, color=colors.colors[1], size=15., scale='width', ax=ax)
    pl.xticks([0, 1], ["with D2-D2", "w/o D2-D2"], rotation=70)
    pl.savefig(fn)


plot_change(go_left_d1d2, "CC (D1-D2) within 'turn left'", out_within_go_left_fn)
plot_change(go_right_d1d2, "CC (D1-D2) within 'turn right'", out_within_go_right_fn)
plot_change(between_d1d1, "CC (D1-D1) between channels", out_between_D1D1_fn)
plot_change(between_d2d2, "CC (D2-D2) between channels", out_between_D2D2_fn)
