import seaborn
import pylab as pl

colors = ["#1A2017", "#336492", "#992A44", "#CC6C13", "#8B8A8D"]
seaborn.set_palette(colors)
seaborn.set_context('paper', font_scale=3.0, rc={"lines.linewidth": 2.5})
seaborn.set_style('whitegrid', {"axes.linewidth": 2.5})
pl.rcParams['font.weight'] = 'bold'
