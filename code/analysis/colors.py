import seaborn
import pylab as pl


colors = ["#1A2017", "#336492", "#992A44", "#CC6C13", "#8B8A8D",seaborn.color_palette("cubehelix", 8)[2]]
seaborn.set_palette(colors)
seaborn.set_context('paper', font_scale=1.5, rc={"lines.linewidth": 1.5})
seaborn.set_style('whitegrid', {"axes.linewidth": 1.5})
pl.rcParams['font.weight'] = 'bold'
