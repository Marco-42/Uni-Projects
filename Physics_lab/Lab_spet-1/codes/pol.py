import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
from scipy.constants import constants

# Setting plot style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '10',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (6, 4),
         'axes.labelsize': '10',
         'figure.titlesize' : '14',
         'axes.titlesize':'12',
         'xtick.labelsize':'10',
         'ytick.labelsize':'10',
         'lines.linewidth': '1',
         'text.usetex': False,
#         'axes.formatter.limits': '-5, -3',
         'axes.formatter.min_exponent': '2',
#         'axes.prop_cycle': cycler('color', 'bgrcmyk')
         'figure.subplot.left':'0.125',
         'figure.subplot.bottom':'0.125',
         'figure.subplot.right':'0.925',
         'figure.subplot.top':'0.925',
         'figure.subplot.wspace':'0.1',
         'figure.subplot.hspace':'0.1',
#         'figure.constrained_layout.use' : True
          }
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

distances_tot = []
gray_values_tot = []

with open("./data/tot.txt", "r") as f:
	next(f) 
	for line in f:
		parts = line.strip().split(",")
		if len(parts) == 2:
			try:
				distances_tot.append(float(parts[0])+26.5)
				gray_values_tot.append(float(parts[1]))
			except ValueError:
				pass 

distances_1 = []
gray_values_1 = []

with open("./data/1.txt", "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            try:
                distances_1.append(float(parts[0]) + 11)
                gray_values_1.append(float(parts[1]))
            except ValueError:
                pass
        
distance_2 = []
gray_values_2 = []

with open("./data/2.txt", "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            try:
                distance_2.append(float(parts[0]) + 2)
                gray_values_2.append(float(parts[1]))
            except ValueError:
                pass 

# Plot dei massimi
plt.figure(figsize=(8, 5))
plt.plot(distances_tot, gray_values_tot, label="Data without polarimeter", alpha=0.7, lw = 2)
plt.plot(distances_1, gray_values_1, label=r"Polarimeter 1", alpha=0.7, lw = 2)
plt.plot(distance_2, gray_values_2, label="Polarimeter 2", alpha=0.7, color = "red", lw = 2)
plt.xlabel("Distance (pixels)", fontsize=14)
plt.ylabel("Gray Value", fontsize=14)
plt.title("Gray Value vs Distance con Massimi Locali", fontsize=14)
plt.xlim(900, 3100)
plt.ylim(10, 160)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
