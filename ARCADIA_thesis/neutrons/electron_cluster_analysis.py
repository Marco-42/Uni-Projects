import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import mplhep as hep
import numpy as np

# Setting the plot style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '12',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (6, 4),
         'axes.labelsize': '14',
         'figure.titlesize' : '16',
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

# Folder containing the multiplicity files
folder = "./cluster_multiplicity"

# Find all files matching the naming pattern
file_list = sorted(
    glob.glob(os.path.join(folder, "cluster_multiplicity_VCASN_*.txt")),
    key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
)

plt.figure(figsize=(8, 6))

color = ['firebrick', 'peru', 'gold']
# Loop over all files
for file_path, colors in zip(file_list, color):

    # Read the VCASN value from the file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    vcasn = None
    start_row = None

    for i, line in enumerate(lines):
        if line.startswith("VCASN:"):
            vcasn = int(line.split(":")[1].strip())
        if line.startswith("multiplicity"):
            start_row = i + 1
            break

    # Skip invalid files
    if vcasn is None or start_row is None:
        continue

    # Read multiplicity data
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=start_row,
        names=["Multiplicity", "Count"]
    )

    mask = df["Multiplicity"] > 4
    # Plot multiplicity distribution as points with Poisson uncertainties
    plt.errorbar(
        df["Multiplicity"][mask],
        df["Count"][mask],
        yerr=np.sqrt(df["Count"][mask]),
        fmt='o',
        markersize=6,
        elinewidth=1.5,
        linewidth=1,
        color=colors,
        label=f"THR {63 - vcasn}",
        zorder=100-vcasn
    )

# Configure the plot
plt.xlabel("Cluster multiplicity")
plt.ylabel("Counts")
plt.title("Cluster multiplicity distributions", size = 13)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.yscale("log")
plt.tight_layout()
plt.show()
