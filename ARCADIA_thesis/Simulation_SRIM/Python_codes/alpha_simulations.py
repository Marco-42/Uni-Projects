# ========================== IMPORT DEPENDENCIES ==========================
# PYSRIM IMPORT
from srim.output import Collision
from srim.output import Results
    
# OTHER IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
from scipy.integrate import simpson
import mplhep as hep
from cycler import cycler
import statistics
from scipy.odr import *
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
    
# Setting plot style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '10',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (6, 4),
         'axes.labelsize': '14',
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
    
# Import STM class
import STM as stm

# Function to extract data from an isolated IONIZ file
import numpy as np

def read_srim_ionization(filename):
    """
    Read a SRIM ionization output file and return the three data columns:
    depth, ionization by ions, and ionization by recoils.

    Parameters
    ----------
    filename : str
        Path to the SRIM output file.

    Returns
    -------
    depth : numpy.ndarray
        Target depth in Angstrom.
    ion_ions : numpy.ndarray
        Ionization energy loss due to ions.
    ion_recoils : numpy.ndarray
        Ionization energy loss due to recoils.
    """

    depth = []
    ion_ions = []
    ion_recoils = []

    start_reading = False

    with open(filename, 'r') as f:
        for line in f:

            # Detect the beginning of the data table
            if "RECOILS" in line and "IONS" in line:
                start_reading = True
                next(f)  # Skip the dashed separator line
                continue

            if start_reading:
                columns = line.split()

                # Check that the line contains three numerical columns
                if len(columns) == 3:
                    try:
                        depth.append(float(columns[0].replace("E", "e")))
                        ion_ions.append(float(columns[1].replace("E", "e")))
                        ion_recoils.append(float(columns[2].replace("E", "e")))
                    except ValueError:
                        # Ignore lines that are not numerical data
                        pass

    return (
        np.array(depth),
        np.array(ion_ions),
        np.array(ion_recoils)
    )

# Path definition
Si_max_path = "./SRIM_results/Si_MAX/IONIZ.txt"
alpha_6MeV_path = "./SRIM_results/alpha_6MeV/IONIZ.txt"
Mg_6MeV_path = "./SRIM_results/Mg_6MeV/IONIZ.txt"
# Getting the ioniz results using the SRIM output files with the STM class
Si_depth, Si_ion_ions, Si_ion_recoils = read_srim_ionization(Si_max_path)
alpha_6MeV_depth, alpha_6MeV_ion_ions, alpha_6MeV_ion_recoils = read_srim_ionization(alpha_6MeV_path)
Mg_6MeV_depth, Mg_6MeV_ion_ions, Mg_6MeV_ion_recoils = read_srim_ionization(Mg_6MeV_path)

plt.figure(figsize=(6,4))

# Palette
colors = ['darkred', 'gold', 'darkorange']

# Silicon recoils
plt.fill_between(
    Si_depth/10000,
    Si_ion_ions,
    color=colors[0],
    alpha=0.5,
    zorder = 2
)
plt.plot(
    Si_depth/10000,
    Si_ion_ions,
    color=colors[0],
    lw=2,
    label="Si 1.464 MeV",
    zorder = 2
)

# Alpha
plt.fill_between(
    alpha_6MeV_depth/10000,
    alpha_6MeV_ion_ions,
    color=colors[1],
    alpha=0.5,
    zorder = 0
)
plt.plot(
    alpha_6MeV_depth/10000,
    alpha_6MeV_ion_ions,
    color=colors[1],
    lw=2,
    label=r"$\alpha$ 6 MeV",
    zorder = 0
)

# Magnesium
plt.fill_between(
    Mg_6MeV_depth/10000,
    Mg_6MeV_ion_ions,
    color=colors[2],
    alpha=0.5,
    zorder = 1
)

plt.plot(
    Mg_6MeV_depth/10000,
    Mg_6MeV_ion_ions,
    color=colors[2],
    lw=2,
    label="Mg 6 MeV",
    zorder = 1
)

plt.xlabel(r"Depth ($\mu\mathrm{m}$)", fontsize = 14)
plt.ylabel(r"$\mathrm{d}E/\mathrm{d}x$ (eV/$\AA$)", fontsize =14)
plt.grid(alpha=0.3)
plt.legend(frameon=True, fontsize = 12)
plt.tight_layout()
plt.show()
# Make an integration all over the depth to get the total energy loss
Si_total_ionization = np.trapz(Si_ion_ions, Si_depth)/1000  # Convert to KeV
alpha_6MeV_total_ionization = np.trapz(alpha_6MeV_ion_ions, alpha_6MeV_depth)/1000  # Convert to KeV
Mg_6MeV_total_ionization = np.trapz(Mg_6MeV_ion_ions, Mg_6MeV_depth)/1000  # Convert to KeV

print(f"Total ionization energy loss (Si) in KeV: {Si_total_ionization:.1f}")
print(f"Energy loss for ionization % of the total energy (Si): {(Si_total_ionization) / 1464:.2%}")
print(f"Total ionization energy loss (Alpha 6 MeV) in KeV: {alpha_6MeV_total_ionization:.1f}")
print(f"Energy loss for ionization % of the total energy (Alpha 6 MeV): {(alpha_6MeV_total_ionization) / 4000:.2%}")
print(f"Total ionization energy loss (Mg 6 MeV) in KeV: {Mg_6MeV_total_ionization:.1f}")
print(f"Energy loss for ionization % of the total energy (Mg 6 MeV): {(Mg_6MeV_total_ionization) / 5000:.2%}")
plt.show()