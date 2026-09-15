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

# Function to read SRIM ionization output files
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

# Function to remap depth for plotting
def remap_depth_for_plot(depth_um):
    return np.where(
        depth_um <= air_max_um,
        depth_um / air_compression,
        (air_max_um / air_compression) + (depth_um - air_max_um) - 0.4
    )

# Alpha energy 
alpha_energy_from_source = 5.486  # MeV

# Path definition
alpha_air_path = "./SRIM_results/Alpha_source_air/IONIZ.txt"
alpha_ARCADIA_path = "./SRIM_results/Alpha_source_ARCADIA/IONIZ.txt"

# Getting the ioniz results using the SRIM output files with the STM class
alpha_air_depth, alpha_air_ion_ions, alpha_air_ion_recoils = read_srim_ionization(alpha_air_path)
alpha_ARCADIA_depth, alpha_ARCADIA_ion_ions, alpha_ARCADIA_ion_recoils = read_srim_ionization(alpha_ARCADIA_path)

# PLOTTING IONIZATION ENERGIES
# Make an integration all over the depth to get the total energy loss
alpha_air_total_ionization = np.trapz(alpha_air_ion_ions, alpha_air_depth)/1000  # Convert to KeV

# Taking data only in the oxide (the first 10 um of the ARCADIA sensor)
oxide_mask = (alpha_ARCADIA_depth >= 0) & (alpha_ARCADIA_depth <= 100000)
alpha_oxide_total_ionization = np.trapz(alpha_ARCADIA_ion_ions[oxide_mask], alpha_ARCADIA_depth[oxide_mask])/1000  # Convert to KeV

# Taking data only in the silicon 
silicon_mask = (alpha_ARCADIA_depth >= 100000) & (alpha_ARCADIA_depth <= 2000000)
alpha_silicon_total_ionization = np.trapz(alpha_ARCADIA_ion_ions[silicon_mask], alpha_ARCADIA_depth[silicon_mask])/1000  # Convert to KeV
print(f"Ionization energy loss in air in KeV: {alpha_air_total_ionization:.1f}")
print(f"Energy loss for ionization in air % of the total energy: {(alpha_air_total_ionization) / (alpha_energy_from_source*1000):.2%}")
print(f"Ionization energy loss in Si02 in KeV: {alpha_oxide_total_ionization:.1f}")
print(f"Energy loss for ionization in Si02 % of the total energy: {(alpha_oxide_total_ionization) / (alpha_energy_from_source*1000):.2%}")
print(f"Ionization energy loss in silicon in KeV: {alpha_silicon_total_ionization:.1f}")
print(f"Energy loss for ionization in silicon % of the total energy: {(alpha_silicon_total_ionization) / (alpha_energy_from_source*1000):.2%}")

# Trasling the depth inside ARCADIA by the distance travelled in air
# The distance travelled in air is the maximum depth of the alpha particle in air
air_distance = np.max(alpha_air_depth)
alpha_ARCADIA_depth = alpha_ARCADIA_depth + air_distance

# Convert depth from Angstrom to micrometer
alpha_air_depth_um = alpha_air_depth / 10000
alpha_ARCADIA_depth_um = alpha_ARCADIA_depth / 10000

fig, ax = plt.subplots(figsize=(10, 6))

# Palette
colors = ['darkred', 'gold', 'darkorange']

# Build a continuous, piecewise x-coordinate:
# air segment is compressed, ARCADIA segment keeps native scale.
air_min_um = np.min(alpha_air_depth_um)
air_max_um = np.max(alpha_air_depth_um)
arc_start_um = np.min(alpha_ARCADIA_depth_um)
arc_end_um = np.max(alpha_ARCADIA_depth_um)
sensor_range_um = arc_end_um - arc_start_um
air_range_um = air_max_um - air_min_um

# Target visual ratio: compressed air span is ~3 times the sensor span.
air_visual_target_um = 2.0 * sensor_range_um
air_compression = air_range_um / air_visual_target_um

x_air_plot = remap_depth_for_plot(alpha_air_depth_um)
x_arcadia_plot = remap_depth_for_plot(alpha_ARCADIA_depth_um)

ax.fill_between(
    x_air_plot,
    alpha_air_ion_ions,
    color=colors[1],
    alpha=0.5,
    zorder=0
)
ax.plot(
    x_air_plot,
    alpha_air_ion_ions,
    color=colors[1],
    lw=2,
    label=r"$\alpha$ in air",
    zorder=1
)

ax.fill_between(
    x_arcadia_plot,
    alpha_ARCADIA_ion_ions,
    color=colors[0],
    alpha=0.5,
    zorder=2
)
ax.plot(
    x_arcadia_plot,
    alpha_ARCADIA_ion_ions,
    color=colors[0],
    lw=2,
    label=r"$\alpha$ in ARCADIA",
    zorder=3
)

# Transition marker between compressed-air and native-sensor scales.
transition_x = air_max_um / air_compression
ax.axvline(transition_x, color='k', ls='--', lw=1.2, alpha=0.7)
ax.text(
    transition_x-0.7,
    0.026 * np.max(alpha_ARCADIA_ion_ions),
    "ARCADIA surface",
    rotation=90,
    va='top',
    ha='right',
    fontsize=12,
    color='k'
)

# Oxide marker line
ax.axvline(transition_x + 10 - 0.1, color='k', ls='--', lw=1.2, alpha=0.7)
ax.text(
    transition_x + 8,
    0.026 * np.max(alpha_ARCADIA_ion_ions),
    "Active volume",
    rotation=90,
    va='top',
    ha='left',
    fontsize=12,
    color='k'
)

# Source marker line
ax.axvline(0, color='k', ls='--', lw=1.2, alpha=0.7)
ax.text(
    2.2,
    0.026 * np.max(alpha_ARCADIA_ion_ions),
    f"Source position - {alpha_energy_from_source:.1f} MeV",
    rotation=90,
    va='top',
    ha='left',
    fontsize=12,
    color='k'
)

# Energy values plotted as text
ax.text(
    22,
    1.3 * np.max(alpha_air_ion_ions),
    f"{alpha_air_total_ionization/1000:.1f} MeV",
    fontsize=12
)

ax.text(
    transition_x + 2,
    1.3 * np.max(alpha_air_ion_ions),
    f"{alpha_oxide_total_ionization/1000:.1f} MeV",
    fontsize=12
)

ax.text(
    transition_x + 12.5,
    1.3 * np.max(alpha_air_ion_ions),
    f"{alpha_silicon_total_ionization/1000:.1f} MeV",
    fontsize=12
)

# Custom ticks shown in true depth (um) for readability.
air_ticks_true = np.array([0, 2500, 5000, 7500, 10000], dtype=float)
air_ticks_true = air_ticks_true[(air_ticks_true >= air_min_um) & (air_ticks_true <= air_max_um)]
arc_ticks_true = np.linspace(arc_start_um, arc_end_um, 5)
arc_ticks_true = np.round(arc_ticks_true, 1)

air_ticks_plot = air_ticks_true / air_compression
arc_ticks_plot = (air_max_um / air_compression) + (arc_ticks_true - air_max_um)

# Bottom axis: show only air-region depth ticks.
ax.set_xticks(air_ticks_plot)
ax.set_xticklabels([f"{t/1000:.0f}" for t in air_ticks_true])

# Optional top axis: depth inside sensor (reset to 0 at ARCADIA entrance).
ax_top = ax.secondary_xaxis('top')
ax_top.set_xticks(arc_ticks_plot)
ax_top.set_xticklabels([f"{np.abs(t - arc_start_um):.0f}" for t in arc_ticks_true])
ax_top.set_xlabel(r"Depth in ARCADIA ($\mu\mathrm{m}$)", fontsize=14, labelpad=10)

ax.set_xlabel(r"Depth in air (mm)", fontsize=14, loc = 'left')
ax.set_ylabel(r"$\mathrm{d}E/\mathrm{d}x$ (eV/$\AA$)", fontsize=14)

# Use a symmetric-log y scale to make both small and large dE/dx regions visible.
all_y = np.concatenate([alpha_air_ion_ions, alpha_ARCADIA_ion_ions])
positive_y = all_y[all_y > 0]
if positive_y.size > 0:
    linthresh = max(np.min(positive_y) * 5, 1e-3)
else:
    linthresh = 1e-3

ax.set_yscale('symlog', linthresh=linthresh, linscale=1.0, base=10)

ax.grid(alpha=0.3)
ax.legend(frameon=True, fontsize=14, loc = 'upper left')

ax.set_ylim(bottom=0, top=np.max(alpha_ARCADIA_ion_ions) * 2)
ax.set_xlim(left=0.001165, right=75.088626)
plt.tight_layout()

plt.show()