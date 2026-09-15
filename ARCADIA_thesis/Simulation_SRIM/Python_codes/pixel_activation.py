# PIXEL ACTIVCATION ANALYSIS 
# We will consider only the motion of the holes, since they are the ones that contribute to the time of the signal
# The result will be a simulation of the effective pixel activation of the sensor
# We will consider both the drift and the diffusion of the holes


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
    
# Import STM class
import STM as stm
    
# Path definition
    
element_names = ["Boron", "Oxygen", "Silicon"]
angles = [0, 30, 45, 60, 70, 80]
density = 2330  # g/cm^3 for silicon and SiO2
image_path = "./SRIM_results/Graphs/"
results_path = "./SRIM_results/"
table_path = os.path.join(results_path, 'RESULTS_TABLE.txt')
output_path = os.path.join(results_path, 'ANALYSIS_RESULT.txt')

# Function to run the analysis for a given element and angle
def run_analysis(element_name, angle):
    
    angle_correction = np.cos(angle * np.pi / 180)
    path = os.path.join("./SRIM_simulations/", f"{element_name}/{angle}/")
    # Getting the data from the RANGE_3D.txt file
    index, range_x, range_y, range_z = stm.get_range_xyz(path)
    
    # Extract data from RESULTS_TABLE.txt
    dati = stm.Textract(table_path, element_name) 
    
    # Defining some constats
    mobility = 480 # (cm^2 /(V * s))    --> TO CHECK
    K = 1.380649e-23 # Boltzmann constant (J/K)
    e = 1.602176634e-19 # Elementary charge (C)
    T = 300 # Temperature (K)   --> TO SET
    
    V = 90 # Bias voltage (V)   --> TO SET
    d = 0.02 # Sensor thickness (cm)   --> TO SET
    E_field = V / d # Electric field (V/cm) supposed to be constat
    
    drift_vel = mobility * E_field # Drift velocity (cm/s)
    D = mobility * K * T / e # Diffusion coefficient (cm^2/s)

    # Finding the holes travel time insiede silicon
    t0 = np.array(dati.rangex) / drift_vel
    
    # Finding the standard deviation of the gaussian distribution at time = t0 after diffusion
    stdev_y = np.sqrt(2 * D * t0 + (np.array(dati.rangey_dev))**2)
    stdev_z = np.sqrt(2 * D * t0 + (np.array(dati.rangez_dev))**2)

    # PLOTTING THE SENSOR ACTIVATION

    # Filter for the specific angle
    index = 0
    for i in range(len(dati.angle)):
        if dati.angle[i] == angle:
            index = i

    N = 3000 # Number of gaussian points
    # Generate random gaussian values for specific mean and standard deviation parameters
    y = np.random.normal(loc = dati.rangey[index], scale = stdev_y[index], size = N)
    z = np.random.normal(loc = dati.rangez[index], scale = stdev_z[index], size = N)


    # --- HEATMAP STYLE: pixel 25x25 μm, 5x5 micropixel, origin ---
    pixel_size = 25  # μm
    micropixel_size = 5  # μm
    n_micropixel_per_pixel = 5
    n_pixel_y = 9  # TO SET 
    n_pixel_z = 5  # TO SET

    # Find the graph's limits
    ymin = -pixel_size/2
    ymax = (n_pixel_y+0.5)*pixel_size
    zmin = -pixel_size/2
    zmax = (n_pixel_z+0.5)*pixel_size

    nbins_y = n_pixel_y * n_micropixel_per_pixel
    nbins_z = n_pixel_z * n_micropixel_per_pixel

    # Find the graph's midpoint
    midpoint_y = pixel_size * n_pixel_y / 2
    midpoint_z = pixel_size * n_pixel_z / 2

    # 2D histo with the percentage of particles in each pixel
    H, yedges, zedges = np.histogram2d(
            y + midpoint_y, z + midpoint_z,
            bins=[nbins_y, nbins_z],
            range=[[ymin, ymax], [zmin, zmax]]
    )
    H_percent = 100 * H / np.sum(H) 

    # Plot
    plt.figure(figsize=(7, 5))
    im = plt.imshow(
            H_percent.T,
            origin='lower',
            aspect='auto',
            extent=[yedges[0], yedges[-1], zedges[0], zedges[-1]],
            cmap='YlOrRd'
    )
    cbar = plt.colorbar(im)
    cbar.set_label('Holes [%]')

    # Grid micropixel
    for i in range(n_pixel_y * n_micropixel_per_pixel + 1):
            plt.axvline(i*micropixel_size - pixel_size/2, color='darkgray', lw=0.3, zorder=1)
    for j in range(n_pixel_z * n_micropixel_per_pixel + 1):
            plt.axhline(j*micropixel_size - pixel_size/2, color='darkgray', lw=0.3, zorder=1)

    # Grid pixel
    for i in range(n_pixel_y+1):
            plt.axvline(i*pixel_size - pixel_size/2, color='gray', lw=0.7, zorder=2)
    for j in range(n_pixel_z+1):
            plt.axhline(j*pixel_size - pixel_size/2, color='gray', lw=0.7, zorder=2)

    plt.xlabel(r'$\Delta y$ [$\mu$m]')
    plt.ylabel(r'$\Delta z$ [$\mu$m]')
    plt.xticks(np.arange(0, n_pixel_y*pixel_size, pixel_size))
    plt.yticks(np.arange(0, n_pixel_z*pixel_size, pixel_size))
    plt.xlim(ymin, ymax)
    plt.ylim(zmin, zmax)
    plt.tight_layout()
    plt.show()

    return stdev_y[index], stdev_z[index], dati.rangey_dev[index], dati.rangez_dev[index]

if __name__ == "__main__":

    for element_name in element_names:
            for angle in angles:
                    try:    
                            stdev_y_after, stdev_z_after, stdev_y_before, stdev_z_before = run_analysis(element_name, angle)
                            print(f"Analysis completed for {element_name} at {angle} degrees.")
                            print(f"Startard dev ($um$) after diffusion (y - z): {stdev_y_after:.2f} - {stdev_z_after:.2f}")
                            print(f"Startard dev ($um$) before diffusion (y - z): {stdev_y_before:.2f} - {stdev_z_before:.2f}\n")
                            print("-----------------------------------------------\n")
                    except Exception as e:
                            print(f"Error for {element_name} at {angle} degrees: {e}")