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
    
element_names = ["Boron", "Oxygen", "Silicon", "Nickel", "Silver"]
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
    
    # Getting the data from IONIZATION.txt file 
    ionization_depth_x, let_ions_x, let_recoils = stm.get_ionization(path)

    let_ions = let_ions_x * angle_correction  # Adjust LET for the angle of incidence
    ionization_depth = ionization_depth_x / angle_correction  # Adjust depth for the angle of incidence
    
    # The vector will contain the range mean values and his standard deviation for x, y and z
    range_mean = []
    range_std = []
    
    # Extract the first let_ions values for ionization_depth after the 10 micron of SiO2

    # Find the LET minimum value between 6 um and 20 um --> it will be after the Si02
    mask_silicon = (ionization_depth > 60000/angle_correction) & (ionization_depth <= 200000/angle_correction)

    masked_ionization_depth = ionization_depth[mask_silicon]
    masked_let_ions = let_ions[mask_silicon]

    min_LET = np.min(let_ions[mask_silicon])
    min_LET_position = masked_ionization_depth[np.argmin(masked_let_ions)]

    # Take the first 5 values after the min_LET_position to be sure to be in the silicon layer
    idx_start = np.argmax(ionization_depth > min_LET_position)
    window_indices = np.arange(idx_start, idx_start + 5)
    mask_silicon = np.zeros_like(ionization_depth, dtype=bool)
    mask_silicon[window_indices] = True
    ionization_depth_silicon = ionization_depth[mask_silicon]

    # print("Bragg zero depth (um): ", bragg_zero/10000)
    # Extract the corresponding let_ions values for the silicon depth range
    let_ions_silicon = let_ions[mask_silicon]

    # The vector will contain the mean let_ions in the first 20 micron of silicon and his standard deviation
    let_ions_mean = [statistics.mean(let_ions_silicon), statistics.stdev(let_ions_silicon)]

    # COMPUTE THE TOTAL ENERGY DEPOSITED BY THE PARTICLE
    E_Si = simpson(let_ions[ionization_depth >= 100000/angle_correction], ionization_depth[ionization_depth >= 100000/angle_correction])
    E_Si02 = simpson(let_ions[ionization_depth < 100000/angle_correction], ionization_depth[ionization_depth < 100000/angle_correction])
    E_total = E_Si + E_Si02

    #print(f"E_si: {E_Si/1000000:.2f} eV, E_SiO2: {E_Si02/1000000:.2f} eV, E_total: {E_total/1000000:.2f} eV")
    
    # PLOTTING THE IONS X RANGE HISTOGRAM + GAUSS FIT 
    fig, ax = plt.subplots(1, 1, figsize=(6.5,6.5),sharex=True)
    _, bin_edges, _ = ax.hist(range_x/10000, bins=30, color='darkorange', alpha=0.7, edgecolor='black', linewidth = 0.1)
    bin_length = bin_edges[1] - bin_edges[0]
    fit_x, gauss_fit_x, mu, sigma = stm.gauss_simplified_fit(range_x/10000, bin_length)
    range_mean.append(mu)
    range_std.append(sigma)
    ax.plot(fit_x, gauss_fit_x, color='darkcyan', label='Gaussian Fit', linewidth=2.1)
    ax.text(ax.get_xlim()[0] + 0.1*(ax.get_xlim()[1]-ax.get_xlim()[0]), ax.get_ylim()[1]*0.8, 
            f'$R_x$ = {mu:.1f} $\\mu m$ \n$\\sigma_x$ = {sigma:.1f} $\\mu m$', fontsize=14)
    ax.set_xlabel(r'Range(depth) $\mu m$')
    ax.set_ylabel('Counts')
    ax.set_title(f'{element_name} ions({angle} degrees)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, f'{element_name}_{angle}_X.png'), dpi=150)
    plt.close(fig)
    
    # PLOTTING THE IONS Y RANGE HISTOGRAM + GAUSS FIT 
    fig, ay = plt.subplots(1, 1, figsize=(6.5,6.5),sharex=True)
    _, bin_edges_y, _ = ay.hist(range_y/10000, bins=40, color='darkorange', alpha=0.7, edgecolor='black', linewidth = 0.1)
    bin_length_y = bin_edges_y[1] - bin_edges_y[0]
    fit_y, gauss_fit_y, mu_y, sigma_y = stm.gauss_simplified_fit(range_y/10000, bin_length_y)
    range_mean.append(mu_y)
    range_std.append(sigma_y)
    ay.plot(fit_y, gauss_fit_y, color='darkcyan', label='Gaussian Fit', linewidth=2.1)
    ay.text(ay.get_xlim()[0] + 0.1*(ay.get_xlim()[1]-ay.get_xlim()[0]), ay.get_ylim()[1]*0.8, 
            f'$R_y$ = {mu_y:.2f} $\\mu m$ \n$\\sigma_y$ = {sigma_y:.2f} $\\mu m$', fontsize=14)
    ay.set_xlabel(r'Range $Y$ $\mu m$')
    ay.set_ylabel('Counts')
    ay.set_title(f'{element_name} ions({angle} degrees) - Y')
    ay.legend()
    ay.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, f'{element_name}_{angle}_Y.png'), dpi=150)
    plt.close(fig)
    
    # PLOTTING THE IONS Z RANGE HISTOGRAM + GAUSS FIT
    fig, az = plt.subplots(1, 1, figsize=(6.5,6.5),sharex=True)
    _, bin_edges_z, _ = az.hist(range_z/10000, bins=40, color='darkorange', alpha=0.7, edgecolor='black', linewidth = 0.1)
    bin_length_z = bin_edges_z[1] - bin_edges_z[0]
    fit_z, gauss_fit_z, mu_z, sigma_z = stm.gauss_simplified_fit(range_z/10000, bin_length_z)
    range_mean.append(mu_z)
    range_std.append(sigma_z)
    az.plot(fit_z, gauss_fit_z, color='darkcyan', label='Gaussian Fit', linewidth=2.1)
    az.text(az.get_xlim()[0] + 0.1*(az.get_xlim()[1]-az.get_xlim()[0]), az.get_ylim()[1]*0.8, 
            f'$R_z$ = {mu_z:.1f} $\\mu m$ \n$\\sigma_z$ = {sigma_z:.1f} $\\mu m$', fontsize=14)
    az.set_xlabel(r'Range $Z$ $\mu m$')
    az.set_ylabel('Counts')
    az.set_title(f'{element_name} ions({angle} degrees) - Z')
    az.grid(True, linestyle='--', alpha=0.2)
    az.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, f'{element_name}_{angle}_Z.png'), dpi=150)
    plt.close(fig)
    
    # 3D SCATTER PLOT OF IONS RANGE IN X, Y, Z
    X = np.array(range_x) / 10000
    Y = np.array(range_y) / 10000
    Z = np.array(range_z) / 10000
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(X, Y, Z, c=Z, cmap='viridis', alpha=0.5, s=10)
    ax3d.set_xlabel('Range X ($\\mu$m)')
    ax3d.set_ylabel('Range Y ($\\mu$m)')
    ax3d.set_zlabel('Range Z ($\\mu$m)')
    ax3d.set_title(f'3D Range Distribution of {element_name} Ions ({angle} degrees)')
    plt.savefig(os.path.join(image_path, f'{element_name}_{angle}_3D.png'), dpi=150)
    plt.tight_layout()
    plt.close(fig3d)
    
    # PLOTTING LET IONS AND RECOILS AS A FUNCTION OF DEPTH
    fig_let, ax_let = plt.subplots(figsize=(6.5, 6.5))
    ax_let.plot(ionization_depth/10000, let_ions, label='LET Ions', color='darkorange', marker='o', linestyle='-')
    ax_let.plot(ionization_depth/10000, let_recoils, label='LET Recoils', color='darkcyan', marker = 'o', linestyle='-')
    ax_let.axvline(x=10/angle_correction, color='red', linestyle='--', label='SiO2-Si Interface')
    ax_let.set_xlabel('Radial depth ($\\mu$m)')
    ax_let.set_ylabel('LET (eV/Å)')
    ax_let.set_title(f'LET of {element_name} Ions and Recoils as a function of Depth')
    ax_let.legend()
    ax_let.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, f'{element_name}_{angle}_LET.png'), dpi=150)
    
    # plt.show()
    
    plt.close(fig_let)
    
    
    # ========================== APPEND ANALYSIS RESULTS ==========================
    with open(output_path, 'a') as f:
        f.write('----------------------------------------------------------\n\n')
        f.write(f"ION TYPE: {element_name} \t\t ANGLE: {angle} degrees\n\n")
        f.write('RANGE ANALYSIS \n\n')
        f.write(f"Total ions simulated: {len(range_x)}\n\n")
        f.write(f"Mean X (mu): {range_mean[0]:.4f} um\n")
        f.write(f"Standard deviation X (sigma): {range_std[0]:.4f} um\n\n")
        f.write(f"Mean Y (mu): {range_mean[1]:.4f} um\n")
        f.write(f"Standard deviation Y (sigma): {range_std[1]:.4f} um\n\n")
        f.write(f"Mean Z (mu): {range_mean[2]:.4f} um\n")
        f.write(f"Standard deviation Z (sigma): {range_std[2]:.4f} um\n")
        f.write('ION LET ANALYSIS \n\n')
        if angle > 45:
            f.write(f"Mean LET Ions in {element_name} (computed between {ionization_depth_silicon[0]/10000:.1f} - {ionization_depth_silicon[-1]/10000:.1f}): **** eV/Å\n")
            f.write(f"Standard deviation LET Ions in {element_name} (computed between {ionization_depth_silicon[0]/10000:.1f} - {ionization_depth_silicon[-1]/10000:.1f}): **** eV/Å\n")
        else:
            f.write(f"Mean LET Ions in {element_name} (computed between {ionization_depth_silicon[0]/10000:.1f} - {ionization_depth_silicon[-1]/10000:.1f}): {stm.LET_conversion(let_ions_mean[0], density):.4f} MeV*cm^2/mg\n")
            f.write(f"Standard deviation LET Ions in {element_name} (first 20 micron): {stm.LET_conversion(let_ions_mean[1], density):.4f} MeV*cm^2/mg\n")
        f.write(f"Total energy deposited in SiO2: {E_Si02/1000000:.1f} MeV\n")
        f.write(f"Total energy deposited in Si: {E_Si/1000000:.1f} MeV\n")
        f.write(f"Total energy deposited: {E_total/1000000:.1f} MeV\n")

        f.write('\n\n')
    
        # ========================== APPEND RESULTS TABLE ==========================
        with open(table_path, 'a') as f:
                if angle == angles[0]:
                        f.write('----------------------------------------------------------\n\n')
                        f.write(f"ION TYPE: {element_name}\n\n")
                if(angle > 45):
                    f.write(f"{angle} \t {range_mean[0]:.4f} \t {range_mean[1]:.4f} \t {range_mean[2]:.4f} \t {range_std[0]:.4f} \t {range_std[1]:.4f} \t {range_std[2]:.4f} \t *** \t *** \t {E_Si/1000000:.1f} \t {E_Si02/1000000:.1f} \t {E_total/1000000:.1f}\n")
                else:
                    f.write(f"{angle} \t {range_mean[0]:.4f} \t {range_mean[1]:.4f} \t {range_mean[2]:.4f} \t {range_std[0]:.4f} \t {range_std[1]:.4f} \t {range_std[2]:.4f} \t {let_ions_mean[0]:.4f} \t {let_ions_mean[1]:.4f} \t {stm.LET_conversion(let_ions_mean[0], density):.4f} \t {stm.LET_conversion(let_ions_mean[1], density):.4f}\t {E_Si/1000000:.1f} \t {E_Si02/1000000:.1f} \t {E_total/1000000:.1f}\n")
    

# ========================== GETTING DATA ==========================

# ========================== RUN ANALYSIS FOR ALL IONS AND ANGLES ==========================

if __name__ == "__main__":
    # Remove the output file if it exists to start fresh
    if os.path.exists(output_path):
            os.remove(output_path)
            os.remove(table_path)
            with open(output_path, 'a') as f:
                    f.write('ANALYSIS RESULT \n\n')
                    f.write('----------------------------------------------------------\n')
                    f.write(f"TARGET MATERIAL: Si02(10 um) + Silicon\n")
            with open(table_path, 'a') as f:
                    f.write('ANALYSIS RESULT - RANGE RESULT IN MICRON \n\n')
                    f.write('----------------------------------------------------------\n')
                    f.write(f"TARGET MATERIAL: Si02(10 um) + Silicon\n")
                    f.write('----------------------------------------------------------\n\n')
                    f.write('Ion angle(deg) \tMean X \tMean Y \tMean Z \tX std dev \tY std dev \tZ std dev \t Mean LET (eV/A) \t Std Dev LET (eV/A) \tMean LET (MeV*cm^2/mg) \t Std Dev LET (MeV*cm^2/mg) \t Energy transfers Si (MeV) \t SiO2 (MeV) \t Total (MeV) \n')

    for element_name in element_names:
            for angle in angles:
                    try:    
                            run_analysis(element_name, angle)
                            print(f"Analysis completed for {element_name} at {angle} degrees.")
                    except Exception as e:
                            print(f"Error for {element_name} at {angle} degrees: {e}")