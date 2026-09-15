# ========================== IMPORT DEPENDENCIES ==========================
# PYSRIM IMPORT
from srim.output import Collision
from srim.output import Results
from srim import TRIM, Ion, Layer, Target

# OTHER IMPORTS
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
from scipy.integrate import simpson
import mplhep as hep
from cycler import cycler
import statistics
from scipy import integrate
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

# Some constants
m_n = 939.565 # neutron mass in MeV
m_Si = 28.0855 * 931.494 # silicon mass in MeV
SRIM_EXE = 'C:\\Users\\Utente\\Desktop\\SRIM-2013' # path to SRIM executable
SRIM_RANGE_DIR = "C:/Users/Utente/Desktop/SRIM-2013/RANGE.txt" # path to SRIM RANGE.txt file
ION_TO_SIMULATE = 2000 

# Get the neutron spectrum from the txt file
def get_neutron_spectrum(file_path):
    """
    Get neutron spectrum from a text file. The file should have two columns: energy (MeV) and flux (n/cm^2/s).
    \nInput: file_path: str, path to the text file containing the neutron spectrum data.
    \nOutput: energies: numpy array of energies (MeV), fluxes: numpy array of fluxes (n/cm^2/s).
    """

    energies = []
    fluxes = []

    # Read the file and extract energy and flux values
    with open(file_path, 'r') as f:
        for line in f:

            # Skip empty lines and comments
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2: # Check if there are at least two columns
                    try:
                        energy = float(parts[0])
                        flux = float(parts[1])
                        energies.append(energy)
                        fluxes.append(flux)
                    except ValueError:
                        continue

    # Return energies and fluxes as numpy arrays
    return np.array(energies), np.array(fluxes)

# Monte Carlo sampling of neutron energies from the spectrum PDF
def sample_neutron_energies(energies, fluxes, N_samples):
    """
    Sample neutron energies from the given spectrum using Monte Carlo sampling.
    \nInput: energies: base numpy array of energies (MeV), fluxes: base numpy array of fluxes (n/cm^2/s), N_samples: int, number of samples to generate.
    \nOutput: sampled_energies: numpy array of sampled neutron energies (MeV).
    """

    # Integrate the flux over the energy range to get the total flux
    total_flux = integrate.simpson(fluxes, energies)

    # Get the PDF of the neutron spectrum by normalizing the flux with the total flux
    spectrum_PDF = fluxes / total_flux

    # Build the cumulative distribution function (CDF) from the PDF
    cdf = integrate.cumulative_trapezoid(spectrum_PDF, energies, initial=0.0)
    cdf /= cdf[-1]  # Normalize the CDF to ensure it goes from 0 to 1

    # Monte Carlo sampling using inverse-CDF method
    rng = np.random.default_rng()
    random_values = rng.random(N_samples)
    sampled_energies = np.interp(random_values, cdf, energies)

    return sampled_energies

# Function definition for silicon recoil energy after neutron scattering
def silicon_recoil_energy(E_n, theta = 0):
    """Calculate the recoil energy at minimum angle of a silicon nucleus after neutron scattering.
    \nParameters:
    \n - E_n : initial neutron energy in MeV (kinetic + rest mass)
    \n - theta : neutron scattering angle in radians
    \nReturns:
    \n - the recoil energy of the silicon nucleus in MeV
    """
    E_n_total = E_n + m_n  # total energy of the neutron in MeV
    p = np.sqrt(E_n_total**2 - m_n**2)  # neutron momentum in MeV

    E_recoil = m_Si * ((E_n_total + m_Si)**2 + (p * np.cos(theta))**2) / ((E_n_total + m_Si)**2 - (p * np.cos(theta))**2) - m_Si
    return E_recoil # return in MeV

# Function to read range directly from SRIM RANGE.txt file
def read_range_from_file():
    """Read Ion Average Range and Straggling values from RANGE.txt file.
    \n Return a tuple (ion_average_range, straggling) if values are found
    \n None if the file does not exist or values are not found.
    """
    range_file = os.path.join(SRIM_RANGE_DIR)

    if not os.path.exists(range_file):
        return None

    pattern = re.compile(
        r"Ion\s+Average\s+Range\s*=\s*([+-]?\d+(?:\.\d+)?E[+-]?\d+)\s*A\s+Straggling\s*=\s*([+-]?\d+(?:\.\d+)?E[+-]?\d+)\s*A",
        re.IGNORECASE,
    )

    with open(range_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                avg_range = float(match.group(1))
                straggling = float(match.group(2))
                return avg_range, straggling

    print("Amaze, Amaze, Amaze")
    return None

# Getting data from the neutron spectrum file
hv, flux = get_neutron_spectrum('neutron_spectrum.txt')

# Integrate the flux over the energy range to get the total flux
total_flux = integrate.simpson(flux, hv)

# Getting the PDF of the neutron spectrum by normalizing the flux with the total flux
spectrum_PDF = flux / total_flux

# Monte Carlo sampling of neutron energies from the spectrum PDF
N_samples = 10000000
sampled_energies = sample_neutron_energies(hv, flux, N_samples)

# Getting the sampled energies histo
hist_pdf, bin_edges = np.histogram(sampled_energies, bins=100, density=True)

# Getting the sampled flux
hist_flux = hist_pdf * total_flux

# Show histogram of sampled energies
plt.figure(figsize=(8, 5))
plt.hist(sampled_energies, bins=100, density=True, alpha=0.6, label='Sampled Neutron Energies (PDF)', color = 'gold')
plt.plot(hv, spectrum_PDF, label='Neutron Spectrum Flux', lw = 2, color = 'darkred')
plt.xlabel('Energy (MeV)')
plt.ylabel('Counts / Total counts')
plt.title('Monte Carlo Sampling of Neutron Energies (Flux-scaled bins)')
plt.legend()
plt.grid()

# Show sampled flux as a function of energy for sampled energies
plt.figure(figsize=(8, 5))
plt.plot(hv, flux, label='Neutron Flux', lw = 2, color = 'darkblue')
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_widths = np.diff(bin_edges)
plt.bar(bin_centers, hist_flux, width=bin_widths, alpha=0.6,
    align='center', label='Sampled bins x Total Flux', color = 'cyan')
plt.ylabel('Flux (n/cm^2/s)')
plt.xlabel('Energy (MeV)')
plt.title('Neutron Flux vs Energy with Sampled Energies')
plt.legend()
plt.grid()

Si_max_recoil_energy = silicon_recoil_energy(sampled_energies)
TEST = sampled_energies*0.133

# Plot the maximum recoil energy of silicon nuclei as a function of the sampled neutron energies
plt.figure(figsize=(8, 5))
plt.hist(Si_max_recoil_energy, bins=100, density=True, alpha=0.6, label='Max Recoil Energy of Si Nuclei (PDF)', color = 'lightcoral')
plt.xlabel('Recoil Energy (MeV)')
plt.ylabel('Recoil Energy distribution (Counts/TOTAL counts)')
plt.title('Maximum Recoil Energy of Silicon Nuclei from Sampled Neutron Energies')
plt.legend()
plt.grid()

# Getting the max silicon recoil energy
MAX_RECOIL = max(Si_max_recoil_energy)
print("Maximum recoil energy of silicon nuclei: {:.2f} MeV".format(MAX_RECOIL))
print(" ")
plt.show()
# START STRIM SIMULATIONS

# Verify SRIM exists first
if not os.path.exists(SRIM_EXE):
    print(f"ERROR: SRIM directory not found at {SRIM_EXE}")
else:
    print(f"SRIM found at: {SRIM_EXE}\n")
    output_file = "SILICON_RANGE_SIMULATION.txt"
    
    # Defining a silicon ion energy vector
    Si_energies = np.linspace(1.118768192320151, MAX_RECOIL, 7)

    print(Si_energies)
    # Defining a list to store the results
    range_results = []
    std_range_results = []

    # Starting the simulation loop for each silicon ion energy
    for ion_energy in Si_energies:
    
        print("SIMULATING SILICON RECOIL ENERGY: {:.2f} MeV".format(ion_energy))
        
        try:
            # Construct the Silicon ion
            ion = Ion('Si', energy=ion_energy*1000000) # convert MeV to eV for SRIM input
    
            # Construct a layer of silicon 20um thick with a displacement energy of 30 eV
            layer = Layer({
                    'Si': {
                        'stoich': 1.0,
                        'E_d': 15,
                        'lattice': 2.0,
                        'surface': 4.7
                    }}, density=2.3212, width=20000.0)
    
            # Construct a target of a single layer of Silicon
            target = Target([layer])
    
            # Initialize a TRIM calculation with given target and ion for ION_TO_SIMULATE ions (test run)
            trim = TRIM(target, ion, number_ions=ION_TO_SIMULATE, calculation=1)
    
            # Run the simulation (generates RANGE.txt in SRIM folder)
            trim.run(SRIM_EXE)
            
            try: 
                # Read projected range directly from RANGE.txt file
                range_data = read_range_from_file()
                
                # Save data
                range_results.append(range_data[0] if range_data else None)
                std_range_results.append(range_data[1] if range_data else None)

                # Print results
                if range_data:
                    print(f"  Ion Average Range: {range_data[0]:.2f} A")
                    print(f"  Straggling: {range_data[1]:.2f} A\n")
                else:
                    print(f"  Range data not found in RANGE.txt for {ion_energy:.2f} MeV\n")
            
            except Exception as e:
                print(f"    Error reading RANGE.txt: {type(e).__name__}")
                print(f"    Message: {str(e)}\n")
                range_results.append(None)
                std_range_results.append(None)
                continue

        except Exception as e:
            print(f"    Error: {type(e).__name__}")
            print(f"    Message: {str(e)}\n")
            continue
        
        print(ion_energy, " ", range_results[-1], " ", std_range_results[-1])
        print(" ")

    # Save all simulated energies with extracted range values to a text file
    range_results_clean = np.array([np.nan if value is None else value for value in range_results], dtype=float)
    std_range_results_clean = np.array([np.nan if value is None else value for value in std_range_results], dtype=float)
    output_data = np.column_stack((Si_energies, range_results_clean, std_range_results_clean))

    np.savetxt(
        output_file,
        output_data,
        header="Ion_Energy_MeV Ion_Average_Range_A Straggling_A",
        fmt="%.8e",
    )
    print(f"Results saved to {output_file}\n")

    # Plotting the results
    plt.figure(figsize=(8, 5))
    plt.errorbar(Si_energies, range_results_clean, yerr=std_range_results_clean, fmt='o', label='Simulated Range with Straggling', color = 'magenta')
    plt.xlabel('Silicon Ion Energy (MeV)')
    plt.ylabel('Ion Average Range (A)')
    plt.title('Simulated Ion Average Range vs Energy with Straggling')
    plt.legend()
    plt.grid()

plt.show()