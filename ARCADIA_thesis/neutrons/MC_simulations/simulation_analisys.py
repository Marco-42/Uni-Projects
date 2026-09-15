# ========================== IMPORT DEPENDENCIES ==========================

# OTHER IMPORTS
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
from scipy.integrate import simpson
import mplhep as hep
from cycler import cycler
from scipy import integrate
from scipy.odr import *
from scipy.interpolate import interp1d
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from numba import jit
import pandas as pd

# Setting the plot style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '12',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (6, 4),
         'axes.labelsize': '16',
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
n_Si = 5e22 # number of silicon atoms per cm^3
barn = 1e-24 # cm^2
arcadia_thickness = 0.02 # cm, thickness of the active silicon layer of arcadia
Z_SILICON = 14 # atomic number of silicon

FILE_PATH = "simulation_results_silicon_range.txt"
ELASTIC_CROSS_SECTION_PATH = "neutron_elastic_cross_section.txt"
ANELASTIC_CROSS_SECTION_PATH = "neutron_anelastic_cross_section.txt"
ALPHA_CROSS_SECTION_PATH = "neutron_alpha_cross_section.txt"
PROTON_CROSS_SECTION_PATH = "neutron_proton_cross_section.txt"

# Incidation angle of the photon on the ARCADIA sreen
photon_incidation_angle = 0

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

# Get the photon spectrum from the file
def read_spectrum(filename, bin_width=0.048):
    """
    Read a spectrum from a plain text (.txt) file.

    The expected file format is:
        # Comment lines (optional)
        bin,counts

    Example:
        # Bin starting from 0 MeV (bin width = 0.048 MeV), Counts
        1,67
        2,8
        3,20

    Parameters
    ----------
    filename : str
        Path to the input .txt file.
    bin_width : float, optional
        Width of each energy bin in MeV. Default is 0.048 MeV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the columns:
        - bin: bin number
        - energy: energy corresponding to the middle of the bin (MeV)
        - counts: number of entries in the bin
    """
    data = []

    with open(filename, "r") as f:
        for line in f:
            # Remove leading/trailing whitespace
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Read bin number and counts
            bin_number, count = map(int, line.split(","))

            # Compute the energy corresponding to the middle of the bin
            energy = (bin_number - 0.5) * bin_width

            # Store the values
            data.append({
                "bin": bin_number,
                "energy": energy,
                "counts": count
            })

    return pd.DataFrame(data)

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

# Getting range data from txt file
def get_range_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    energies = []
    ranges = []
    std_ranges = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue  # Skip empty/header lines

        parts = re.split(r'\s+', stripped_line)
        if len(parts) >= 2:
            try:
                # Read the first 3 columns and normalize decimal separator.
                energy = float(parts[0].replace(',', '.'))
                range_ = float(parts[1].replace(',', '.'))
                std_ = float(parts[2].replace(',', '.')) if len(parts) >= 3 else 0.0
                energies.append(energy)
                ranges.append(range_)
                std_ranges.append(std_)

            except ValueError:
                continue  # Skip lines that cannot be converted to float

        elif len(parts) == 2: 
                energy = float(parts[0].replace(',', '.'))
                range_ = float(parts[1].replace(',', '.'))
                std_ = 0.0  # Set standard deviation to 0 for lines with only 2 columns
                energies.append(energy)
                ranges.append(range_)
                std_ranges.append(std_)

    return np.array(energies), np.array(ranges), np.array(std_ranges)

# Getting the cross section from txt file
def get_cross_section_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    energies = []
    cross_sections = []

    # Check if the file contains effective data and change , to . for the decimal point
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue  # Skip empty/header lines

        parts = re.split(r'\s+', stripped_line)
        if len(parts) >= 2:
            try:
                # Read only first 2 columns and normalize decimal separator.
                energy = float(parts[0].replace(',', '.'))
                cross_section = float(parts[1].replace(',', '.'))

                if cross_section < 100 and energy < 12000000: # Filter out unrealistic values
                    energies.append(energy)
                    cross_sections.append(cross_section)

            except ValueError:
                continue  # Skip lines that cannot be converted to float
    
    return np.array(energies), np.array(cross_sections)

# Function definition for the Klein-Nishina formula for Compton scattering
def Klein_Nishina(Egamma, x, scatter_angle = False):
    """Calculate the Klein-Nishina formula for Compton scattering.
    \nParameters:
    \n - Egamma : initial photon energy in keV
    \n - x : photon scattering angle in radians
    \n - scatter_angle : if True, calculate the cross section for the scattered theta photon, otherwise for the solid angle
    \nReturns:
    \n - the Klein-Nishina differential cross section in m^2
    """
    # Constants
    r_e = 2.818e-15  # classical electron radius in m
    m_e = 511.0      # electron rest mass energy in keV
    alpha = Egamma / m_e  # dimensionless photon energy
    coseno = np.cos(x)
    denominator = (1 + alpha * (1 - coseno))

    if scatter_angle:
        return Z_SILICON * (r_e**2 * np.pi * np.sin(x)) * ((1 + coseno**2) / denominator**2) * (1 + (alpha**2 * (1 - coseno)**2) / ((1 + coseno**2) * denominator))
    else:
        return Z_SILICON * (r_e**2 * 0.5) * ((1 + coseno**2) / denominator**2) * (1 + (alpha**2 * (1 - coseno)**2) / ((1 + coseno**2) * denominator))

# Function to sample scattering angles from the Klein-Nishina distribution using inverse transform sampling
def sample_theta_kn(Egamma, n_samples, n_grid=20000, rng=None):
    """
    Sample scattering angles from the Klein-Nishina distribution using inverse transform sampling.
    Parameters: 
    \n - Egamma : initial photon energy in keV
    \n - n_samples : number of scattering angles to sample
    \n - n_grid : number of points to use for the numerical integration and CDF construction
    \n - rng : random number generator (optional)
    \nReturns:
    \n - theta_samples : numpy array of sampled scattering angles in radians
    \n - theta : numpy array of scattering angles used for the CDF construction
    \n - pdf : numpy array of the Klein-Nishina PDF values corresponding to the theta array
    """
    
    if rng is None:
        rng = np.random.default_rng()

    theta = np.linspace(0.0, np.pi, n_grid)

    # Unnormalized density in theta
    pdf = Klein_Nishina(Egamma, theta, scatter_angle=True)
    pdf = np.clip(pdf, 0.0, None)

    # Normalize
    norm = integrate.simpson(Klein_Nishina(Egamma, theta, scatter_angle=True), theta)
    pdf /= norm

    # Build CDF numerically
    cdf = integrate.cumulative_trapezoid(pdf, theta, initial=0.0)
    cdf /= cdf[-1]

    # Inverse-CDF sampling
    u = rng.random(n_samples)
    theta_samples = np.interp(u, cdf, theta)
    return theta_samples, theta, pdf

# Monte Carlo sampling of photon energies from a Gaussian distribution
def sample_photon_energies(mean, sigma, N_samples):
    """Sample photon energies from a Gaussian distribution using Monte Carlo sampling.
    \nParameters:
    \n - mean : mean energy of the Gaussian distribution in keV
    \n - sigma : standard deviation of the Gaussian distribution in keV
    \n - N_samples : number of samples to generate
    \nReturns:
    \n - sampled_energies : numpy array of sampled photon energies in keV
    """
    rng = np.random.default_rng()
    sampled_energies = rng.normal(loc=mean, scale=sigma, size=N_samples)
    return sampled_energies

# Monte Carlo sampling of photon energies from a double Gaussian distribution
def sample_photon_energies_double_gaussian(mean1, sigma1, mean2, sigma2, N_samples):
    """Sample photon energies from a double Gaussian distribution using Monte Carlo sampling.
    \nParameters:
    \n - mean1 : mean of the first Gaussian distribution in keV
    \n - sigma1 : standard deviation of the first Gaussian distribution in keV
    \n - mean2 : mean of the second Gaussian distribution in keV
    \n - sigma2 : standard deviation of the second Gaussian distribution in keV
    \n - N_samples : number of samples to generate
    \nReturns:
    \n - sampled_energies : numpy array of sampled photon energies in keV
    """
    rng = np.random.default_rng()
    target_peak_ratio = 0.5
    weight_ratio = target_peak_ratio * (sigma1 / sigma2)
    p1 = weight_ratio / (1 + weight_ratio)
    n1 = int(round(N_samples * p1))
    n2 = N_samples - n1
    sampled_energies1 = rng.normal(loc=mean1, scale=sigma1, size=n1)
    sampled_energies2 = rng.normal(loc=mean2, scale=sigma2, size=n2)
    sampled_energies = np.concatenate([sampled_energies1, sampled_energies2])
    return sampled_energies

# Photon energy after compton scattering
def scattered_photon_energy(Egamma, x):
    """Calculate the energy of the scattered photon after Compton scattering.
    \nParameters:
    \n - Egamma : initial photon energy in keV
    \n - x : photon scattering angle in radians
    \nReturns:
    \n - the energy of the scattered photon in keV
    """
    m_e = 511.0  # electron rest mass energy in keV
    return Egamma / (1 + (Egamma / m_e) * (1 - np.cos(x)))

# Electron energy after compton scattering
def scattered_electron_energy(Egamma, x):
    """Calculate the energy of the scattered electron after Compton scattering.
    \nParameters:
    \n - Egamma : initial photon energy in keV
    \n - x : photon scattering angle in radians
    \nReturns:
    \n - the energy of the scattered electron in keV
    """
    return Egamma - scattered_photon_energy(Egamma, x)

# Electron angle after compton scattering
def scattered_electron_angle(Egamma, gamma_angle):
    """Calculate the angle of the scattered electron after Compton scattering.
    \nParameters:
    \n - Egamma : initial photon energy in keV
    \n - gamma_angle : photon scattering angle in radians
    \nReturns:
    \n - the angle of the scattered electron in radians
    """

    m_e = 511.0  # electron rest mass energy in keV
    return np.pi/2 - np.arctan((1 + Egamma/m_e) * np.tan(gamma_angle/2))

# Electron angle from electron energy after compton scattering
def scattered_electron_angle_from_electron_energy(Egamma, electron_energy):
    """Calculate the angle of the scattered electron after Compton scattering from the energy of the scattered electron.
    \nParameters:
    \n - Egamma : initial photon energy in keV
    \n - electron_energy : energy of the scattered electron in keV
    \nReturns:
    \n - the angle of the scattered electron in radians
    """

    m_e = 511.0  # electron rest mass energy in keV
    return np.arctan((np.sqrt(Egamma**2 - m_e**2)) / (Egamma + m_e - electron_energy))

# Function to get the x, y projection of the electron scattering solid angle
def simulate_solid_angle(scattering_angle, e_range, degree_offset = 30):
    """
    Calculate the x, y projection of the electron scattering solid angle.
    \nConsidering the phi distribution as uniform
    \nParameters:
    \n - scattering_angle : the scattering angle of the electron in radians
    \n - e_range : the range of the electron in microns
    \n - degree_offset : the angle of the tilt in degrees(pre-set to 30°)
    \nReturns:
    \n - x_rotated : coordinates of the solid angle projection(scattering plane x - z)
    \n y_rotated - coordinates of the solid angle projection(screeen plane perpendicolare to the scattering one)
    """

    # Define theta and phi ranges
    theta = scattering_angle * np.ones(1)  # Convert degrees to radians

    # Extract casually a number between 0 and 360 for the phi angle, considering it as uniformly distributed
    rng = np.random.default_rng()
    phi = rng.uniform(0, 2 * np.pi, 1) * np.ones(1)  # Single random value for phi

    degree_offset = np.radians(degree_offset)  # Convert degree_offset to radians

    # Simulate the interacting point of the photon considering the interaction
    # probability as uniform
    interaction_point_z = rng.uniform(0, arcadia_thickness*10000) # in microns
    r = np.linspace(0, e_range, 1000)  # Range of the electron in microns, with 1000 points for smoothness

    # Create a meshgrid for theta and phi
    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) + interaction_point_z # Shift the z coordinate by the interaction point to simulate the interaction happening at different depths in the sensor

    # Define rotation matrix for 30-degree tilt (only y-axis is rotated, z-axis remains unchanged)
    rotation_matrix = np.array([
        [np.cos(degree_offset), 0, np.sin(degree_offset)],
        [0, 1, 0],
        [-np.sin(degree_offset), 0, np.cos(degree_offset)]
    ])

    # Apply rotation to the coordinates (exclude z-axis rotation)
    coords = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotated_coords = rotation_matrix @ coords

    # Reshape rotated coordinates back to original shape
    x_rotated = rotated_coords[0].reshape(x.shape)
    y_rotated = rotated_coords[1].reshape(y.shape)
    z_rotated = rotated_coords[2].reshape(z.shape)  # Update z_rotated to the rotated z coordinates
    
    to_return_x = 0.0
    to_return_y = 0.0

    # Rotate the interaction point
    interaction_point_z = interaction_point_z * np.cos(degree_offset)

    # Remove axis swapping
    # x_rotated, y_rotated = y_rotated, x_rotated
    for i in range(z_rotated.shape[0]):
        for j in range(z_rotated.shape[1]):

            massimo = np.max(z_rotated[0, :])
            minimo = np.min(z_rotated[0, :])
            
            # If the electron goes outside the active silicon layer of arcadia, set the x and y coordinates to the last point before going outside
            if z_rotated[i, j] > arcadia_thickness*10000 and massimo > (arcadia_thickness*10000): # Set z values greater than 200 to 200 (arcadia_thickness in microns)
                to_return_x = x_rotated[i, j-1]
                to_return_y = y_rotated[i, j-1]
                #print(theta[0] * 180 / np.pi + degree_offset, phi, e_range, to_return_x, to_return_y)
                break
                
            # If the electron doesn't go outside the active silicon layer of arcadia, set the x and y coordinates to the last point of the range
            if interaction_point_z <massimo < (arcadia_thickness*10000): # Set z values greater than 200 to 200 (arcadia_thickness in microns)
                to_return_x = x_rotated[-1, -1]
                to_return_y = y_rotated[-1, -1]
                # print(np.max(z_rotated[0, :]), " ", interaction_point_z)
                break

            # If the electron is backscattered, and it goes outside set the x and y coordinates to the last point before going outside
            if minimo < 0 and z_rotated[i, j] < 0: # Set z values greater than 200 to 200 (arcadia_thickness in microns)
                to_return_x = x_rotated[i, j-1]
                to_return_y = y_rotated[i, j-1]
                break
            
            # If the electron is backscattered, and it doesn't go outside set the x and y coordinates to the last point of the range
            if 0 < minimo < interaction_point_z: # Set z values greater than 200 to 200 (arcadia_thickness in microns)
                to_return_x = x_rotated[0, -1]
                to_return_y = y_rotated[0, -1]
                break

    return to_return_x, to_return_y

# Function to get the number of pixel activated for one compton scattering
def pixel_activation(x, y):
    """
    Calculate the number of pixels activated by a particle moving along a trajectory.
    The particle activates all pixels it passes through along its path.

    Parameters:
    - x: x-coordinate of the particle's final position (in microns).
    - y: y-coordinate of the particle's final position (in microns).

    Returns:
    - num_pixels: The total number of activated pixels.
    """
    pixel_size = 25  # microns

    # Convert the trajectory to pixel coordinates
    x_pixel_start, y_pixel_start = 0, 0  # Assume the particle starts at the origin (0, 0)
    
    # Ensure x and y are scalars before converting to integers
    x_pixel_end = int((x // pixel_size).item() if np.ndim(x) > 0 else x // pixel_size)
    y_pixel_end = int((y // pixel_size).item() if np.ndim(y) > 0 else y // pixel_size)

    # Use Bresenham's line algorithm to find all pixels along the trajectory
    activated_pixels = set()
    dx = abs(x_pixel_end - x_pixel_start)
    dy = abs(y_pixel_end - y_pixel_start)
    sx = 1 if x_pixel_start < x_pixel_end else -1
    sy = 1 if y_pixel_start < y_pixel_end else -1
    err = dx - dy

    x_pixel, y_pixel = x_pixel_start, y_pixel_start
    while True:
        activated_pixels.add((x_pixel, y_pixel))
        if x_pixel == x_pixel_end and y_pixel == y_pixel_end:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x_pixel += sx
        if e2 < dx:
            err += dx
            y_pixel += sy
    # print("Activated Pixels:")
    # print(len(activated_pixels))
    # print(np.sqrt(x**2 + y**2)/25)
    # return len(activated_pixels)
    return int(np.sqrt(x**2 + y**2)/25)

# =========================================================
#                     NEUTRON ANALYSIS
# =========================================================

ENERGIES, RANGES, STD_RANGES = get_range_data(FILE_PATH)
ELASTIC_STANDARD_ENERGIES, ELASTIC_CROSS_SECTION = get_cross_section_data(ELASTIC_CROSS_SECTION_PATH)
ANELASTIC_STANDARD_ENERGIES, ANELASTIC_CROSS_SECTION = get_cross_section_data(ANELASTIC_CROSS_SECTION_PATH)
NEUTRON_ALPHA_ENERGIES, NEUTRON_ALPHA_CROSS_SECTION = get_cross_section_data(ALPHA_CROSS_SECTION_PATH)
PROTON_ENERGIES, PROTON_CROSS_SECTION = get_cross_section_data(PROTON_CROSS_SECTION_PATH)

plt.figure(figsize=(6, 4))
plt.errorbar(ENERGIES, RANGES/10000, yerr=STD_RANGES/10000, fmt='o', label='Simulated Data', color = 'firebrick')
plt.xlabel('Energy (MeV)')
plt.ylabel(r'Range ($\mu$m)')
plt.title('Range of Silicon nucleus in Silicon')
plt.legend(loc = 'lower right', fontsize = 12)
plt.grid()

print("Max range: ", np.max(RANGES/10000), " +- ", np.max(STD_RANGES/10000), " um")
plt.figure(figsize=(6, 4))
plt.plot(ELASTIC_STANDARD_ENERGIES/1000000, ELASTIC_CROSS_SECTION, label='Elastic scattering', color = 'dodgerblue')
plt.plot(ANELASTIC_STANDARD_ENERGIES/1000000, ANELASTIC_CROSS_SECTION, label='Inelastic scattering', color = 'firebrick')
plt.plot(NEUTRON_ALPHA_ENERGIES/1000000, NEUTRON_ALPHA_CROSS_SECTION, label='Alpha emission', color = 'darkgreen')
plt.plot(PROTON_ENERGIES/1000000, PROTON_CROSS_SECTION, label='Proton emission', color = 'darkorange')
plt.xlabel('Energy (MeV)')
plt.ylabel('Cross Section (b)')
plt.title('Neutron Cross Sections in Silicon')
plt.xlim(0.142, 13)
plt.legend(ncol = 2)
plt.xscale('log')
plt.yscale('log')
plt.grid()

# Converting barn -> cm^2 
elastic_cross_section_cm2 = ELASTIC_CROSS_SECTION * barn
anelastic_cross_section_cm2 = ANELASTIC_CROSS_SECTION * barn
alpha_cross_section_cm2 = NEUTRON_ALPHA_CROSS_SECTION * barn
proton_cross_section_cm2 = PROTON_CROSS_SECTION * barn

# Getting data from the neutron spectrum file
hv, flux = get_neutron_spectrum('neutron_spectrum_V0.txt')

# Integrate the flux over the energy range to get the total flux
total_flux = integrate.simpson(flux, hv)

# Getting the PDF of the neutron spectrum by normalizing the flux with the total flux
spectrum_PDF = flux / total_flux

# Monte Carlo sampling of neutron energies from the spectrum PDF
N_samples = 10000000
#N_samples = 100
N_bin = 100
sampled_energies = sample_neutron_energies(hv, flux, N_samples)

# Getting the sampled energies histo
hist_pdf, bin_edges = np.histogram(sampled_energies, bins=N_bin, density=False)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_widths = np.diff(bin_edges)

# hist_pdf contain the number of neutron with a specific energy, saved in bin_centers
# Founding the nearest cross section values con bin_centers
elastic_cross_section_interp = interp1d(ELASTIC_STANDARD_ENERGIES/1000000, elastic_cross_section_cm2, bounds_error=False, fill_value=0)
anelastic_cross_section_interp = interp1d(ANELASTIC_STANDARD_ENERGIES/1000000, anelastic_cross_section_cm2, bounds_error=False, fill_value=0)
alpha_cross_section_interp = interp1d(NEUTRON_ALPHA_ENERGIES/1000000, alpha_cross_section_cm2, bounds_error=False, fill_value=0)
proton_cross_section_interp = interp1d(PROTON_ENERGIES/1000000, proton_cross_section_cm2, bounds_error=False, fill_value=0)

# Getting the cross section values for the sampled energies using the interpolating functions
elastic_cross_section_values = elastic_cross_section_interp(bin_centers)
anelastic_cross_section_values = anelastic_cross_section_interp(bin_centers)
alpha_cross_section_values = alpha_cross_section_interp(bin_centers)
proton_cross_section_values = proton_cross_section_interp(bin_centers)

elastic_mu = n_Si * elastic_cross_section_values # macroscopic cross section in cm^-1 for the sampled energies
anelastic_mu = n_Si * anelastic_cross_section_values # macroscopic cross section in cm^-1 for the sampled energies
alpha_mu = n_Si * alpha_cross_section_values # macroscopic cross section in cm^-1 for the sampled energies
proton_mu = n_Si * proton_cross_section_values # macroscopic cross section in cm^-1 for the sampled energies

elastic_hist_pdf_scaled = hist_pdf * (1 - np.exp(-elastic_mu * arcadia_thickness)) # Number of neutrons that interact in the active silicon layer of arcadia for each energy bin
anelastic_hist_pdf_scaled = hist_pdf * (1 - np.exp(-anelastic_mu * arcadia_thickness)) # Number of neutrons that interact in the active silicon layer of arcadia for each energy bin
alpha_hist_pdf_scaled = hist_pdf * (1 - np.exp(-alpha_mu * arcadia_thickness)) # Number of neutrons that interact in the active silicon layer of arcadia for each energy bin
proton_hist_pdf_scaled = hist_pdf * (1 - np.exp(-proton_mu * arcadia_thickness)) # Number of neutrons that interact in the active silicon layer of arcadia for each energy bin

# Show histogram of sampled energies
plt.figure(figsize=(8, 5))
plt.hist(sampled_energies, bins=N_bin, density=True, alpha=0.6, label='Sampled Neutron Energies (PDF)', color = 'gold', weights=np.ones_like(sampled_energies))
plt.plot(hv, spectrum_PDF, label='Neutron Spectrum Flux', lw = 2, color = 'darkred', ls = '--')
plt.xlabel('Energy (MeV)')
plt.ylabel('Counts / Total counts')
plt.title('Monte Carlo Sampling of Neutron Energies [AmBe] - {:.0f} samples'.format(N_samples))
plt.legend()
plt.grid()

# Show histogram of sampled energies scaled by the interaction probability in the active silicon layer of arcadia
plt.figure(figsize=(8, 5))
plt.bar(bin_centers, elastic_hist_pdf_scaled, width=bin_widths, alpha=0.7, label='Elastic scattering', color = 'dodgerblue')
plt.bar(bin_centers, anelastic_hist_pdf_scaled, width=bin_widths, alpha=0.7, label='Inelastic scattering', color = 'firebrick')
plt.bar(bin_centers, proton_hist_pdf_scaled, width=bin_widths, alpha=0.7, label='Proton scattering', color = 'darkorange')
plt.bar(bin_centers, alpha_hist_pdf_scaled, width=bin_widths, alpha=0.7, label='Alpha scattering', color = 'darkgreen')
plt.xlabel('Energy (MeV)')
plt.ylabel('Counts')
plt.title('Monte Carlo Sampling of Neutron interaction probability in Silicon')
plt.legend()
plt.grid()

# Show histogram of sampled energies scaled by the interaction probability in the active silicon layer of arcadia(normalized by the total neutron simulated and shown as a percentage)
plt.figure(figsize=(8, 5))
plt.bar(bin_centers, elastic_hist_pdf_scaled *100 / N_samples, width=bin_widths, alpha=0.7, label='Elastic scattering', color = 'dodgerblue')
plt.bar(bin_centers, anelastic_hist_pdf_scaled *100 / N_samples, width=bin_widths, alpha=0.7, label='Inelastic scattering', color = 'firebrick')
plt.bar(bin_centers, proton_hist_pdf_scaled *100 / N_samples, width=bin_widths, alpha=0.7, label='Proton scattering', color = 'darkorange')
plt.bar(bin_centers, alpha_hist_pdf_scaled *100 / N_samples, width=bin_widths, alpha=0.7, label='Alpha scattering', color = 'darkgreen')
plt.xlabel('Energy (MeV)')
plt.ylabel('Normalized Counts (%)')
plt.legend()
plt.grid()

# Percentual of interacting neutrons in the active silicon layer of arcadia
elastic_interacting_neutrons = elastic_hist_pdf_scaled / hist_pdf * 100
anelastic_interacting_neutrons = anelastic_hist_pdf_scaled / hist_pdf * 100
alpha_interacting_neutrons = alpha_hist_pdf_scaled / hist_pdf * 100
proton_interacting_neutrons = proton_hist_pdf_scaled / hist_pdf * 100

plt.figure(figsize=(8, 5))
plt.bar(bin_centers, elastic_interacting_neutrons, width=bin_widths, alpha=0.6, label='Percentage of Elastic Interacting', color = 'dodgerblue')
plt.bar(bin_centers, anelastic_interacting_neutrons, width=bin_widths, alpha=0.6, label='Percentage of Anelastic Interacting', color = 'firebrick')
plt.bar(bin_centers, alpha_interacting_neutrons, width=bin_widths, alpha=0.6, label='Percentage of Alpha Interacting', color = 'darkgreen')
plt.bar(bin_centers, proton_interacting_neutrons, width=bin_widths, alpha=0.6, label='Percentage of Proton Interacting', color = 'darkorange')
plt.xlabel('Energy (MeV)')
plt.ylabel('Percentage of Interacting Neutrons (%)')
plt.title('Percentage of Interacting Neutrons in Silicon as a Function of Neutron Energy')
plt.legend()
plt.grid()

# Total interacting neutrons in the active silicon layer of arcadia
elastic_total_interacting_neutrons = np.sum(elastic_hist_pdf_scaled)
anelastic_total_interacting_neutrons = np.sum(anelastic_hist_pdf_scaled)
alpha_total_interacting_neutrons = np.sum(alpha_hist_pdf_scaled)
proton_total_interacting_neutrons = np.sum(proton_hist_pdf_scaled)

print(" ")
print("Total number of sampled neutrons: {:.0f}".format(N_samples))
print("\nTotal number of interacting elastic neutrons: {:.2f}".format(elastic_total_interacting_neutrons))
print("Percentage of interacting elastic neutrons: {:.2f}%".format(elastic_total_interacting_neutrons * 100/ N_samples))
print("\nTotal number of interacting anelastic neutrons: {:.2f}".format(anelastic_total_interacting_neutrons))
print("Percentage of interacting anelastic neutrons: {:.2f}%".format(anelastic_total_interacting_neutrons * 100/ N_samples))
print("\nTotal number of interacting alpha neutrons: {:.2f}".format(alpha_total_interacting_neutrons))
print("Percentage of interacting alpha neutrons: {:.3f}%".format(alpha_total_interacting_neutrons * 100/ N_samples))
print("\nTotal number of interacting proton neutrons: {:.2f}".format(proton_total_interacting_neutrons))
print("Percentage of interacting proton neutrons: {:.3f}%".format(proton_total_interacting_neutrons * 100/ N_samples))
print(" ")
print("Total number of interacting neutrons: {:.3f}".format(elastic_total_interacting_neutrons + anelastic_total_interacting_neutrons + alpha_total_interacting_neutrons + proton_total_interacting_neutrons))
print("Percentage of interacting neutrons: {:.3f}%".format((elastic_total_interacting_neutrons + anelastic_total_interacting_neutrons + alpha_total_interacting_neutrons + proton_total_interacting_neutrons) * 100/ N_samples))

Si_max_recoil_energy = silicon_recoil_energy(bin_centers, theta = 0) # Get the maximum recoil energy of silicon nuclei for each sampled neutron energy (theta = 0 for maximum recoil)
print("Maxium recoil energy: ", max(Si_max_recoil_energy))

# Plot the maximum recoil energy of silicon nuclei as a function of the sampled neutron energies
plt.figure(figsize=(8, 5))
plt.bar(Si_max_recoil_energy, elastic_hist_pdf_scaled, width=Si_max_recoil_energy[1] - Si_max_recoil_energy[0], alpha=0.6, label='Max Recoil Energy of Si Nuclei', color = 'goldenrod')
plt.xlabel('Recoil Energy (MeV)')
plt.ylabel('Counts')
plt.title('Maximum Recoil Energy of Silicon Nuclei')
plt.legend()
plt.grid()

# Plot the maximum recoil energy of silicon nuclei as a function of the sampled neutron energies normalized by the total number of sampled neutrons
plt.figure(figsize=(8, 5))
plt.bar(Si_max_recoil_energy, elastic_hist_pdf_scaled * 100 / N_samples, width=Si_max_recoil_energy[1] - Si_max_recoil_energy[0], alpha=0.6, label='Max Recoil Energy of Si Nuclei', color = 'goldenrod')
plt.xlabel('Recoil Energy (MeV)')
plt.ylabel('Normalized Counts (%)')
plt.title('Maximum Recoil Energy of Silicon Nuclei')
plt.legend()
plt.grid()

# # Getting the max silicon recoil energy
# MAX_RECOIL = max(Si_max_recoil_energy)
# print("Maximum recoil energy of silicon nuclei: {:.2f} MeV".format(MAX_RECOIL))
# print(" ")

# =========================================================
#                     PHOTONS ANALYSIS
# =========================================================

# Getting the data from the photon GEANT4 data file

# =========================================================
#                     COMPTON ANALYSIS
#                  NUMBER OF INTERACTIONS
# =========================================================

# Defining a distribution of photon energies for the Compton scattering analysis
# Making a Monte Carlo sampling(as done for neutrons) of photon energies from a Gaussian distribution
# Gaussian distribution centered at 4.4 MeV with a sigma of 100 keV
mean_photon_energy = 4440 # keV
sigma_photon_energy = 100 # keV
sample_photon = sample_photon_energies(mean=mean_photon_energy, sigma=sigma_photon_energy, N_samples=N_samples)

# Founding the energy histogram distribution
photon_energy_hist, photon_bin_edges = np.histogram(sample_photon, bins=N_bin, density=False)
photon_bin_centers = 0.5 * (photon_bin_edges[:-1] + photon_bin_edges[1:])
photon_bin_widths = np.diff(photon_bin_edges)

# Integrating the Klein-Nishina formula over the scattering angle to get the total cross section
angles = np.linspace(0, np.pi, 100000) # scattering angles from 0 to pi radians

# calculate the Klein-Nishina values for each sampled photon energy and scattering angle
# Matrix with rows corresponding to photon energies and columns corresponding to scattering angles
klein_nishina_values = Klein_Nishina(photon_bin_centers[:, np.newaxis], angles, True)

# total cross section for each sampled photon energy
compton_total_CS = integrate.simpson(klein_nishina_values, angles, axis=1)

# Printing the photon energy distribution
# plt.figure(figsize=(8, 5))
# plt.bar(photon_bin_centers, photon_energy_hist, width=photon_bin_widths, alpha=0.6, label='Sampled Photon Energies (Gaussian)', color = 'orchid')
# plt.plot(np.linspace(4000, 4800, 1000), norm.pdf(np.linspace(4000, 4800, 1000), mean_photon_energy, 100), label='Gaussian PDF', lw = 2, color = 'darkred', ls = '--')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Counts / Total counts')
# plt.title('Monte Carlo Sampling of Photon Energies [Gaussian] - {:.0f} samples'.format(N_samples))
# plt.legend()
# plt.grid()

# Plotting the total Compton cross section as a function of photon energy
plt.figure(figsize=(8, 5))
plt.plot(photon_bin_centers, compton_total_CS*10000/barn, label='Total Compton Cross Section', color = 'mediumvioletred')
plt.xlabel('Photon Energy (keV)')
plt.ylabel('Total Compton Cross Section (barn)')
plt.title('Total Compton Cross Section as a Function of Photon Energy')
plt.legend()
plt.grid()

# Plotting the Klein-Nishina cross section for the mean photon energy as a function of scattering angle
plt.figure(figsize=(8, 5))
plt.plot(angles, Klein_Nishina(mean_photon_energy, angles, True)*10000/barn, label='Klein-Nishina Cross Section (Mean Energy)', color = 'mediumvioletred')
plt.xlabel('Scattering Angle (degrees)')
plt.ylabel('Klein-Nishina Cross Section (barn)')
plt.title('Klein-Nishina Cross Section as a Function of Scattering Angle for {:.0f} keV Photons'.format(mean_photon_energy))
plt.legend()
plt.grid()

# Finding the compton probability for a photon in silicon
# Converting the total Compton cross section from m^2 to cm^2
compton_total_CS_cm2 = compton_total_CS * 1e4 # cm^2

# macroscopic cross section in cm^-1 for the sampled photon energies
mu = n_Si * compton_total_CS_cm2

# Number of photons that interact in the active silicon layer of arcadia making compton
photon_compton_hist_scaled = photon_energy_hist * (1 - np.exp(-mu * arcadia_thickness))

# Plotting the number of photons that make compton in arcadia making as a function of photon energy
plt.figure(figsize=(8, 5))
plt.bar(photon_bin_centers, photon_compton_hist_scaled, width=photon_bin_widths, alpha=0.6, label='Number of Photons that make Compton in Arcadia', color = 'darkorange')
plt.xlabel('Photon Energy (keV)')
plt.ylabel('Counts')
plt.title('Number of Photons that make Compton in Arcadia as a Function of Photon Energy')
plt.legend()
plt.grid()

# Finding the percentage of photons that make compton
# Avoid division by zero: where energy_hist is 0, set percentage to 0
photon_compton_percentage = np.divide(photon_compton_hist_scaled, photon_energy_hist, 
                                       where=photon_energy_hist!=0, 
                                       out=np.zeros_like(photon_compton_hist_scaled)) * 100
# plt.figure(figsize=(8, 5))
# plt.bar(photon_bin_centers, photon_compton_percentage, width=photon_bin_widths, alpha=0.6, label='Percentage of Photons that make Compton in Arcadia', color = 'darkred')
# plt.xlabel('Photon Energy (keV)')
# plt.ylabel('Percentage of Photons that make Compton in Arcadia (%)')
# plt.title('Percentage of Photons that make Compton in Arcadia')
# plt.legend()

total_compton_photons = np.sum(photon_compton_hist_scaled)
print(" ")
print("Total number of sampled photons: {:.0f}".format(N_samples))
print("Total number of photons that make Compton: {:.2f}".format(total_compton_photons))
print("Percentage of photons that make Compton: {:.2f}%".format(total_compton_photons * 100 / N_samples))


# =========================================================
#                     COMPTON ANALYSIS
#                 SCATTERING ANGLE SAMPLING
# =========================================================

# Getting the sample angles for each sampled photon energy
# Use the interacting-photon distribution to set the number of angular samples per bin.

# Getting the number of photons that interact for each energy bin
interacting_photon_counts = np.rint(photon_compton_hist_scaled).astype(int)
sampled_angles = []
for i, n_samples_interacting in enumerate(interacting_photon_counts):
    sampled_angles.append(
        sample_theta_kn(photon_bin_centers[i], n_samples_interacting, n_grid=100000)[0]
    )

# Plotting the sapling for a specific photon energy (the mean of the distribution)
mean_energy_index = np.argmin(np.abs(photon_bin_centers - mean_photon_energy))
plt.figure(figsize=(8, 5))
plt.hist(sampled_angles[mean_energy_index], bins=50, density=True, alpha=0.6, label='Sampled Scattering Angles (Mean Energy)', color = 'orchid')
plt.plot(angles, Klein_Nishina(photon_bin_centers[mean_energy_index], angles, True)/np.trapezoid(Klein_Nishina(photon_bin_centers[mean_energy_index], angles, True), angles), label='Klein-Nishina PDF (Mean Energy)', lw = 2, color = 'darkred', ls = '--')
plt.xlabel('Scattering Angle (radians)')
plt.ylabel('Density')
plt.title('Monte Carlo Sampling of Scattering Angles for {:.0f} keV Photons - {:.0f} samples'.format(photon_bin_centers[mean_energy_index], interacting_photon_counts[mean_energy_index]))
plt.legend()
plt.grid()

# Computing the CDF of the scattering angles for the mean photon energy
# CDF_mean_energy = integrate.cumulative_trapezoid(Klein_Nishina(photon_bin_centers[mean_energy_index], angles, True), angles, initial=0.0)
# CDF_mean_energy /= CDF_mean_energy[-1] # Normalize the CDF
# plt.figure(figsize=(8, 5))
# plt.plot(angles, CDF_mean_energy, label='CDF of Scattering Angles (Mean Energy)', color = 'mediumvioletred')
# plt.xlabel('Scattering Angle (radians)')
# plt.ylabel('CDF')
# plt.title('Cumulative Distribution Function of Scattering Angles for {:.0f} keV Photons'.format(photon_bin_centers[mean_energy_index]))
# plt.legend()
# plt.grid()


# =========================================================
#                     COMPTON ANALYSIS
#                SCATTERED ELECTRON ENERGIES
# =========================================================
# Making a guassian fit of the intercatin photon energy distribution
# Fit a Gaussian function to the photon energy histogram
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Initial guess for the fit parameters
initial_guess = [max(photon_compton_hist_scaled), mean_photon_energy, sigma_photon_energy]

# Perform the fit
popt, pcov = curve_fit(gaussian, photon_bin_centers, photon_compton_hist_scaled, p0=initial_guess)

print("Fitted Gaussian parameters - Interacting Photons:")
print("Mean (mu): {:.2f}".format(popt[1]))
print("Standard Deviation (sigma): {:.2f}".format(popt[2]))
print("Initial photons distribution: ")
print("Mean (mu): {:.2f}".format(mean_photon_energy))
print("Standard Deviation (sigma): {:.2f}".format(sigma_photon_energy))

N_points = 30000
N_bin = 125
#N_points = 356381
# N_points = 390450
# Sample the scattering angle
KN = sample_theta_kn(popt[1], N_points, n_grid=100000)[0]

# Plotting the klein nishina distribution for the fitted mean energy
plt.figure(figsize=(8, 5))
plt.hist(KN, bins=N_bin, density=True, alpha=0.5, label='Compton events', color = 'darkcyan')
plt.plot(angles, Klein_Nishina(popt[1], angles, True)/np.trapezoid(Klein_Nishina(popt[1], angles, True), angles), label='Klein-Nishina distribution', lw = 2, color = 'rebeccapurple', ls = '--')
plt.xlabel('Scattering Angle (radians)')
plt.ylabel(r'dN/d$\theta_\gamma$')
#plt.title('Monte Carlo Sampling of Scattering Angles for {:.0f} keV Photons - {:.0f} samples'.format(popt[1], N_points))
plt.legend()
plt.grid()

# Extract electron range values from file
# Energy in MeV and electron range in g/cm^2
electron_range_energies, electron_ranges, electron_std_ranges = get_range_data('electrons_range_from_NIST.txt')

# Normalizing the range for the silicon density and the energy in KeV
electron_ranges = (electron_ranges / 2.3212) * 10000 # in microns
electron_range_energies = electron_range_energies * 1000 # in keV
electron_std_ranges = (electron_std_ranges / 2.3212) * 10000 # in microns

# Computing the angle of the compton scattering for these specific energies
electron_scattering_angles = [scattered_electron_angle(popt[1], angle) for angle in KN]

# Plotting the electron ranges
plt.figure(figsize=(8, 5))
plt.errorbar(electron_range_energies, electron_ranges/1000, yerr=electron_std_ranges/1000, fmt='o', label='Electron Range Data', color = 'steelblue')
plt.xlabel('Electron Energy (keV)', size = 12)
plt.ylabel(r'Electron Range (mm)', size = 12)
plt.title('Electron Range in Silicon ')
plt.legend()
plt.grid()

# Defining a class to stock a single scattering events data
class scattering_event:

    def __init__(self, electron_energy, electron_angle, photon_energy, photon_angle):
        self.electron_energy = electron_energy
        self.electron_angle = electron_angle
        self.photon_energy = photon_energy
        self.photon_angle = photon_angle
        self.electron_range = np.interp(electron_energy, electron_range_energies, electron_ranges) # in microns
        solid_angle = simulate_solid_angle(electron_angle, self.electron_range, photon_incidation_angle)
        self.total_pixel_activated = pixel_activation(solid_angle[0], solid_angle[1])

        if self.total_pixel_activated >= 3 and self.electron_angle > np.radians(60):
            print("EVENT DETECTED: ", self.electron_angle, " ", self.electron_energy)
total_compton_events = []

# Getting the photon energy histogramm from this distribution
print(" ")
print("Start Compton pixel multiplicity simulation")
for simulated_angle in tqdm(KN):

    total_compton_events.append(scattering_event(scattered_electron_energy(popt[1], simulated_angle), 
                                                 scattered_electron_angle(popt[1], simulated_angle), 
                                                 scattered_photon_energy(popt[1], simulated_angle),
                                                 simulated_angle))

# Plotting the histogram of scattered photon energies
plt.figure(figsize=(8, 5))
photon_energies_from_events = [event.photon_energy for event in total_compton_events]
electron_energies_from_events = [event.electron_energy for event in total_compton_events]
plt.hist(photon_energies_from_events, bins=N_bin, density = True, alpha=0.6, label='Scattered Photon Energies (Fitted Mean Energy)', color = 'mediumvioletred')
plt.hist(electron_energies_from_events, bins=N_bin, density = True, alpha=0.6, label='Scattered Electron Energies (Fitted Mean Energy)', color = 'darkorange')
plt.xlabel('Scattered Photon Energy (keV)')
plt.ylabel('Density')
plt.title('Histogram of Scattered Photon Energies for {:.0f} keV Photons'.format(popt[1]))
plt.legend()
plt.grid()

# Founding the electron angle distribution
plt.figure(figsize=(8, 5))
electron_angles_from_events = [event.electron_angle for event in total_compton_events]
electron_hist_print = np.histogram(electron_angles_from_events, bins=N_bin, density = True)
plt.hist(electron_angles_from_events, bins=N_bin, density = True, alpha=0.7, label='Scattered Electron Angles (Fitted Mean Energy)', color = 'darkred')
plt.xlabel('Scattered Electron Angle (radians)')
plt.ylabel('Density')
plt.title('Histogram of Scattered Electron Angles for {:.0f} keV Photons'.format(popt[1]))
plt.legend()
plt.grid()

# Printing bin angle - counts for the electron distribution
print("Angle - Counts")
for i in range(len(electron_hist_print[0])):
    print("{:.4f} {:.4f}".format(electron_hist_print[1][i], electron_hist_print[0][i]))

# Plotting the electron angle distribution + electron energy as a function of angle in a double y-axis plot
plt.figure(figsize=(8, 5))
ax1 = plt.gca()
ax1.hist(electron_angles_from_events, bins=N_bin, density=True, alpha=0.7, label='Compton events', color = 'darkred')
ax1.set_xlabel('Scattered Electron Angle (radians)')
ax1.set_ylabel(r'dN/d$\theta_\mathrm{e}$')
ax1.tick_params(axis='y')
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
angles_array = np.array(electron_angles_from_events)
energies_array = np.array(electron_energies_from_events)
order = np.argsort(angles_array)
angles_sorted = angles_array[order]
energies_sorted = energies_array[order]
ax2.plot(angles_sorted, energies_sorted, alpha=1, color='darkorange', lw=2.5,
         label='Electron Energy')
ax2.set_ylabel('Energy (keV)')
ax2.tick_params(axis='y')
ax2.set_ylim(bottom=0)
# plt.title('Scattered Electron Angles and Energies for {:.0f} keV Photons'.format(popt[1]))

ax1.grid(True, axis='both')
ax2.grid(False)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

# Integrating the electron angle distribution from 45° to 70°
electron_bin_centers = 0.5 * (electron_hist_print[1][:-1] + electron_hist_print[1][1:])
angle_mask = (electron_bin_centers >= np.radians(45)) & (electron_bin_centers <= np.radians(70))
integrated_counts_compton = np.sum(electron_hist_print[0][angle_mask])

# Printing the proportion to the total number of events
print("Total compton with high multiplicity: {:.2f}%".format(integrated_counts_compton*100/np.sum(electron_hist_print[0])))

# Plotting the electron angles in function of the scattered photon angle
plt.figure(figsize=(8, 5))
plt.scatter([event.photon_angle for event in total_compton_events], [event.electron_angle for event in total_compton_events], alpha=0.6, ls = '-', color='darkgreen')
plt.xlabel('Scattered Photon Angle (radians)')
plt.ylabel('Scattered Electron Angle (radians)')
plt.title('Electron Angles vs. Scattered Photon Angles for {:.0f} keV Photons'.format(popt[1]))
plt.grid()

# Save histogram data to a text file
pixel_multiplicity_from_events = [event.total_pixel_activated for event in total_compton_events]
cluster_multiplicity, counts = np.unique(pixel_multiplicity_from_events, return_counts=True)
counts = counts # Multiply each count by 10
with open("simulated_cluster_multiplicity.txt", "a") as f:
    for cm, count in zip(cluster_multiplicity, counts):
        f.write(f"{cm}\t{count}\n")

# Plotting the histogram of simulated cluster multiplicity
plt.figure(figsize=(8, 5))
plt.bar(cluster_multiplicity, height=counts, alpha=0.6, label='Simulated Cluster Multiplicity (Fitted Mean Energy)', color = 'darkblue')
plt.xlabel('Cluster Multiplicity (Number of Activated Pixels)')
plt.ylabel('Density')
plt.title('Histogram of Simulated Cluster Multiplicity from Compton Scattering for {:.0f} keV Photons'.format(popt[1]))
plt.legend()
plt.grid()

plt.show()
