# THEORETICAL BACKGROUND FOR NEUTRON INTERACTION AND COMPTON SCATTERING

# Import necessary libraries
import pandas as pd
import numpy as np
import statistics
from scipy import integrate
from tqdm import tqdm
import matplotlib.pyplot as plt
from cycler import cycler
import mplhep as hep

from ARCADIA_HELPER import *

# Graph style
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

r_e = 2.818e-15  # classical electron radius in m
m_e = 511.0      # electron rest mass energy in keV
Z_SILICON = 14  # atomic number of silicon
# hv = 4438  # photon energy in keV
hv = 3900  # photon energy in keV
N_energy = 4.200 # neutron energy in MeV
m_n = 939.565 # neutron mass in MeV
m_Si = 28.0855 * 931.494 # silicon mass in MeV
E_neutron = N_energy + m_n

BARN_MEV = 1e31

# Function definition for compton cross section in function of Egamma and scattering angle
# Probability of finding a photon scattered at an angle x with energy Egamma' after the scattering
def compton_cross_section(x, Egamma):
    """Calculate the Compton scattering cross section for a given photon energy.
    \nParameters:
    \n - x : photon scattering angle in radians
    \n - Egamma : initial photon energy in keV
    \nReturns:
    \n - the function value for x
    """
    # Constants
    r_e = 2.818e-15  # classical electron radius in m
    m_e = 511.0      # electron rest mass energy in keV
    alpha = Egamma / m_e  # dimensionless photon energy

    return Z_SILICON * (r_e**2 * np.pi) * ( ((1+(np.cos(x))**2)*(1+alpha*(1-np.cos(x))) + (alpha**2)*(1-np.cos(x))**2) / (1 + alpha * (1-np.cos(x))) ) / (Egamma * alpha)

# Function definition for compton cross section in function of Egamma and scattered electron energy
def Klein_Nishina(Egamma, x, scatter_angle = False):
    """Calculate the Klein-Nishina formula for Compton scattering.
    \nParameters:
    \n - Egamma : initial photon energy in keV
    \n - x : photon scattering angle in radians
    \n - scatter_angle : if True, calculate the cross section for the scattered theta photon, otherwise for the solid angle
    \nReturns:
    \n - the Klein-Nishina differential cross section in m^2/keV
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

# Function definition for scattered cross section in function of initial photon energy and scattered electron energy
def Klein_Nishina_energy(Egamma, E_electron):
    """Calculate the Klein-Nishina formula for Compton scattering based on scattered photon energy.
    \nParameters:
    \n - Egamma : initial photon energy in keV
    \n - E_electron : scattered electron energy in keV
    \nReturns:
    \n - the Klein-Nishina differential cross section in m^2/keV
    """
    # Constants
    r_e = 2.818e-15  # classical electron radius in m
    m_e = 511.0      # electron rest mass energy in keV
    alpha = Egamma / m_e  # dimensionless photon energy

    E_electron = np.asarray(E_electron)
    Egamma_prime = Egamma - E_electron  # scattered photon energy in keV
    E_electron_max = (2 * Egamma**2) / (m_e + 2 * Egamma)  # maximum scattered electron energy in keV
    print("Max electron energy: ", E_electron_max)
    result = np.zeros_like(E_electron, dtype=float)
    valid = E_electron < E_electron_max
    result[valid] = (
        Z_SILICON * (np.pi * r_e**2)
        * (
            2
            - 2 * (E_electron[valid] / (alpha * Egamma_prime[valid]))
            + (E_electron[valid] ** 2 / (alpha * Egamma_prime[valid]) ** 2)
            + (E_electron[valid] ** 2 / (Egamma * Egamma_prime[valid]))
        )
        / (alpha * Egamma)
    )

    return result.item() if result.ndim == 0 else result

# Photon energy after scattering
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

# Electron energy after scattering
def scattered_electron_energy(Egamma, x):
    """Calculate the energy of the scattered electron after Compton scattering.
    \nParameters:
    \n - Egamma : initial photon energy in keV
    \n - x : photon scattering angle in radians
    \nReturns:
    \n - the energy of the scattered electron in keV
    """
    return Egamma - scattered_photon_energy(Egamma, x)

# Function to sample scattering angles from the Klein-Nishina distribution using inverse transform sampling
def sample_theta_kn(Egamma, n_samples, n_grid=20000, rng=None):
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

# Function definition for silicon recoil energy after neutron scattering
def silicon_recoil_energy_after_neutron_scattering(E_n, theta):
    """Calculate the recoil energy of a silicon nucleus after neutron scattering.
    \nParameters:
    \n - E_n : initial neutron energy in MeV (kinetic + rest mass)
    \n - theta : neutron scattering angle in radians
    \nReturns:
    \n - the recoil energy of the silicon nucleus in MeV
    """

    p = np.sqrt(E_n**2 - m_n**2)  # neutron momentum in MeV

    E_recoil = m_Si * ((E_n + m_Si)**2 + (p * np.cos(theta))**2) / ((E_n + m_Si)**2 - (p * np.cos(theta))**2) - m_Si
    return E_recoil # return in MeV

# Function to convert an angle from radians to degrees
def to_deg(angle_rad):
    """Convert an angle from radians to degrees.
    \nParameters:
    \n - angle_rad : angle in radians
    \nReturns:
    \n - the angle in degrees
    """
    return angle_rad * 180 / np.pi

# Function to convert an angle from degrees to radians
def to_rad(angle_deg):
    """Convert an angle from degrees to radians.
    \nParameters:
    \n - angle_deg : angle in degrees
    \nReturns:
    \n - the angle in radians
    """
    return angle_deg * np.pi / 180

OFFSET_ANGLE = to_rad(15) # offset angle in radians

print("COMPTON SCATTERING")
print(" ")
print("Photon energy: ", hv, "keV")
print(" ")

Ek = np.linspace(0, (2 * hv**2) / (m_e + 2 * hv) + 0.1, 10000)
x = np.linspace(0, np.pi, 10000)  # scattering angle from 0 to pi radians

y = Klein_Nishina_energy(hv, Ek)

# Plotting the Klein-Nishina cross section as a function of the scattered electron energy
plt.figure(figsize=(6,4))
plt.plot(Ek, y * BARN_MEV, label=f'Egamma = {hv} keV')
plt.xlabel(r'Scattered Electron Energy $E_{e}$ (keV)')
plt.ylabel(r'Klein-Nishina Cross Section (barn/MeV)')
plt.title('Klein-Nishina Cross Section vs Scattered Electron Energy')
plt.legend(loc = 'upper right')
plt.grid()
plt.yscale ('log')
#plt.savefig("./Klein_Nishina_cross_section_electron_energy.png", dpi=300)

# # Plotting the Compton scattering cross section as a function of the scattering angle
# plt.figure(figsize=(6,4))
# y = compton_cross_section(x, hv)
# plt.plot(x, y * BARN_MEV, label=f'Egamma = {hv} keV')
# plt.xlabel(r'Photon Scattering $\theta$')
# plt.ylabel(r'$d\sigma/dE_{\gamma 2}$ (barn/MeV)')
# plt.title('Compton Scattering Cross Section vs Scattering Angle')
# plt.legend(loc = 'lower right')
# plt.grid()
# plt.savefig("./compton_cross_section.png", dpi=300)

# Plotting the scattered photon energy and scattered electron energy as a function of the scattering angle
plt.figure(figsize=(6,4))
y = scattered_photon_energy(hv, x)
y2 = scattered_electron_energy(hv, x)
plt.plot(x, y, label=f'Egamma')
plt.plot(x, y2, label=f'Eelectron')
plt.xlabel(r'Scattering angle')
plt.ylabel(r'Scattered Energy $E_{\gamma 2}$ (keV)')
plt.title('Scattered Energy vs Scattering Angle')
plt.legend(loc = 'upper right')
plt.grid()
#plt.savefig("./scattered_energy.png", dpi=300)

# Plotting the Klein-Nishina cross section as a function of the scattering angle
plt.figure(figsize=(6,4))
y = Klein_Nishina(hv, x)
plt.plot(x, y * BARN_MEV, label=f'Egamma = {hv} keV')
plt.xlabel(r'Photon Scattering $\theta$')
plt.ylabel(r'Klein-Nishina Cross Section (barn/MeV)')
plt.title('Klein-Nishina Cross Section vs Scattering Angle')
plt.legend(loc = 'upper right')
plt.grid()
#plt.savefig("./Klein_Nishina_cross_section.png", dpi=300)

# Converting from the total solid angle to the scattering angle and integrating
# Integrating the Klein-Nishina cross section over the scattering angle to get the total cross section
tot_cross_sections_solid = integrate.simpson(Klein_Nishina(hv, x, True), x)

print("Total cross section with solid angle: ", tot_cross_sections_solid * 1e28, "barns")
# dsigma / dOmega in function of theta
Klein_Nishina_PDF = Klein_Nishina(hv, x) / tot_cross_sections_solid

# Plotting the Klein-Nishina PDF as a function of the scattering angle
plt.figure(figsize=(6,4))
plt.plot(x, Klein_Nishina_PDF, label=f'Egamma = {hv} keV')
plt.xlabel(r'Photon Scattering $\theta$')
plt.ylabel(r'Klein-Nishina PDF')
plt.title('Klein-Nishina PDF vs Scattering Angle')
plt.legend(loc = 'upper right')
plt.grid()
#plt.savefig("./Klein_Nishina_PDF.png", dpi=300)

# Plotting the scattered electron energy multiplied by the Klein-Nishina cross section as a function of the scattering angle
plt.figure(figsize=(6,4))
plt.plot(x, scattered_electron_energy(hv, x) * Klein_Nishina(hv, x) * BARN_MEV, label=f'Egamma = {hv} keV')
plt.xlabel(r'Photon Scattering $\theta$')
plt.ylabel(r'Scattered Electron Energy $E_{e}$ (keV) * Klein-Nishina Cross Section (barn/MeV)')
plt.title('Scattered Electron Energy * Klein-Nishina Cross Section vs Scattering Angle')
plt.legend(loc = 'upper right')
plt.grid()
#plt.savefig("./scattered_electron_energy_Klein_Nishina.png", dpi=300)

tot_cross_section_energy = integrate.simpson(Klein_Nishina(hv, x, True) * scattered_electron_energy(hv, x), x)

mean_angle = integrate.simpson(x * Klein_Nishina_PDF * 2 * np.pi * np.sin(x), x)
mean_E_electron = integrate.simpson(2 * np.pi * np.sin(x) * scattered_electron_energy(hv, x) * Klein_Nishina_PDF, x)

print("Mean scattered electron energy: ", mean_E_electron)
print("Photon mean angle: ", to_deg(mean_angle))

mean_electron_angle = np.pi / 2 - np.arctan((1 + hv / m_e) * np.tan(mean_angle / 2))
print("Mean electron angle: ", to_deg(mean_electron_angle))

# Sampling from the Klein-Nishina distribution
Egamma = hv # photon energy in keV
samples, theta_grid, pdf_grid = sample_theta_kn(Egamma, n_samples=200000)

# Plotting the histogram of the sampled angles and the target PDF
plt.figure(figsize=(6,4))
plt.hist(samples, bins=120, density=True, alpha=0.5, label="samples")
plt.plot(theta_grid, pdf_grid, "r-", lw=2, label="target PDF")
plt.xlabel(r"$\theta$ [rad]")
plt.ylabel("density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Histogram of Sampled Scattering Angles vs Target PDF')
#plt.savefig("./sampled_angles_histogram.png", dpi=300)

# Plotting the electron energy for the sampled angles
electron_energy_samples = scattered_electron_energy(Egamma, samples)
plt.figure(figsize=(6,4))
plt.hist(electron_energy_samples, bins=120, density=True, alpha=0.5, label="samples")
plt.xlabel(r'Scattered Electron Energy $E_{e}$ (keV)')
plt.ylabel("density")
plt.title('Scattered Electron Energy Distribution from Sampled Angles')
plt.legend()
plt.grid(True, alpha=0.3)
#plt.savefig("./sampled_electron_energy_histogram.png", dpi=300)

# Check 1: histogram area should be ~1 when density=True
# mask_0_3000 = (electron_energy_samples >= 0.0) & (electron_energy_samples <= 3000.0)
# electron_energy_samples_0_3000 = electron_energy_samples[mask_0_3000]
# hist_density, hist_edges = np.histogram(
#     electron_energy_samples_0_3000,
#     bins=120,
#     range=(0.0, 3000.0),
#     density=True,
# )
# hist_integral_0_3000 = np.sum(hist_density * np.diff(hist_edges))
# fraction_0_3000 = electron_energy_samples_0_3000.size / electron_energy_samples.size

# print("Integral(histogram PDF in [0,3000]) =", hist_integral_0_3000)
# print("P(0 <= E_e <= 3000) from samples =", fraction_0_3000)

# Searching for pixels
nu_pixels = ARCADIA_THICKNESS * np.tan(OFFSET_ANGLE + mean_electron_angle) / 25 
print("Mean number of pixels: ", nu_pixels)

# STUDING THE NEUTRON SCATTERING
print(" ")
print("NEUTRON SCATTERING")
print(" ")
print("Neutron energy: ", N_energy, "MeV")
print(" ")
beta = np.sqrt(E_neutron**2 - m_n**2) / E_neutron
beta_cm = np.sqrt(E_neutron**2 - m_n**2) / (E_neutron + m_Si)

theta_max_cm = np.arccos(-beta_cm/beta)

phi_si_max = np.arctan(np.sin(theta_max_cm) * np.sqrt(1 - beta_cm**2) / (1- np.cos(theta_max_cm)))

print("Max phi (Si):", to_deg(phi_si_max))

# Plotting the silicon recoil energy as a function of the neutron scattering angle
plt.figure(figsize=(6,4))
theta_lab = np.linspace(0, phi_si_max, 10000)
E_recoil_lab = silicon_recoil_energy_after_neutron_scattering(E_neutron, theta_lab)
plt.plot(theta_lab, E_recoil_lab, label=f'E_n = {N_energy} MeV')
plt.xlabel(r'Neutron Scattering Angle $\theta$ (rad)')
plt.ylabel(r'Silicon Recoil Energy $E_{recoil}$ (MeV)')
plt.title('Silicon Recoil Energy vs Neutron Scattering Angle')
plt.legend(loc = 'upper right')
plt.grid()
#plt.savefig("./silicon_recoil_energy.png", dpi=300)

print("Max silicon recoil energy: ", np.max(E_recoil_lab) * 1000, "keV")
print("Mean silicon recoil energy: ", np.mean(E_recoil_lab) * 1000, "keV")
plt.show()
