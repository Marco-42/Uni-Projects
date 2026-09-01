# Finger_plot.py data analysis and visualization for SiPM (Silicon photon multipliers) data

# Import necessary libraries
import ast
import numpy as np
import statistics
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from cycler import cycler
import mplhep as hep
import SiPM_helper as hp
import os
from scipy.stats import landau

# Graph style
# Setting plot style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '14',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (8, 6),
         'axes.labelsize': '14',
         'figure.titlesize' : '14',
         'axes.titlesize':'14',
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

bin_size = 10
pedestal_position = 2055.5
pedestal_fitting_interval = 100
# Taking data from files
column_number_HG = hp.get_column_number(2, "HG", "C", 27)
column_number_LG = hp.get_column_number(2, "LG", "C", 27)

path_origin = "./Radioactive_data/WITH_GAGG_HG33/"

print("Collum: ", column_number_HG)
peaks_positions = []
peaks_positions_err = []

# =======================================
#         Taking data using LYSO
# =======================================

# Taking data of the LYSO source
ADC_LYSO_HG, counts_LYSO_HG = hp.extract_SiPM_data(path_origin + "LYSO_40.0/gain33_60/data.csv", column_number_HG, False)

# Taking data of cobalt source
ADC_Co_HG, counts_Co_HG = hp.extract_SiPM_data(path_origin + "COBALT_40.0/gain33_60/data.csv", column_number_HG, False)

# Taking data of cesium source
ADC_Cs_HG, counts_Cs_HG = hp.extract_SiPM_data(path_origin + "CESIUM_40.0/gain33_60/data.csv", column_number_HG, False)

# Taking data of sodium source
ADC_Na_HG, counts_Na_HG = hp.extract_SiPM_data(path_origin + "SODIUM_40.0/gain33_60/data.csv", column_number_HG, False)


# Normalizing the data
ADC_LYSO_HG, counts_LYSO_HG, LYSO_HG_err = hp.normalize_data_with_pedestal(ADC_LYSO_HG, counts_LYSO_HG, sy=np.sqrt(counts_LYSO_HG), pedestal_last_point=pedestal_position + 70)
ADC_Co_HG, counts_Co_HG, Co_HG_err = hp.normalize_data_with_pedestal(ADC_Co_HG, counts_Co_HG, sy=np.sqrt(counts_Co_HG), pedestal_last_point=pedestal_position + 70)
ADC_Cs_HG, counts_Cs_HG, Cs_HG_err = hp.normalize_data_with_pedestal(ADC_Cs_HG, counts_Cs_HG, sy=np.sqrt(counts_Cs_HG), pedestal_last_point=pedestal_position + 70)
ADC_Na_HG, counts_Na_HG, Na_HG_err = hp.normalize_data_with_pedestal(ADC_Na_HG, counts_Na_HG, sy=np.sqrt(counts_Na_HG), pedestal_last_point=pedestal_position + 70)

# Rebinning data
ADC_LYSO_HG, counts_LYSO_HG, LYSO_HG_err = hp.rebin_xy(ADC_LYSO_HG, counts_LYSO_HG, bin_size, x_mode="mean", y_mode="sum", sy = LYSO_HG_err)
ADC_Co_HG, counts_Co_HG, Co_HG_err = hp.rebin_xy(ADC_Co_HG, counts_Co_HG, bin_size, x_mode="mean", y_mode="sum", sy = Co_HG_err)
ADC_Cs_HG, counts_Cs_HG, Cs_HG_err = hp.rebin_xy(ADC_Cs_HG, counts_Cs_HG, bin_size, x_mode="mean", y_mode="sum", sy= Cs_HG_err)
ADC_Na_HG, counts_Na_HG, Na_HG_err = hp.rebin_xy(ADC_Na_HG, counts_Na_HG, bin_size, x_mode="mean", y_mode="sum", sy= Na_HG_err)

# =======================================
#          Pedestal gaussian fit
# =======================================

# Taking data for each pedestal fit
pedestal_mask = (ADC_LYSO_HG > pedestal_position - pedestal_fitting_interval) & (ADC_LYSO_HG < pedestal_position + pedestal_fitting_interval)

# Fitting(with a gaussian function) the pedestal for each source
PED_LYSO_popt, PED_LYSO_pcov = hp.gauss_fit(ADC_LYSO_HG[pedestal_mask], counts_LYSO_HG[pedestal_mask], LYSO_HG_err[pedestal_mask])
PED_Co_popt, PED_Co_pcov = hp.gauss_fit(ADC_Co_HG[pedestal_mask], counts_Co_HG[pedestal_mask], Co_HG_err[pedestal_mask])
PED_Cs_popt, PED_Cs_pcov = hp.gauss_fit(ADC_Cs_HG[pedestal_mask], counts_Cs_HG[pedestal_mask], Cs_HG_err[pedestal_mask])
PED_Na_popt, PED_Na_pcov = hp.gauss_fit(ADC_Na_HG[pedestal_mask], counts_Na_HG[pedestal_mask], Na_HG_err[pedestal_mask])

# Printing the fitted pedestal values
print("Pedestal fit results:")
print(f"LYSO: mean = {PED_LYSO_popt[1]:.2f} +- {PED_LYSO_pcov[1]:.2f}, sigma = {PED_LYSO_popt[2]:.2f}")
print(f"Cobalt: mean = {PED_Co_popt[1]:.2f} +- {PED_Co_pcov[1]:.2f}, sigma = {PED_Co_popt[2]:.2f}")
print(f"Cesium: mean = {PED_Cs_popt[1]:.2f} +- {PED_Cs_pcov[1]:.2f}, sigma = {PED_Cs_popt[2]:.2f}")
print(f"Sodium: mean = {PED_Na_popt[1]:.2f} +- {PED_Na_pcov[1]:.2f}, sigma = {PED_Na_popt[2]:.2f}")

# Subtract the fitted pedestal values
ADC_LYSO_HG -= PED_LYSO_popt[1]
ADC_Co_HG -= PED_Co_popt[1]
ADC_Cs_HG -= PED_Cs_popt[1]
ADC_Na_HG -= PED_Na_popt[1]

# Mask the data
mask = (ADC_LYSO_HG >= 0) 
ADC_LYSO_HG = ADC_LYSO_HG[mask]
counts_LYSO_HG = counts_LYSO_HG[mask]
LYSO_HG_err = LYSO_HG_err[mask]
mask = (ADC_Co_HG >= 0)
ADC_Co_HG = ADC_Co_HG[mask]
counts_Co_HG = counts_Co_HG[mask]
Co_HG_err = Co_HG_err[mask]
mask = (ADC_Cs_HG >= 0)
ADC_Cs_HG = ADC_Cs_HG[mask]
counts_Cs_HG = counts_Cs_HG[mask]
Cs_HG_err = Cs_HG_err[mask]
mask = (ADC_Na_HG >= 0)
ADC_Na_HG = ADC_Na_HG[mask]
counts_Na_HG = counts_Na_HG[mask]
Na_HG_err = Na_HG_err[mask]

# Plotting all the data together
plt.figure(figsize=(8, 6))
plt.plot(ADC_LYSO_HG, counts_LYSO_HG, label="LYSO", ls='-', color = 'dodgerblue')
plt.plot(ADC_Co_HG, counts_Co_HG, label="Cobalt", ls='-', color = 'darkred')
plt.plot(ADC_Cs_HG, counts_Cs_HG, label="Cesium", ls='-', color = 'darkorange')
plt.plot(ADC_Na_HG, counts_Na_HG, label="Sodium", ls='-', color = 'green')
plt.xlabel("ADC")
plt.ylabel("Normalized Counts")
plt.legend()
plt.xlim()

# Fitting each source peak with a gaussian distribution and printing the results
# LYSO1 - LYSO2 - LYSO3 - SODIUM1 - SODIUM2 - CESIUM1 - COBALT1 
visual_peak_positions = [510, 780, 1250, 1100, 2800, 1550, 3050, 3500]
intervals = [(350, 2000), (920, 1300), (2480, 3220), (1340, 1800), (2770, 3820)] # --> FOR HG
#intervals = [(200, 1500), (645, 970), (1800, 2400), (940, 1600), (2015, 3100)] # --> FOR LG

# Fitting next of each peaks FOR HIGH GAIN
fitting_mask = (ADC_LYSO_HG < intervals[0][1]) & (ADC_LYSO_HG > intervals[0][0])
lyso1_popt, lyso1_pcov = hp.triple_gaussian_fit(ADC_LYSO_HG[fitting_mask], counts_LYSO_HG[fitting_mask], LYSO_HG_err[fitting_mask], p0=[0.6*max(counts_LYSO_HG[fitting_mask]), 520, 100, max(counts_LYSO_HG[fitting_mask]), 780, 120, 0.1*max(counts_LYSO_HG[fitting_mask]), 1280, 100])
fitting_mask = (ADC_Na_HG < intervals[1][1]) & (ADC_Na_HG > intervals[1][0])
na1_popt, na1_pcov = hp.gauss_fit(ADC_Na_HG[fitting_mask], counts_Na_HG[fitting_mask], Na_HG_err[fitting_mask])
fitting_mask = (ADC_Na_HG < intervals[2][1]) & (ADC_Na_HG > intervals[2][0])
na2_popt, na2_pcov = hp.gauss_fit(ADC_Na_HG[fitting_mask], counts_Na_HG[fitting_mask], Na_HG_err[fitting_mask])
fitting_mask = (ADC_Cs_HG < intervals[3][1]) & (ADC_Cs_HG > intervals[3][0])
cs1_popt, cs1_pcov = hp.gauss_fit(ADC_Cs_HG[fitting_mask], counts_Cs_HG[fitting_mask], Cs_HG_err[fitting_mask])
fitting_mask = (ADC_Co_HG < intervals[4][1]) & (ADC_Co_HG > intervals[4][0])
co1_popt, co1_pcov = hp.double_gaussian_fit(ADC_Co_HG[fitting_mask], counts_Co_HG[fitting_mask], Co_HG_err[fitting_mask], p0=[max(counts_Co_HG[fitting_mask]), 3020, 200, 0.7*max(counts_Co_HG[fitting_mask]), 3400, 500])

# Fitting next of each peaks FOR LOW GAIN
# fitting_mask = (ADC_LYSO_HG < intervals[0][1]) & (ADC_LYSO_HG > intervals[0][0])
# lyso1_popt, lyso1_pcov = hp.triple_gaussian_fit(ADC_LYSO_HG[fitting_mask], counts_LYSO_HG[fitting_mask], LYSO_HG_err[fitting_mask], p0=[0.6*max(counts_LYSO_HG[fitting_mask]), 360, 50, max(counts_LYSO_HG[fitting_mask]), 560, 75, 0.1*max(counts_LYSO_HG[fitting_mask]), 920, 100])
# fitting_mask = (ADC_Na_HG < intervals[1][1]) & (ADC_Na_HG > intervals[1][0])
# na1_popt, na1_pcov = hp.gauss_fit(ADC_Na_HG[fitting_mask], counts_Na_HG[fitting_mask], Na_HG_err[fitting_mask])
# fitting_mask = (ADC_Na_HG < intervals[2][1]) & (ADC_Na_HG > intervals[2][0])
# na2_popt, na2_pcov = hp.gauss_fit(ADC_Na_HG[fitting_mask], counts_Na_HG[fitting_mask], Na_HG_err[fitting_mask])
# fitting_mask = (ADC_Cs_HG < intervals[3][1]) & (ADC_Cs_HG > intervals[3][0])
# cs1_popt, cs1_pcov = hp.gauss_fit(ADC_Cs_HG[fitting_mask], counts_Cs_HG[fitting_mask], Cs_HG_err[fitting_mask])
# fitting_mask = (ADC_Co_HG < intervals[4][1]) & (ADC_Co_HG > intervals[4][0])
# co1_popt, co1_pcov = hp.double_gaussian_fit(ADC_Co_HG[fitting_mask], counts_Co_HG[fitting_mask], Co_HG_err[fitting_mask], p0=[max(counts_Co_HG[fitting_mask]), 2210, 240, 0.8*max(counts_Co_HG[fitting_mask]), 2470, 160])

# Plotting all the results(position in ADC of each peak) of the gaussian fits
print(" ")
print("Fitted peak positions (in ADC) +- sigma:")
print(f"LYSO1: {lyso1_popt[1]:.2f} +- {lyso1_popt[2]:.2f}")
print(f"LYSO2: {lyso1_popt[4]:.2f} +- {lyso1_popt[5]:.2f}")
print(f"LYSO3: {lyso1_popt[7]:.2f} +- {lyso1_popt[8]:.2f}")
print(f"Sodium1: {na1_popt[1]:.2f} +- {na1_popt[2]:.2f}")
print(f"Sodium2: {na2_popt[1]:.2f} +- {na2_popt[2]:.2f}")
print(f"Cesium1: {cs1_popt[1]:.2f} +- {cs1_popt[2]:.2f}")
print(f"Cobalt1: {co1_popt[1]:.2f} +- {co1_popt[2]:.2f}")
print(f"Cobalt2: {co1_popt[4]:.2f} +- {co1_popt[5]:.2f}")

sigma_err = [lyso1_pcov[2], lyso1_pcov[5], lyso1_pcov[8], na1_pcov[2], cs1_pcov[2], co1_pcov[2], co1_pcov[5]]
sigma_err = np.array(sigma_err)
mean_err = [lyso1_pcov[1], lyso1_pcov[4], lyso1_pcov[7], na1_pcov[1], cs1_pcov[1], co1_pcov[1], co1_pcov[4]]

# Plotting all the results(position in ADC of each peak) of the gaussian fits
print(" ")
print("Fitted peak positions (in ADC) +- sigma +- sigma_err:")
print(f"LYSO1: {lyso1_popt[1]:.2f} +- {np.sqrt(lyso1_popt[2]**2 + PED_LYSO_pcov[1]**2):.2f} +- {sigma_err[0]:.2f}")
print(f"LYSO2: {lyso1_popt[4]:.2f} +- {np.sqrt(lyso1_popt[5]**2 + PED_LYSO_pcov[1]**2):.2f} +- {sigma_err[1]:.2f}")
print(f"LYSO3: {lyso1_popt[7]:.2f} +- {np.sqrt(lyso1_popt[8]**2 + PED_LYSO_pcov[1]**2):.2f} +- {sigma_err[2]:.2f}")
print(f"Sodium1: {na1_popt[1]:.2f} +- {np.sqrt(na1_popt[2]**2 + PED_Na_pcov[1]**2):.2f} +- {sigma_err[3]:.2f}")
print(f"Cesium1: {cs1_popt[1]:.2f} +- {np.sqrt(cs1_popt[2]**2 + PED_Cs_pcov[1]**2):.2f} +- {sigma_err[4]:.2f}")
print(f"Cobalt1: {co1_popt[1]:.2f} +- {np.sqrt(co1_popt[2]**2 + PED_Co_pcov[1]**2):.2f} +- {sigma_err[5]:.2f}")
print(f"Cobalt2: {co1_popt[4]:.2f} +- {np.sqrt(co1_popt[5]**2 + PED_Co_pcov[1]**2):.2f} +- {sigma_err[6]:.2f}")

# Filling all the points
peaks_positions = [lyso1_popt[1], lyso1_popt[4], lyso1_popt[7], na1_popt[1], cs1_popt[1], co1_popt[1], co1_popt[4]]
peaks_positions_err = [lyso1_popt[2], lyso1_popt[5], lyso1_popt[8], na1_popt[2], cs1_popt[2], co1_popt[2], co1_popt[5]]

# Plotting all the data together with found peaks
plt.figure(figsize=(8, 6))
plt.errorbar(ADC_LYSO_HG, counts_LYSO_HG, yerr=LYSO_HG_err, label="LYSO", fmt='.', color = 'dodgerblue')
plt.errorbar(ADC_Co_HG, counts_Co_HG, yerr=Co_HG_err, label="Cobalt", fmt='.', color = 'darkred')
plt.errorbar(ADC_Cs_HG, counts_Cs_HG, yerr=Cs_HG_err, label="Cesium", fmt='.', color = 'darkorange')
plt.errorbar(ADC_Na_HG, counts_Na_HG, yerr=Na_HG_err, label="Sodium", fmt='.', color = 'green')

# Printing the fitted peak positions
plt.plot(np.linspace(intervals[0][0], intervals[0][1], 1000), hp.triple_gaussian(np.linspace(intervals[0][0], intervals[0][1], 1000), *lyso1_popt), label="LYSO Fit", color='dodgerblue', ls='--')
plt.plot(np.linspace(intervals[1][0], intervals[1][1], 1000), hp.gauss_function(np.linspace(intervals[1][0], intervals[1][1], 1000), *na1_popt), label="Sodium Fit 1", color='green', ls='--')
plt.plot(np.linspace(intervals[2][0], intervals[2][1], 1000), hp.gauss_function(np.linspace(intervals[2][0], intervals[2][1], 1000), *na2_popt), label="Sodium Fit 2", color='green', ls='--')
plt.plot(np.linspace(intervals[3][0], intervals[3][1], 1000), hp.gauss_function(np.linspace(intervals[3][0], intervals[3][1], 1000), *cs1_popt), label="Cesium Fit", color='darkorange', ls='--')
plt.plot(np.linspace(intervals[4][0], intervals[4][1], 1000), hp.double_gaussian(np.linspace(intervals[4][0], intervals[4][1], 1000), *co1_popt), label="Cobalt Fit", color='darkred', ls='--')

# Plotting the single gaussian function from the triple gaussian fit
# plt.plot(np.linspace(intervals[0][0], intervals[0][1], 1000), hp.gauss_function(np.linspace(intervals[0][0], intervals[0][1], 1000), lyso1_popt[0], lyso1_popt[1], lyso1_popt[2]), label="LYSO Fit 1", color='dodgerblue', ls=':')
# plt.plot(np.linspace(intervals[0][0], intervals[0][1], 1000), hp.gauss_function(np.linspace(intervals[0][0], intervals[0][1], 1000), lyso1_popt[3], lyso1_popt[4], lyso1_popt[5]), label="LYSO Fit 2", color='dodgerblue', ls=':')
# plt.plot(np.linspace(intervals[0][0], intervals[0][1], 1000), hp.gauss_function(np.linspace(intervals[0][0], intervals[0][1], 1000), lyso1_popt[6], lyso1_popt[7], lyso1_popt[8]), label="LYSO Fit 3", color='dodgerblue', ls=':')
plt.xlabel("ADC")
plt.ylabel("Normalized Counts")
plt.legend(ncol = 2)
plt.xlim(0, 5000)

# LYSO1 - LYSO2 - LYSO3 - Sodium1 - Sodium2 - Cesium1 - Cobalt1 - Cobalt2
calibration_energies = [202 + 88, 307+ 88, 307 + 200 + 88, 511, 662, 1170, 1330]

exp_energies = np.array(peaks_positions)
exp_energies_err = np.array(peaks_positions_err)
calibration_energies = np.array(calibration_energies)

# exp_energies_err = np.sqrt(peaks_positions_err**2 + pe)
# Making a linear fit with x and y data - CALIBRATION CURVE
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR

# 1. Define your model function (parameters as first argument, x as second)
def fit_function(beta, x):
    # Example: Linear model y = m*x + c
    return beta[0] * x + beta[1]

# 3. Create a model and data object and fit removing the third lyso peak
mask_333 = [True, True, False, True, True, True, True]  # Mask to exclude the third peak of LYSO
model = Model(fit_function)
data = RealData(exp_energies[mask_333], calibration_energies[mask_333], sx=exp_energies_err[mask_333])  # Assuming no error in calibration energies

# 4. Initialize ODR with data, model, and initial parameter guesses
odr_run = ODR(data, model, beta0=[2.0, 0.0])  # Initial guess for slope and intercept

# 5. Run the regression
output = odr_run.run()

# Display results
output.pprint()
# Plotting the data and the linear fit
plt.figure(figsize=(8, 6))
plt.errorbar(exp_energies[mask_333], calibration_energies[mask_333], xerr=exp_energies_err[mask_333], yerr=0, fmt='o', label='Data', markersize=5, color='black')
plt.errorbar(exp_energies[2], calibration_energies[2], xerr=exp_energies_err[2], yerr=0, fmt='o', label='LYSO 3rd Peak', markersize=5, color='purple')
plt.errorbar(na2_popt[1], 1062, yerr=0, xerr=na2_popt[2], fmt='o', markersize=5, color='purple')
plt.plot(np.linspace(-100, 5000), output.beta[0] * np.linspace(-100, 5000) + output.beta[1], label='Linear Fit', color='red')
plt.text(exp_energies[0] + 400, calibration_energies[0]-20, r"LYSO 290 KeV", fontsize=10, color='black')
plt.text(exp_energies[1] + 400, calibration_energies[1]-20, r"LYSO 395 KeV", fontsize=10, color='black')
plt.text(exp_energies[4] + 400, calibration_energies[4]-20, r"$^{137}Cs$ 662 KeV", fontsize=10, color='black')
plt.text(exp_energies[3] + 400, calibration_energies[3]-20, r"$^{22}Na$ 511 KeV", fontsize=10, color='black')
plt.text(exp_energies[3] - 1000, calibration_energies[3]+100, r"LYSO 597 KeV", fontsize=10, color='black')
plt.text(exp_energies[5] - 1200, calibration_energies[5], r"$^{57}Co$ 1170 KeV", fontsize=10, color='black')
plt.text(exp_energies[6] - 1200, calibration_energies[6], r"$^{60}Co$ 1330 KeV", fontsize=10, color='black')
plt.ylabel("Energy (keV)")
plt.xlabel("ADC")
plt.legend(loc = 'upper left')

# Print fitting results
print("Linear fit data")
print(f"Slope = {output.beta[0]:.4f} ± {output.sd_beta[0]:.4f} KeV/ADC")
print(f"Q = {output.beta[1]:.4f} ± {output.sd_beta[1]:.4f} KeV")

experimental_energy_in_ADC = exp_energies
experimental_sigma_in_ADC = exp_energies_err
error_on_sigma = sigma_err

resolution_in_ADC = 2*experimental_sigma_in_ADC / experimental_energy_in_ADC
resolution_in_ADC_err = 2*resolution_in_ADC * np.sqrt((error_on_sigma / experimental_sigma_in_ADC)**2 + (mean_err/ experimental_energy_in_ADC)**2)

# Plotting the resolution in ADC
plt.figure(figsize=(8, 6))
plt.errorbar(experimental_energy_in_ADC, 100*resolution_in_ADC, yerr=100*resolution_in_ADC_err, fmt='o', label='Data', markersize=5, color='black')
plt.xlabel("Energy (ADC)")
plt.ylabel("Resolution (2σ/E)[%]")
plt.title("Resolution vs Energy in ADC")
plt.legend()


print(" ")
print("Resolution in ADC:")
for i, res in enumerate(100*resolution_in_ADC):
    print(f"  {100*resolution_in_ADC[i]:.2f} ± {100*resolution_in_ADC_err[i]:.2f} %")


# Fitting a 1/sqrt(E) function to the resolution in ADC removing the lyso third peak
def inverse_sqrt_function(x, a, b, c):
    return np.sqrt(a**2 / x + b**2 + c**2 / x**2)

# Fitting the resolution in ADC but removing the third peak of lyso (third element of the array)
fit_mask = np.array([True, True, False, True, True, True, True])  # Mask to exclude the third peak of LYSO
popt_res, pcov_res = hp.curve_fit(inverse_sqrt_function, calibration_energies[fit_mask], 100*resolution_in_ADC[fit_mask], sigma=resolution_in_ADC_err[fit_mask], p0 = [610, 0, 30])


# Plotting the resolution in ADC but removing the third peak of lyso (third element of the array)
plt.figure(figsize=(8, 6))
plt.errorbar([q for q in calibration_energies if q != calibration_energies[2]], 
             [100*r for r in resolution_in_ADC if r != resolution_in_ADC[2]], 
             yerr=[100*err for err in resolution_in_ADC_err if err != resolution_in_ADC_err[2]], 
             fmt='o', label='Data', markersize=5, color='black')
plt.plot(np.linspace(0, 4000, 10000), inverse_sqrt_function(np.linspace(0, 4000, 10000), *popt_res), label='Fit (excluding LYSO 3rd peak)', color='darkorange')

# Plotting in red the third peak of lyso
plt.errorbar(calibration_energies[2], 100*resolution_in_ADC[2], yerr=100*resolution_in_ADC_err[2], fmt='o', label='LYSO 3rd Peak', markersize=5, color='red')

plt.xlabel("Energy (KeV)")
plt.ylabel("Resolution (2σ/E)[%]")
plt.title("Resolution vs Energy in ADC (excluding third peak)")
plt.legend()


# Converting all my original points in KeV
LYSO_KEV, LYSO_KEV_err = fit_function(output.beta, ADC_LYSO_HG), np.sqrt((output.sd_beta[0] * ADC_LYSO_HG)**2 + output.sd_beta[1]**2)
SODIUM_KEV, SODIUM_KEV_err = fit_function(output.beta, ADC_Na_HG), np.sqrt((output.sd_beta[0] * ADC_Na_HG)**2 + output.sd_beta[1]**2)
CESIUM_KEV, CESIUM_KEV_err = fit_function(output.beta, ADC_Cs_HG), np.sqrt((output.sd_beta[0] * ADC_Cs_HG)**2 + output.sd_beta[1]**2)
COBALT_KEV, COBALT_KEV_err = fit_function(output.beta, ADC_Co_HG), np.sqrt((output.sd_beta[0] * ADC_Co_HG)**2 + output.sd_beta[1]**2)

# Redo all the gaussian fitting in KeV
intervals = [(fit_function(output.beta, 350), fit_function(output.beta, 2000)), (fit_function(output.beta, 920), fit_function(output.beta, 1300)), (fit_function(output.beta, 2480), fit_function(output.beta, 3220)), (fit_function(output.beta, 1340), fit_function(output.beta, 1800)), (fit_function(output.beta, 2770), fit_function(output.beta, 3820))] # --> FOR HG
#intervals = [(200, 1500), (645, 970), (1800, 2400), (940, 1600), (2015, 3100)] # --> FOR LG

# Fitting next of each peaks FOR HIGH GAIN
fitting_mask = (LYSO_KEV < intervals[0][1]) & (LYSO_KEV > intervals[0][0])
lyso1_popt, lyso1_pcov = hp.triple_gaussian_fit(LYSO_KEV[fitting_mask], counts_LYSO_HG[fitting_mask], LYSO_HG_err[fitting_mask], p0=[0.6*max(counts_LYSO_HG[fitting_mask]), fit_function(output.beta, 520), output.beta[0]*100, max(counts_LYSO_HG[fitting_mask]), fit_function(output.beta, 780), output.beta[0]*120, 0.1*max(counts_LYSO_HG[fitting_mask]), fit_function(output.beta, 1280), output.beta[0]*100])
fitting_mask = (SODIUM_KEV < intervals[1][1]) & (SODIUM_KEV > intervals[1][0])
na1_popt, na1_pcov = hp.gauss_fit(SODIUM_KEV[fitting_mask], counts_Na_HG[fitting_mask], Na_HG_err[fitting_mask])
fitting_mask = (SODIUM_KEV < intervals[2][1]) & (SODIUM_KEV > intervals[2][0])
na2_popt, na2_pcov = hp.gauss_fit(SODIUM_KEV[fitting_mask], counts_Na_HG[fitting_mask], Na_HG_err[fitting_mask])
fitting_mask = (CESIUM_KEV < intervals[3][1]) & (CESIUM_KEV > intervals[3][0])
cs1_popt, cs1_pcov = hp.gauss_fit(CESIUM_KEV[fitting_mask], counts_Cs_HG[fitting_mask], Cs_HG_err[fitting_mask])
fitting_mask = (COBALT_KEV < intervals[4][1]) & (COBALT_KEV > intervals[4][0])
co1_popt, co1_pcov = hp.double_gaussian_fit(COBALT_KEV[fitting_mask], counts_Co_HG[fitting_mask], Co_HG_err[fitting_mask], p0=[max(counts_Co_HG[fitting_mask]), fit_function(output.beta, 3020), output.beta[0]*200, 0.7*max(counts_Co_HG[fitting_mask]), fit_function(output.beta, 3400), output.beta[0]*500])

# Plotting the converted spectrums with gaussian fit
plt.figure(figsize=(8, 6))
plt.errorbar(LYSO_KEV, counts_LYSO_HG, xerr=LYSO_KEV_err, yerr=LYSO_HG_err, label="LYSO", fmt='.', color = 'dodgerblue')
plt.errorbar(SODIUM_KEV, counts_Na_HG, xerr=SODIUM_KEV_err, yerr=Na_HG_err, label="Na-22", fmt='.', color = 'forestgreen')
plt.errorbar(CESIUM_KEV, counts_Cs_HG, xerr=CESIUM_KEV_err, yerr=Cs_HG_err, label="Cs-137", fmt='.', color = 'firebrick')
plt.errorbar(COBALT_KEV, counts_Co_HG, xerr=COBALT_KEV_err, yerr=Co_HG_err, label="Co-60", fmt='.', color = 'indigo')
plt.plot(np.linspace(intervals[0][0], intervals[0][1], 1000), hp.triple_gaussian(np.linspace(intervals[0][0], intervals[0][1], 1000), *lyso1_popt), label="LYSO Fit", color='dodgerblue', ls='--')
plt.plot(np.linspace(intervals[1][0], intervals[1][1], 1000), hp.gauss_function(np.linspace(intervals[1][0], intervals[1][1], 1000), *na1_popt), label="Sodium Fit 1", color='forestgreen', ls='--')
plt.plot(np.linspace(intervals[2][0], intervals[2][1], 1000), hp.gauss_function(np.linspace(intervals[2][0], intervals[2][1], 1000), *na2_popt), label="Sodium Fit 2", color='forestgreen', ls='--')
plt.plot(np.linspace(intervals[3][0], intervals[3][1], 1000), hp.gauss_function(np.linspace(intervals[3][0], intervals[3][1], 1000), *cs1_popt), label="Cesium Fit", color='firebrick', ls='--')
plt.plot(np.linspace(intervals[4][0], intervals[4][1], 1000), hp.double_gaussian(np.linspace(intervals[4][0], intervals[4][1], 1000), *co1_popt), label="Cobalt Fit", color='indigo', ls='--')
plt.xlabel("Energy (KeV)")
plt.ylabel("Counts")
plt.title("Converted Spectrums")
plt.legend()

# Filling all the points
peaks_positions_KeV = [lyso1_popt[1], lyso1_popt[4], lyso1_popt[7], na1_popt[1], cs1_popt[1], co1_popt[1], co1_popt[4], na2_popt[1]]
peaks_sigma_KeV = [lyso1_popt[2], lyso1_popt[5], lyso1_popt[8], na1_popt[2], cs1_popt[2], co1_popt[2], co1_popt[5], na2_popt[2]]
peaks_position_err_KeV = [lyso1_pcov[1], lyso1_pcov[4], lyso1_pcov[7], na1_pcov[1], cs1_pcov[1], co1_pcov[1], co1_pcov[4], na2_pcov[1]]
sigma_err_KeV = [lyso1_pcov[2], lyso1_pcov[5], lyso1_pcov[8], na1_pcov[2], cs1_pcov[2], co1_pcov[2], co1_pcov[5], na2_pcov[2]]
resolution_in_KeV = 2*np.array(peaks_sigma_KeV) / np.array(peaks_positions_KeV)
resolution_in_KeV_err = resolution_in_KeV * np.sqrt((np.array(peaks_position_err_KeV) / np.array(peaks_positions_KeV))**2 + (np.array(sigma_err_KeV) / np.array(peaks_sigma_KeV))**2)

# Plotting the resolution in KeV marking the third peak of lyso
# Fitting also here 1/sqrt(E) function to the resolution in KeV removing the lyso third peak
mask_3 = [True, True, False, True, True, True, True, True]  # Mask to exclude the third peak of LYSO
popt_res_KeV, pcov_res_KeV = hp.curve_fit(inverse_sqrt_function, np.array(peaks_positions_KeV)[mask_3], 100*np.array(resolution_in_KeV)[mask_3], sigma=100*np.array(resolution_in_KeV_err)[mask_3], p0 = [320, 0, 30])

plt.figure(figsize=(8, 6))
plt.errorbar(np.array(peaks_positions_KeV)[mask_3], 100*np.array(resolution_in_KeV)[mask_3], yerr=100*np.array(resolution_in_KeV_err)[mask_3], fmt='o', label='Resolution')
plt.errorbar(peaks_positions_KeV[2], 100*resolution_in_KeV[2], yerr=100*resolution_in_KeV_err[2], fmt='o', label='LYSO 3rd Peak', color='red')
plt.plot(np.linspace(0, 4000, 10000), inverse_sqrt_function(np.linspace(0, 4000, 10000), *popt_res_KeV), label='Fit (excluding LYSO 3rd peak)', color='darkorange')
plt.xlabel('Peak Position (keV)')
plt.ylabel('Resolution (2σ/E)[%]')
plt.title('Resolution of Radioactive Sources')
plt.legend()

print(" ")
print("Resolution in KeV:")
print("Energy - Resolution")

for i, res in enumerate(100*resolution_in_KeV):
    print(f" {peaks_positions_KeV[i]:.2f} {100*resolution_in_KeV[i]:.2f} ± {100*resolution_in_KeV_err[i]:.2f} %")

# Print the position in KeV of each fitted peak (mean +- err_on_mean, sigma +- err_on_sigma)
print(" ")
print("Fitted peak positions (in ADC) +- err_on_mean, sigma +- err_on_sigma:")
print(f"LYSO1: {lyso1_popt[1]:.2f} +- {lyso1_pcov[1]:.2f}, {lyso1_popt[2]:.2f} +- {lyso1_pcov[2]:.2f}")
print(f"LYSO2: {lyso1_popt[4]:.2f} +- {lyso1_pcov[4]:.2f}, {lyso1_popt[5]:.2f} +- {lyso1_pcov[5]:.2f}")
print(f"LYSO3: {lyso1_popt[7]:.2f} +- {lyso1_pcov[7]:.2f}, {lyso1_popt[8]:.2f} +- {lyso1_pcov[8]:.2f}")
print(f"Sodium1: {na1_popt[1]:.2f} +- {na1_pcov[1]:.2f}, {na1_popt[2]:.2f} +- {na1_pcov[2]:.2f}")
print(f"Sodium2: {na2_popt[1]:.2f} +- {na2_pcov[1]:.2f}, {na2_popt[2]:.2f} +- {na2_pcov[2]:.2f}")
print(f"Cesium1: {cs1_popt[1]:.2f} +- {cs1_pcov[1]:.2f}, {cs1_popt[2]:.2f} +- {cs1_pcov[2]:.2f}")
print(f"Cobalt1: {co1_popt[1]:.2f} +- {co1_pcov[1]:.2f}, {co1_popt[2]:.2f} +- {co1_pcov[2]:.2f}")
print(f"Cobalt2: {co1_popt[4]:.2f} +- {co1_pcov[4]:.2f}, {co1_popt[5]:.2f} +- {co1_pcov[5]:.2f}")
print(" ")
print("MEAN - ERROR ON MEAN - SIGMA - ERROR ON SIGMA")
print(f"{lyso1_popt[1]:.2f} {lyso1_pcov[1]:.2f} {lyso1_popt[2]:.2f} {lyso1_pcov[2]:.2f}")
print(f"{lyso1_popt[4]:.2f} {lyso1_pcov[4]:.2f} {lyso1_popt[5]:.2f} {lyso1_pcov[5]:.2f}")
print(f"{lyso1_popt[7]:.2f} {lyso1_pcov[7]:.2f} {lyso1_popt[8]:.2f} {lyso1_pcov[8]:.2f}")
print(f"{na1_popt[1]:.2f} {na1_pcov[1]:.2f} {na1_popt[2]:.2f} {na1_pcov[2]:.2f}")
print(f"{na2_popt[1]:.2f} {na2_pcov[1]:.2f} {na2_popt[2]:.2f} {na2_pcov[2]:.2f}")
print(f"{cs1_popt[1]:.2f} {cs1_pcov[1]:.2f} {cs1_popt[2]:.2f} {cs1_pcov[2]:.2f}")
print(f"{co1_popt[1]:.2f} {co1_pcov[1]:.2f} {co1_popt[2]:.2f} {co1_pcov[2]:.2f}")
print(f"{co1_popt[4]:.2f} {co1_pcov[4]:.2f} {co1_popt[5]:.2f} {co1_pcov[5]:.2f}")

plt.figure(figsize=(8, 6))
plt.errorbar(np.array(peaks_positions_KeV)[mask_3], 50*np.array(resolution_in_KeV)[mask_3], yerr=50*np.array(resolution_in_KeV_err)[mask_3], fmt='o', label='Resolution')
plt.errorbar(peaks_positions_KeV[2], 50*resolution_in_KeV[2], yerr=50*resolution_in_KeV_err[2], fmt='o', label='LYSO 3rd Peak', color='red')
plt.plot(np.linspace(0, 4000, 10000), inverse_sqrt_function(np.linspace(0, 4000, 10000), *popt_res_KeV)/2, label='Fit (excluding LYSO 3rd peak)', color='darkorange')
plt.xlabel('Peak Position (keV)')
plt.ylabel('Resolution/2 (σ/E)[%]')
plt.title('Resolution of Radioactive Sources')
plt.legend()

print(" ")
print("Resolution in KeV:")
print("Energy - Resolution")

plt.show()