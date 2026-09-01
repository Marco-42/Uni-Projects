import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from SiPM_helper import (
    beta_function,
    gauss_fit,
    gauss_function,
    get_column_number,
    extract_SiPM_data,
    rebin_xy,
    beta_function,
    triple_gaussian_beta,
    triple_gaussian_beta_fit,
    beta2_sig,
    beta2_sig_fit,
    sigmoid
)

# ============================================================
# Triple Gaussian model
# ============================================================

# ============================================================
# Data loading
# ============================================================

# Path to the CSV file containing the acquisition data
file_path = "./data/40.5/gain3/data.csv"

# Compute the column number corresponding to:
# HG gain, DAQ2, CITIROC C, channel 26
column = get_column_number(
    DAQ=2,
    gain_type="HG",
    CITIROC="C",
    channel=27
)

# Extract the selected column
x, y = extract_SiPM_data(
    file_path,
    column,
    one_based_column= False
)

x, y = rebin_xy(x, y, 5, "mean", "sum")

# Creating a txt file and saving data on it
with open("fitted_data.txt", "w") as f:
    for i in range(len(x)):
        f.write(f"{x[i]} {y[i]} {np.sqrt(y[i])}\n")

# Taking only the points between a certain range
mask1 = (x > 2160) & (x < 2320)
mask2 = (x > 2610) & (x < 2980)
mask3 = (x > 3380) & (x < 3840)

gaus1_x = x[mask1]
gaus1_y = y[mask1]
gaus2_x = x[mask2]
gaus2_y = y[mask2]
gaus3_x = x[mask3]
gaus3_y = y[mask3]

# ============================================================
# Histogram creation
# ============================================================

# Poisson uncertainties
errors1 = np.sqrt(gaus1_y)
errors2 = np.sqrt(gaus2_y)
errors3 = np.sqrt(gaus3_y)

# Fitting a guassian distribution for each dataset
popt1, pcov1 = gauss_fit(gaus1_x, gaus1_y, errors1)#, p0 = [50.71, 2323.0, 54.58])

popt2, pcov2 = gauss_fit(gaus2_x, gaus2_y, errors2)#, p0 = [259.45, 2967.64, 207.68])

popt3, pcov3 = gauss_fit(gaus3_x, gaus3_y, errors3)#, p0 = [540.73, 3843.00, 369.57])

def gauss_with_condition(x, A, mu, sigma):
    return np.where(x < mu, gauss_function(x, A, mu, sigma), 0)
    
# # Removing the gaussian from the original data taking only the points of gautian before the mean
y_beta = y - gauss_with_condition(x, *popt1) - gauss_with_condition(x, *popt2) - gauss_with_condition(x, *popt3)
YBETA2 = y - gauss_function(x, *popt1) - gauss_function(x, *popt2) - gauss_function(x, *popt3)

# Plotting all
plt.figure(figsize=(10, 6))
plt.errorbar(x, y, yerr=np.sqrt(y), fmt='o', label='Data', markersize=3, color='black')
plt.plot(x, gauss_function(x, *popt1), label='Gaussian 1', color='red')
plt.plot(x, gauss_function(x, *popt2), label='Gaussian 2', color='green')
plt.plot(x, gauss_function(x, *popt3), label='Gaussian 3', color='blue')
# plt.plot(x, y_beta, label='Data - Gaussians', color='orange')

plt.figure(figsize=(10, 6))
plt.errorbar(x, y_beta, yerr=np.sqrt(y), fmt='o', label='Data - Gaussians', markersize=3, color='orange')

plt.show()
