import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from SiPM_helper import (
    get_column_number,
    extract_SiPM_data,
    gauss_fit,
    gauss_function,
    rebin_xy
)

# ============================================================
# Triple Gaussian model
# ============================================================

def triple_gaussian(
    x,
    A1, mu1, sigma1,
    A2, mu2, sigma2,
    A3, mu3, sigma3
):
    """
    Sum of three Gaussian functions.
    """
    
    # Defining each gaussian
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    g3 = A3 * np.exp(-(x - mu3)**2 / (2 * sigma3**2))

    return g1 + g2 + g3

def single_gaussian(x, A1, mu1, sigma1):
    
    return A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
# ============================================================
# Data loading
# ============================================================

# Path to the CSV file containing the acquisition data
file_path = "./data/40.0/gain3333/data.csv"

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

x, y, yerr = rebin_xy(x, y, 8, "mean", "sum", np.sqrt(y))

# Taking only the points between a certain range
mask = (x > 2057 + 350) & (x < 2057 + 2000)
x_masked = x[mask]
y_masked = y[mask]
errors = np.array(yerr[mask])

# ============================================================
# Histogram creation
# ============================================================

# ============================================================
# Initial parameter estimates
# ============================================================

# Locate the highest peak in the histogram
peak_max = max(y_masked)

# Initial parameter guesses:
# [A1, mu1, sigma1,
#  A2, mu2, sigma2,
#  A3, mu3, sigma3]
p0 = [
    0.8*peak_max,
    2057 + 520,
    100,

    peak_max,
    2057 + 780,
    120,

    peak_max * 0.1, 
    2057 + 1280,
    100
]

# ============================================================
# Fit execution
# ============================================================

popt, pcov = curve_fit(
    triple_gaussian,
    x_masked,
    y_masked,
    sigma=errors,
    absolute_sigma=True,
    p0=p0,
    maxfev=50000
)

# Parameter uncertainties
perr = np.sqrt(np.diag(pcov))

# ============================================================
# Plot results
# ============================================================

# Generate a smooth x-axis for visualization
x_fit = np.linspace(
    2395,
    3810,
    5000
)

# Evaluate the fitted model
y_fit = triple_gaussian(x_fit, *popt)

plt.figure(figsize=(8, 5))


# Plot histogram points with uncertainties
plt.errorbar(
    x,
    y,
    yerr=yerr,
    fmt='.',
    color='black',
    label='Data',
    zorder = 0
)

# Plot fitted function
plt.plot(
    x_fit,
    y_fit,
    color='red',
    label='Triple Gaussian Fit',
    lw = 2,
    zorder = 1
)

colors = ['darkorange', 'gold', 'purple']
# Plotting each gaussian
for i in range(3):

    A = popt[3 * i]
    mu = popt[3 * i + 1]
    sigma = popt[3 * i + 2]

    dA = perr[3 * i]
    dmu = perr[3 * i + 1]
    dsigma = perr[3 * i + 2]

    print(f"\nGaussian {i + 1}")
    print(f"Amplitude = {A:.2f} ± {dA:.2f}")
    print(f"Mean      = {mu:.2f} ± {dmu:.2f}")
    print(f"Sigma     = {sigma:.2f} ± {dsigma:.2f}")

    plt.plot(x_fit, single_gaussian(x_fit, A, mu, sigma), label = f'Gaussian {i + 1} A = {A:.0f} ± {dA:.0f} $\mu$ = {mu:.0f} ± {dmu:.0f} $\sigma$ = {sigma:.0f} ± {dsigma:.0f}', ls = '--', color = colors[i], lw = 1, zorder = 1)
plt.xlabel("ADC Channel")
plt.ylabel("Counts")
plt.title("Energy Spectrum Fit")
plt.legend()

plt.tight_layout()
plt.show()