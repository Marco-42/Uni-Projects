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

voltage = [40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0]  
distances = [80, 100, 120, 170, 200, 220, 270]
bin_size = [1, 2, 2, 3, 3, 3, 3]
fitting_points = [20, 10, 10, 10, 10, 10, 10]

number_of_peaks = 8

CODE = "5_3_2_2_2" 
data_directory_name = "PST_" + CODE + "_HV_"
path_name = "./data/"

for v in voltage:

    # Gaussian fit parameters
    chi_square = []
    mean_values = []
    sigma_values = []
    A_values = []

    SiPM_ID = hp.SiPM(CODE)
    # Extracting the column number
    # DAQ - gain type - CITORC - Channel number
    column_number = hp.get_column_number(1, "HG", "C", 4)
    
    # Extracting the data from the CSV file
    ADC, counts = hp.extract_SiPM_data(path_name + data_directory_name + str(v) + "V/" + "data.csv", column_number, False)

    # Rebinning the data with a bin size
    ADC, counts = hp.rebin_xy(ADC, counts, bin_size[voltage.index(v)], x_mode="mean", y_mode="sum")

    # Finding peaks in the data
    rebinned_distance = max(1, int(distances[voltage.index(v)] / bin_size[voltage.index(v)]))
    peaks_indices, peaks_values = hp.find_peaks(
        counts,
        use_scipy=True,
        prominence=40,
        distance=rebinned_distance,
    )

    # Fitting a Gaussian function to each peaks taking points within a range of [-20, +20]
    for i in range(len(peaks_indices)):
        
        # Taking points for fit
        x_peak = ADC[peaks_indices[i]-fitting_points[voltage.index(v)]:peaks_indices[i]+fitting_points[voltage.index(v)]]
        y_peak = counts[peaks_indices[i]-fitting_points[voltage.index(v)]:peaks_indices[i]+fitting_points[voltage.index(v)]]
        sy_peak = np.sqrt(y_peak)  # Assuming Poisson statistics for the uncertainties
        popt, perr = hp.gauss_fit(x_peak, y_peak, sy_peak, p0 = [peaks_values[i], ADC[peaks_indices[i]], 10])
        A, mu, sigma = popt
        A_err, mu_err, sigma_err = perr
        mean_values.append([mu, mu_err])
        sigma_values.append([sigma, sigma_err])
        A_values.append([A, A_err])
        
        # Calculating chi-square for the fit
        chi_square.append(np.sum((y_peak - hp.gauss_function(x_peak, *popt)) ** 2 / (sy_peak ** 2)))

    # Plotting the finger plot
    plt.figure()
    plt.plot(ADC, counts, label = "Data", color = "black")
    plt.plot(ADC[peaks_indices], peaks_values, "x", label="Peaks")

    # Plotting the Gaussian fit for each peak skipping the pedastal(first peak) and plotting the first number_of_peaks peaks after it
    for i in range(1, min(number_of_peaks+1, len(peaks_indices))):
        x_fit = ADC[peaks_indices[i]-fitting_points[voltage.index(v)]:peaks_indices[i]+fitting_points[voltage.index(v)]]
        y_fit = hp.gauss_function(x_fit, A_values[i][0], mean_values[i][0], sigma_values[i][0])
        plt.plot(x_fit, y_fit, label=f"$\mu$: {mean_values[i][0]:.0f} ± {mean_values[i][1]:.0f}", lw = 2)
    plt.xlabel("ADC")
    plt.ylabel("Counts")
    plt.title("Finger Plot for " + str(v) + " V")
    plt.legend()
    # Use ADC values at peak indices to set x range (indices -> ADC units)
    if len(peaks_indices) >= 2:
        left_val = ADC[peaks_indices[0]] - 100
        right_val = ADC[peaks_indices[-1]] + 100
        plt.xlim(left_val, right_val)
    else:
        plt.xlim(1500, 5000)
    plt.savefig(path_name + data_directory_name + str(v) + "V/" + "finger_plot.png")

    # Saving the fit parameters in a txt file (ensure results dir exists; recreate file)
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{CODE}_{v}.txt")
    if os.path.exists(result_file):
        os.remove(result_file)
    with open(result_file, "w") as f:
        for i in range(1, min(number_of_peaks+1, len(peaks_indices))):
            f.write(f"{mean_values[i][0]} {mean_values[i][1]}\n")

plt.show()