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
          'figure.figsize': (8, 6),
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

distance_vs_voltage = []
distance_between_peaks = []

distance_first_peak_and_pedestal = []

sigma_vs_voltage = []

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
    plt.figure(figsize=(8, 6))
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
    plt.tight_layout()
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

    # Finding distance between peaks and its uncertainty using error propagation and saving it in a list for each voltage
    for i in range(2, min(number_of_peaks+1, len(peaks_indices))):
        distance_between_peaks.append([mean_values[i][0] - mean_values[i-1][0], np.sqrt(mean_values[i][1]**2 + mean_values[i-1][1]**2)])

    # Finding distance between the first peak and the pedestal and its uncertainty using error propagation and saving it in a list for each voltage
    distance_first_peak_and_pedestal.append([mean_values[1][0] - mean_values[0][0], np.sqrt(mean_values[1][1]**2 + mean_values[0][1]**2)])

    # Finding the mean distance for the fixed voltage
    distance_vs_voltage.append([statistics.mean([d[0] for d in distance_between_peaks]), np.sqrt(statistics.mean([d[1]**2 for d in distance_between_peaks]))/len(distance_between_peaks)])

    # Plotting distance between peaks for the fixed voltage
    plt.figure(figsize=(8, 6))
    plt.errorbar(np.arange(len(distance_between_peaks)), [d[0] for d in distance_between_peaks], yerr=[d[1] for d in distance_between_peaks], fmt="o", color = 'darkblue')
    plt.axhline(y=statistics.mean([d[0] for d in distance_between_peaks]), color='darkorange', linestyle='--', label=f"Mean: {statistics.mean([d[0] for d in distance_between_peaks]):.2f} ± {np.sqrt(statistics.mean([d[1]**2 for d in distance_between_peaks]))/len(distance_between_peaks):.2f}")
    plt.fill_between(np.arange(len(distance_between_peaks)), statistics.mean([d[0] for d in distance_between_peaks]) - np.sqrt(statistics.mean([d[1]**2 for d in distance_between_peaks]))/len(distance_between_peaks), statistics.mean([d[0] for d in distance_between_peaks]) + np.sqrt(statistics.mean([d[1]**2 for d in distance_between_peaks]))/len(distance_between_peaks), alpha=0.5, color='gold')
    plt.xlabel("Peak Number")
    plt.ylabel("Distance between Peaks (ADC)")
    plt.title("Distance between peaks for " + str(v) + " V")
    plt.tight_layout()
    plt.savefig(path_name + data_directory_name + str(v) + "V/" + "distance_between_peaks.png")

    # Plotting the same thing but for gaussian sigma
    plt.figure(figsize=(8, 6))
    plt.errorbar(np.arange(len(sigma_values[1:number_of_peaks + 1][:])), [s[0] for s in sigma_values[1:number_of_peaks + 1][:]], yerr=[s[1] for s in sigma_values[1:number_of_peaks + 1][:]], fmt="o", color = 'darkred')
    plt.axhline(y=statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]), color='dodgerblue', linestyle='--', label=f"Mean: {statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]):.2f} ± {np.sqrt(statistics.mean([s[1]**2 for s in sigma_values[1:number_of_peaks + 1][:]]))/len(sigma_values[1:number_of_peaks + 1][:]):.2f}")
    plt.fill_between(np.arange(len(sigma_values[1:number_of_peaks + 1][:])), statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]) - np.sqrt(statistics.mean([s[1]**2 for s in sigma_values[1:number_of_peaks + 1][:]]))/len(sigma_values[1:number_of_peaks + 1][:]), statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]) + np.sqrt(statistics.mean([s[1]**2 for s in sigma_values[1:number_of_peaks + 1][:]]))/len(sigma_values[1:number_of_peaks + 1][:]), alpha=0.5, color='dodgerblue')
    plt.xlabel("Peak Number")
    plt.ylabel("Sigma of Gaussian Fit (ADC)")
    plt.title("Sigma of Gaussian Fit for " + str(v) + " V")
    plt.tight_layout()
    plt.savefig(path_name + data_directory_name + str(v) + "V/" + "sigma_values.png")

    # Getting the mean sigma value for the fixed voltage and saving it in a list for each voltage
    sigma_vs_voltage.append([statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]), np.sqrt(statistics.mean([s[1]**2 for s in sigma_values[1:number_of_peaks + 1][:]]))/len(sigma_values[1:number_of_peaks + 1][:])])

    distance_between_peaks = []  # Clear the list for the next voltage

# Fitting distance between peaks vs voltage with a linear function and plotting it
voltage_array = np.array(voltage)
distance_array = np.array([d[0] for d in distance_vs_voltage])
distance_error_array = np.array([d[1] for d in distance_vs_voltage])

popt, perr = hp.linear_fit(voltage_array, distance_array, distance_error_array)
a, b = popt
a_err, b_err = perr

# Plotting distance between peaks vs voltage
plt.figure(figsize=(8, 6))
plt.errorbar(voltage, [d[0] for d in distance_vs_voltage], yerr=[d[1] for d in distance_vs_voltage], fmt="o")
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt), label=f"Fit: y = {a:.2f} ± {a_err:.2f} * x + {b:.2f} ± {b_err:.2f}", color='red')
plt.xlabel("Voltage (V)")
plt.ylabel("Mean Distance between Peaks (ADC)")
plt.title("Mean Distance between Peaks vs Voltage")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "distance_vs_voltage.png"))

# Lineare fit of the distance between the first peak and the pedestal vs voltage
voltage_array = np.array(voltage)
distance_first_peak_and_pedestal_array = np.array([d[0] for d in distance_first_peak_and_pedestal])
distance_first_peak_and_pedestal_error_array = np.array([d[1] for d in distance_first_peak_and_pedestal])
popt, perr = hp.linear_fit(voltage_array, distance_first_peak_and_pedestal_array, distance_first_peak_and_pedestal_error_array)
a, b = popt
a_err, b_err = perr

# Plotting distance between the first peak and the pedestal for the fixed voltage
plt.figure(figsize=(8, 6))
plt.errorbar(voltage, [d[0] for d in distance_first_peak_and_pedestal], yerr=[d[1] for d in distance_first_peak_and_pedestal], fmt="o", color = 'darkblue')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt), label=f"Fit: y = {a:.2f} ± {a_err:.2f} * x + {b:.2f} ± {b_err:.2f}", color='red')
plt.xlabel("Voltage (V)")
plt.ylabel("Distance between First Peak and Pedestal (ADC)")
plt.title("Distance between First Peak and Pedestal vs Voltage")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "distance_first_peak_and_pedestal.png"))

# Plotting sigma of Gaussian fit vs voltage
voltage_array = np.array(voltage)
sigma_array = np.array([s[0] for s in sigma_vs_voltage])
sigma_error_array = np.array([s[1] for s in sigma_vs_voltage])
popt, perr = hp.linear_fit(voltage_array, sigma_array, sigma_error_array)
a, b = popt
a_err, b_err = perr

# Plotting sigma of Gaussian fit vs voltage
plt.figure(figsize=(8, 6))
plt.errorbar(voltage, [s[0] for s in sigma_vs_voltage], yerr=[s[1] for s in sigma_vs_voltage], fmt="o", color = 'darkred')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt), label=f"Fit: y = {a:.2f} ± {a_err:.2f} * x + {b:.2f} ± {b_err:.2f}", color='blue')
plt.xlabel("Voltage (V)")
plt.ylabel("Mean Sigma of Gaussian Fit (ADC)")
plt.title("Mean Sigma of Gaussian Fit vs Voltage")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "sigma_vs_voltage.png"))
plt.show()
