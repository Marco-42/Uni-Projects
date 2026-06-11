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
bin_size = [1, 2, 2, 3, 3, 6, 6]
fitting_points = [20, 10, 10, 10, 10, 6, 6]
langau_widths = [400, 400, 700, 1000, 1000, 1200, 1400]
number_of_peaks = 8

CODE = "5_3_2_1_1" 
data_directory_name = "PST_" + CODE + "_HV_"
path_name = "./data/"

distance_vs_voltage = []
distance_between_peaks = []

distance_first_peak_and_pedestal = []

sigma_vs_voltage = []

first_peak_position = []
pedestal_position = []
second_peak_position = []
third_peak_position = []
fourth_peak_position = []

for v in voltage:

    # Gaussian fit parameters
    chi_square = []
    mean_values = []
    sigma_values = []
    A_values = []

    SiPM_ID = hp.SiPM(CODE)
    # Extracting the column number
    # DAQ - gain type - CITORC - Channel number
    column_number = hp.get_column_number(1, "HG", "C", 5)
    
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

    # Plotting the finger plot
    plt.figure(figsize=(8, 6))
    plt.plot(ADC, counts, label = "Data", color = "black")
    plt.plot(ADC[peaks_indices], peaks_values, "x", label="Peaks")
    
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
    plt.savefig(path_name + data_directory_name + str(v) + "V/" + "finger_plot_not_fit.png")

    # Finding distance between peaks and its uncertainty using error propagation and saving it in a list for each voltage
    for i in range(2, min(number_of_peaks+1, len(peaks_indices))):
        distance_between_peaks.append([mean_values[i][0] - mean_values[i-1][0], np.sqrt(mean_values[i][1]**2 + mean_values[i-1][1]**2)])

    # Finding distance between the first peak and the pedestal and its uncertainty using error propagation and saving it in a list for each voltage
    distance_first_peak_and_pedestal.append([mean_values[1][0] - mean_values[0][0], np.sqrt(mean_values[1][1]**2 + mean_values[0][1]**2)])

    # Finding the mean distance for the fixed voltage
    distance_vs_voltage.append([statistics.mean([d[0] for d in distance_between_peaks]), np.sqrt(statistics.mean([d[1]**2 for d in distance_between_peaks])/len(distance_between_peaks))])

    # Plotting distance between peaks for the fixed voltage
    plt.figure(figsize=(8, 6))
    plt.errorbar(np.arange(len(distance_between_peaks)), [d[0] for d in distance_between_peaks], yerr=[d[1] for d in distance_between_peaks], fmt="o", color = 'darkblue')
    plt.axhline(y=statistics.mean([d[0] for d in distance_between_peaks]), color='darkorange', linestyle='--', label=f"Mean: {statistics.mean([d[0] for d in distance_between_peaks]):.2f} ± {np.sqrt(statistics.mean([d[1]**2 for d in distance_between_peaks])/len(distance_between_peaks)):.2f}")
    plt.fill_between(np.arange(len(distance_between_peaks)), statistics.mean([d[0] for d in distance_between_peaks]) - np.sqrt(statistics.mean([d[1]**2 for d in distance_between_peaks])/len(distance_between_peaks)), statistics.mean([d[0] for d in distance_between_peaks]) + np.sqrt(statistics.mean([d[1]**2 for d in distance_between_peaks])/len(distance_between_peaks)), alpha=0.5, color='gold')
    plt.xlabel("Peak Number")
    plt.ylabel("Distance between Peaks (ADC)")
    plt.title("Distance between peaks for " + str(v) + " V")
    plt.tight_layout()
    plt.savefig(path_name + data_directory_name + str(v) + "V/" + "distance_between_peaks.png")

    # Plotting the same thing but for gaussian sigma
    plt.figure(figsize=(8, 6))
    plt.errorbar(np.arange(len(sigma_values[1:number_of_peaks + 1][:])), [s[0] for s in sigma_values[1:number_of_peaks + 1][:]], yerr=[s[1] for s in sigma_values[1:number_of_peaks + 1][:]], fmt="o", color = 'darkred')
    plt.axhline(y=statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]), color='dodgerblue', linestyle='--', label=f"Mean: {statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]):.2f} ± {np.sqrt(statistics.mean([s[1]**2 for s in sigma_values[1:number_of_peaks + 1][:]])/len(sigma_values[1:number_of_peaks + 1][:])):.2f}")
    plt.fill_between(np.arange(len(sigma_values[1:number_of_peaks + 1][:])), statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]) - np.sqrt(statistics.mean([s[1]**2 for s in sigma_values[1:number_of_peaks + 1][:]])/len(sigma_values[1:number_of_peaks + 1][:])), statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]) + np.sqrt(statistics.mean([s[1]**2 for s in sigma_values[1:number_of_peaks + 1][:]])/len(sigma_values[1:number_of_peaks + 1][:])), alpha=0.5, color='dodgerblue')
    plt.xlabel("Peak Number")
    plt.ylabel("Sigma of Gaussian Fit (ADC)")
    plt.title("Sigma of Gaussian Fit for " + str(v) + " V")
    plt.tight_layout()
    plt.savefig(path_name + data_directory_name + str(v) + "V/" + "sigma_values.png")

    # Getting the mean sigma value for the fixed voltage and saving it in a list for each voltage
    sigma_vs_voltage.append([statistics.mean([s[0] for s in sigma_values[1:number_of_peaks + 1][:]]), np.sqrt(statistics.mean([s[1]**2 for s in sigma_values[1:number_of_peaks + 1][:]])/len(sigma_values[1:number_of_peaks + 1][:]))])

    distance_between_peaks = []  # Clear the list for the next voltage

    # Getting the position of the first peak and the pedestal
    first_peak_position.append([mean_values[1][0], mean_values[1][1]])
    pedestal_position.append([mean_values[0][0], mean_values[0][1]])
    second_peak_position.append([mean_values[2][0], mean_values[2][1]])
    third_peak_position.append([mean_values[3][0], mean_values[3][1]])
    fourth_peak_position.append([mean_values[4][0], mean_values[4][1]])

    bg_idx, bg_v = hp.extract_bg(ADC, counts, mean_values[0][0], mean_values[-1][0], True, prominence=5, distance=rebinned_distance)
    
    # Taking the bg mean value aroung the founded bg points(take 10 point on left and on right)
    bg_values = []
    bg_error = []

    for idx in bg_idx:
        bg_values.append(statistics.mean(counts[max(0, idx-1):min(len(counts), idx+1)]))
        bg_error.append(np.sqrt(statistics.mean([s for s in counts[max(0, idx-1):min(len(counts), idx+1)]]) / 2))
    
    bg_popt, bg_perr = hp.bg_fitting(ADC[bg_idx], bg_values, bg_error, max(bg_values), langau_widths[voltage.index(v)], 100, 7000)
    
    finge = hp.FingerPlotPoint(mean_values[:][0], mean_values[:][1], bg_idx, bg_values, bg_error, v)

    # Plotting all the information(peaks, fingerplot, gauss fit of peaks and bg points)
    plt.figure(figsize=(8, 6))
    plt.plot(ADC, counts, label = "Data", color = "black")
    #plt.plot(ADC[peaks_indices], peaks_values, "x", label="Peaks", color = 'dodgerblue')
    for i in range(2, min(number_of_peaks+1, len(peaks_indices))):
        x_fit = ADC[peaks_indices[i]-fitting_points[voltage.index(v)]:peaks_indices[i]+fitting_points[voltage.index(v)]]
        y_fit = hp.gauss_function(x_fit, A_values[i][0], mean_values[i][0], sigma_values[i][0])
        plt.plot(x_fit, y_fit, color = 'dodgerblue', lw = 1.5)
    x_fit = ADC[peaks_indices[1]-fitting_points[voltage.index(v)]:peaks_indices[1]+fitting_points[voltage.index(v)]]
    y_fit = hp.gauss_function(x_fit, A_values[1][0], mean_values[1][0], sigma_values[1][0])
    plt.plot(x_fit, y_fit, color = 'dodgerblue', lw = 1.5, label = "Peaks gaussian fit")
    plt.errorbar(ADC[bg_idx], bg_values, yerr=bg_error, fmt="o", label="BG Points", color = 'darkorange')
    plt.plot(np.linspace(0, max(ADC), 10000), hp.background_model(np.linspace(0, max(ADC), 10000), *bg_popt), label=f"BG Fit: Landau + Gaussian", color='darkred', lw = 2)
    plt.fill_between(np.linspace(min(ADC), max(ADC), 1000), 0, hp.background_model(np.linspace(min(ADC), max(ADC), 1000), *bg_popt), alpha=0.5, color='orange')
    plt.xlabel("ADC")
    plt.ylabel("Counts")
    plt.title("Finger Plot with BG for " + str(v) + " V")
    plt.legend()
    if len(peaks_indices) >= 2:
        left_val = ADC[peaks_indices[0]] - 100
        right_val = ADC[peaks_indices[-1]] + 100
        plt.xlim(left_val, right_val)
    else:
        plt.xlim(1500, 5000)
    plt.tight_layout()
    plt.savefig(path_name + data_directory_name + str(v) + "V/" + "finger_plot_with_bg.png")

    # Getting the integration of each peaks(not the pedestal)
    integrated_peaks = []
    # Saving the fit parameters in a txt file (ensure results dir exists; recreate file)
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{CODE}_{v}.txt")
    if os.path.exists(result_file):
        os.remove(result_file)

    print(" ")
    print("VOLTAGE: ", v)
    for j in range(len(bg_idx)-1):
        integrated_peaks.append(hp.integration_without_bg(ADC, counts, hp.background_model(ADC, *bg_popt), ADC[bg_idx[j]], ADC[bg_idx[j + 1]]))
        print(integrated_peaks[-1])
        with open(result_file, "a") as f:
            f.write(f"{integrated_peaks[-1]}\n")




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

# Linear fitting the position of the first peak and second one
voltage_array = np.array(voltage)
first_peak_position_array = np.array([p[0] for p in first_peak_position])
first_peak_position_error_array = np.array([p[1] for p in first_peak_position])
second_peak_position_array = np.array([p[0] for p in second_peak_position])
second_peak_position_error_array = np.array([p[1] for p in second_peak_position])
popt, perr = hp.linear_fit(voltage_array, first_peak_position_array, first_peak_position_error_array)
a, b = popt
a_err, b_err = perr
popt_second, perr_second = hp.linear_fit(voltage_array, second_peak_position_array, second_peak_position_error_array)
a_second, b_second = popt_second
a_second_err, b_second_err = perr_second

# Fitting the third and fourth peaks
third_peak_position_array = np.array([p[0] for p in third_peak_position])
third_peak_position_error_array = np.array([p[1] for p in third_peak_position])
popt_third, perr_third = hp.linear_fit(voltage_array, third_peak_position_array, third_peak_position_error_array)
a_third, b_third = popt_third
a_third_err, b_third_err = perr_third

fourth_peak_position_array = np.array([p[0] for p in fourth_peak_position])
fourth_peak_position_error_array = np.array([p[1] for p in fourth_peak_position])
popt_fourth, perr_fourth = hp.linear_fit(voltage_array, fourth_peak_position_array, fourth_peak_position_error_array)
a_fourth, b_fourth = popt_fourth
a_fourth_err, b_fourth_err = perr_fourth

# Linear fit of pedestal
voltage_array = np.array(voltage)
pedestal_position_array = np.array([p[0] for p in pedestal_position])
pedestal_position_error_array = np.array([p[1] for p in pedestal_position])
popt_pedestal, perr_pedestal = hp.linear_fit(voltage_array, pedestal_position_array, pedestal_position_error_array)
a_pedestal, b_pedestal = popt_pedestal
a_pedestal_err, b_pedestal_err = perr_pedestal

# Plotting the position of the first peak and the pedestal vs voltage
plt.figure(figsize=(8, 6))
plt.errorbar(voltage, [p[0] for p in first_peak_position], yerr=[p[1] for p in first_peak_position], fmt="o", label="First Peak", color='green')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt), label=f"Fit: y = {a:.2f} ± {a_err:.2f} * x + {b:.2f} ± {b_err:.2f}", color='blue')
plt.errorbar(voltage, [p[0] for p in pedestal_position], yerr=[p[1] for p in pedestal_position], fmt="o", label="Pedestal", color='orange')
plt.errorbar(voltage, [p[0] for p in second_peak_position], yerr=[p[1] for p in second_peak_position], fmt="o", label="Second Peak", color='red')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt_second), label=f"Fit: y = {a_second:.2f} ± {a_second_err:.2f} * x + {b_second:.2f} ± {b_second_err:.2f}", color='magenta')
plt.errorbar(voltage, [p[0] for p in pedestal_position], yerr=[p[1] for p in pedestal_position], fmt="o", label="Pedestal", color='orange')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt_pedestal), label=f"Fit: y = {a_pedestal:.2f} ± {a_pedestal_err:.2f} * x + {b_pedestal:.2f} ± {b_pedestal_err:.2f}", color='cyan')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt_third), label=f"Fit: y = {a_third:.2f} ± {a_third_err:.2f} * x + {b_third:.2f} ± {b_third_err:.2f}", color='purple')
plt.errorbar(voltage, [p[0] for p in third_peak_position], yerr=[p[1] for p in third_peak_position], fmt="o", label="Third Peak", color='purple')
plt.errorbar(voltage, [p[0] for p in fourth_peak_position], yerr=[p[1] for p in fourth_peak_position], fmt="o", label="Fourth Peak", color='brown')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt_fourth), label=f"Fit: y = {a_fourth:.2f} ± {a_fourth_err:.2f} * x + {b_fourth:.2f} ± {b_fourth_err:.2f}", color='brown')
plt.xlabel("Voltage (V)")
plt.ylabel("Position (ADC)")
plt.title("Position of First Peak and Pedestal vs Voltage")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "first_peak_and_pedestal_position.png"))

# Plotting the position of the first, second and third peak removing the pedestal position
plt.figure(figsize=(8, 6))
plt.errorbar(voltage, [p[0]- b_pedestal for p in first_peak_position], yerr=[p[1] for p in first_peak_position], fmt="o", label="First Peak", color='green')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt) - b_pedestal, label=f"Fit: y = {a:.2f} ± {a_err:.2f} * x + {b + b_pedestal:.2f} ± {b_err:.2f}", color='blue')
plt.errorbar(voltage, [p[0]- b_pedestal for p in second_peak_position], yerr=[p[1] for p in second_peak_position], fmt="o", label="Second Peak", color='red')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt_second) - b_pedestal, label=f"Fit: y = {a_second:.2f} ± {a_second_err:.2f} * x + {b_second + b_pedestal:.2f} ± {b_second_err:.2f}", color='magenta')
plt.errorbar(voltage, [p[0]- b_pedestal for p in third_peak_position], yerr=[p[1] for p in third_peak_position], fmt="o", label="Third Peak", color='purple')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt_third) - b_pedestal, label=f"Fit: y = {a_third:.2f} ± {a_third_err:.2f} * x + {b_third + b_pedestal:.2f} ± {b_third_err:.2f}", color='purple')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt_fourth) - b_pedestal, label=f"Fit: y = {a_fourth:.2f} ± {a_fourth_err:.2f} * x + {b_fourth + b_pedestal:.2f} ± {b_fourth_err:.2f}", color='brown')
plt.errorbar(voltage, [p[0]- b_pedestal for p in fourth_peak_position], yerr=[p[1] for p in fourth_peak_position], fmt="o", label="Fourth Peak", color='brown')
plt.xlabel("Voltage (V)")
plt.ylabel("Position (ADC)")
plt.title("Position of First Peak and Pedestal vs Voltage")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "first_peak_and_pedestal_position.png"))

# Computing the ratio between the third and second peaks position and between the fourth and the second ones
ratio_3_2 = []
ratio_4_2 = []
for i in range(len(voltage)):
    ratio_3_2.append([(third_peak_position[i][0] - b_pedestal) / (second_peak_position[i][0] - b_pedestal), np.sqrt((third_peak_position[i][1] / second_peak_position[i][0])**2 + (third_peak_position[i][0] * second_peak_position[i][1] / second_peak_position[i][0]**2)**2)])
    ratio_4_2.append([(fourth_peak_position[i][0] - b_pedestal) / (second_peak_position[i][0] - b_pedestal), np.sqrt((fourth_peak_position[i][1] / second_peak_position[i][0])**2 + (fourth_peak_position[i][0] * second_peak_position[i][1] / second_peak_position[i][0]**2)**2)])

# Linear fit of ratios
voltage_array = np.array(voltage)
ratio_3_2_array = np.array([r[0] for r in ratio_3_2])
ratio_3_2_error_array = np.array([r[1] for r in ratio_3_2])
popt, perr = hp.linear_fit(voltage_array, ratio_3_2_array, ratio_3_2_error_array)
a, b = popt
a_err, b_err = perr
ratio_4_2_array = np.array([r[0] for r in ratio_4_2])
ratio_4_2_error_array = np.array([r[1] for r in ratio_4_2])
popt_second, perr_second = hp.linear_fit(voltage_array, ratio_4_2_array, ratio_4_2_error_array)
a_second, b_second = popt_second
a_second_err, b_second_err = perr_second

# Plotting the two ratios
plt.figure(figsize=(8, 6))
plt.plot(voltage, [r[0] for r in ratio_3_2], "o", label="Third Peak / Second Peak", color='darkorange')
plt.plot(voltage, [r[0] for r in ratio_4_2], "o", label="Fourth Peak / Second Peak", color='darkblue')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt), label=f"Fit: y = {a:.2f} ± {a_err:.2f} * x + {b:.2f} ± {b_err:.2f}", color='darkorange')
plt.plot(np.linspace(min(voltage), max(voltage), 100), hp.linear_func(np.linspace(min(voltage), max(voltage), 100), *popt_second), label=f"Fit: y = {a_second:.2f} ± {a_second_err:.2f} * x + {b_second:.2f} ± {b_second_err:.2f}", color='darkblue')
plt.xlabel("Voltage (V)")
plt.ylabel("Ratio")
plt.title("Ratios between peak positions")
plt.tight_layout()
plt.legend(loc = 'upper left')
plt.savefig(os.path.join(results_dir, "ratios_from_pedestal.png"))

plt.show()
