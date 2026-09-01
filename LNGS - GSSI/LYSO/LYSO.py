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

voltage = [39.5, 40.0, 40.5, 41.0, 41.5, 42.0]  
gain = [3, 23, 33, 48, 53]
# voltage = [43.4]
# gain = [48]

bin_size = [(5, 5, 5, 5, 5), (5, 5, 5, 5, 5), (5, 5, 5, 5, 5), (5, 5, 5, 5, 5), (5, 5, 5, 5, 5), (8, 8, 8, 8, 8)]

distances = [(200, 200, 400, 400, 400), (200, 200, 400, 400, 400), (200, 200, 400, 400, 400), (200, 200, 400, 400, 400), (200, 200, 400, 400, 400), (200, 200, 400, 400, 400)]
prominence = [(40, 40, 40, 40, 40), (40, 40, 40, 40, 40), (40, 40, 40, 40, 40),  (40, 40, 40, 40, 40), (40, 40, 40, 40, 40), (50, 50, 50, 50, 50)]

# bin_size = [(2, 2, 8, 8, 8)]

# distances = [(200, 200, 400, 500, 3000)]
# prominence = [(40, 40, 40, 40, 50)]
distances = np.array(distances).reshape(len(voltage), len(gain))
prominence = np.array(prominence).reshape(len(voltage), len(gain))
bin_size = np.array(bin_size).reshape(len(voltage), len(gain))

path_name = "./data/"

max_values = np.zeros((len(voltage), len(gain)))
max_values_positions = np.zeros((len(voltage), len(gain)))

# Bias fixed gain changing
for v in voltage:

    # Plotting data in a two windows plot for high gain and low gain with different x axis (same bias different gain)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    fig.suptitle("Voltage: {} V".format(v))

    for g in gain:

        # Gaussian fit parameters
        chi_square = []
        mean_values = []
        sigma_values = []
        A_values = []

        # Extract che collum number for the specific voltage and gain
        column_number_HG = hp.get_column_number(2, "HG", "D", 17)
        column_number_LG = hp.get_column_number(2, "LG", "D", 17)
        
        try: 
            # Extracting the data from the CSV file
            ADC_HG, counts_HG = hp.extract_SiPM_data(path_name + str(v) + "/gain" + str(g) + "/data.csv", column_number_HG, False)
            ADC_LG, counts_LG = hp.extract_SiPM_data(path_name + str(v) + "/gain" + str(g) + "/data.csv", column_number_LG, False)
        except FileNotFoundError:
            print(f"File not found for voltage {v} and gain {g}. Skipping.")
            continue

        # Rebinning the data with a bin size
        ADC_HG_rebinned, counts_HG_rebinned = hp.rebin_xy(ADC_HG, counts_HG, bin_size[voltage.index(v)][gain.index(g)], x_mode="mean", y_mode="sum")
        ADC_LG_rebinned, counts_LG_rebinned = hp.rebin_xy(ADC_LG, counts_LG, bin_size[voltage.index(v)][gain.index(g)], x_mode="mean", y_mode="sum")

        # Finding peaks in the data
        print(distances[voltage.index(v)][gain.index(g)])
        rebinned_distance = max(1, int(distances[voltage.index(v)][gain.index(g)] / bin_size[voltage.index(v)][gain.index(g)]))
        
        # Searching peaks outiside the pedestal
        counts_HG_rebinned_pedestal = counts_HG_rebinned[ADC_HG_rebinned > 2200]

        # Finding the maximum value in the high gain spectrum outside the pedestal and its position in ADC
        max_values[voltage.index(v)][gain.index(g)] = max(counts_HG_rebinned_pedestal)
        max_values_positions[voltage.index(v)][gain.index(g)] = ADC_HG_rebinned[np.where(counts_HG_rebinned == max(counts_HG_rebinned_pedestal))[0][0]]

        peaks_indices, peaks_values = hp.find_peaks(
            counts_HG_rebinned_pedestal,
            use_scipy=True,
            prominence=prominence[voltage.index(v)][gain.index(g)],
            distance=rebinned_distance,
        )

        # Adjusting the peaks indices to match the original rebinned data
        peaks_indices = peaks_indices + np.where(ADC_HG_rebinned > 2200)[0][0]

        ax2.plot(ADC_HG_rebinned, counts_HG_rebinned, label="High Gain voltage {} gain {}".format(v, g))
        ax1.plot(ADC_LG_rebinned, counts_LG_rebinned, label="Low Gain voltage {} gain {}".format(v, g))
        # ax2.plot(ADC_HG_rebinned[peaks_indices], counts_HG_rebinned[peaks_indices], "x", label="Peaks")
        # ax1.set_title(f"Voltage: {v} V, Gain: {g}")
        ax1.set_ylabel("Counts")
        ax2.set_xlabel("ADC")
        ax2.set_ylabel("Counts")
        ax1.legend()
        ax2.legend()

    # Creating a txt file with the maximum position in ADC for each gain for the specific voltage
    with open(path_name + str(v) + f"/Max_position_ADC_voltage_{v}.txt", "w") as f:
        f.write(f"Voltage: {v} V\n")
        f.write(f"Gain - Max position in ADC\n")
        for g in gain:
            f.write(f"{g} {max_values_positions[voltage.index(v)][gain.index(g)]}\n")

    # Plotting the maximum position in ADC in function of gain for a fixed voltage
    plt.figure(figsize=(8, 6))
    plt.plot(gain, max_values_positions[voltage.index(v)], marker="o", label="Max position in ADC")
    plt.title(f"Max position in ADC for Voltage: {v} V")
    plt.xlabel("Gain")
    plt.ylabel("Max position in ADC")
    plt.legend()
    plt.savefig(path_name + str(v) + f"/Max_position_ADC_voltage_{v}.png", dpi=300, bbox_inches='tight')

    plt.show()

# Gain fixed bias changing
for g in gain:

    # Plotting data in two windows for high gain and low gain with differente x axis (same gain different bias)
    fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=False)
    fig2.suptitle("Gain: {} ".format(g))

    for v in voltage:
        # Extract che collum number for the specific voltage and gain
        column_number_HG = hp.get_column_number(2, "HG", "C", 27)
        column_number_LG = hp.get_column_number(2, "LG", "C", 27)
        
        try: 
            # Extracting the data from the CSV file
            ADC_HG, counts_HG = hp.extract_SiPM_data(path_name + str(v) + "/gain" + str(g) + "/data.csv", column_number_HG, False)
            ADC_LG, counts_LG = hp.extract_SiPM_data(path_name + str(v) + "/gain" + str(g) + "/data.csv", column_number_LG, False)
        except FileNotFoundError:
            print(f"File not found for voltage {v} and gain {g}. Skipping.")
            continue

        # Rebinning the data with a bin size
        ADC_HG_rebinned, counts_HG_rebinned = hp.rebin_xy(ADC_HG, counts_HG, bin_size[voltage.index(v)][gain.index(g)], x_mode="mean", y_mode="sum")
        ADC_LG_rebinned, counts_LG_rebinned = hp.rebin_xy(ADC_LG, counts_LG, bin_size[voltage.index(v)][gain.index(g)], x_mode="mean", y_mode="sum")

        # Finding maximum value in the high gain spectrum outside the pedestal
        counts_HG_rebinned_pedestal = counts_HG_rebinned[ADC_HG_rebinned > 2200]
        max_values[voltage.index(v)][gain.index(g)] = max(counts_HG_rebinned_pedestal)
        max_values_positions[voltage.index(v)][gain.index(g)] = ADC_HG_rebinned[np.where(counts_HG_rebinned == max(counts_HG_rebinned_pedestal))[0][0]]


        ax4.plot(ADC_HG_rebinned, counts_HG_rebinned, label="High Gain voltage {} gain {}".format(v, g))
        ax3.plot(ADC_LG_rebinned, counts_LG_rebinned, label="Low Gain voltage {} gain {}".format(v, g))
        ax3.set_ylabel("Counts")
        ax4.set_xlabel("ADC")
        ax4.set_ylabel("Counts")
        ax3.legend()
        ax4.legend()
        plt.savefig(path_name + f"Spectrum_same_gain_{g}.png", dpi=300, bbox_inches='tight')

    # Creating a txt file with the maximum position in ADC for each voltage for the specific gain
    with open(path_name + f"Max_position_ADC_gain_{g}.txt", "w") as f:
        f.write(f"Gain: {g}\n")
        f.write(f"Voltage - Max position in ADC\n")
        for v in voltage:
            f.write(f"{v} {max_values_positions[voltage.index(v)][gain.index(g)]}\n")
    
    # Plotting the maximum position in ADC in function of voltage for a fixed gain 
    # Don't plot the points that have y values = 0
    plt.figure(figsize=(8, 6))
    plt.plot([v for v in voltage if max_values_positions[voltage.index(v)][gain.index(g)] != 0], [q for q in max_values_positions[:, gain.index(g)] if q != 0], marker="o", label="Max position in ADC")
    plt.title(f"Max position in ADC for Gain: {g}")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Max position in ADC")
    plt.legend()
    plt.savefig(path_name + f"Max_position_ADC_gain_{g}.png", dpi=300, bbox_inches='tight')    

    plt.show()