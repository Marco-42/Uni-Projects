
# PROGRAMM FOR DATA ANALYSIS OF GRAY VALUES VS DISTANCE - ZEEMAN EFFECT

# ====================== CLASS IMPORT ======================

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler

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

# Variable definition:
DELTA_RU = 0.0387115774490153 # peak distance in nm
LAMBDA = 585.3 # emission radiation wavelength in nm (Neon)
PEAK_NUMBER = 13 # number of peaks in the data
MEAN_INTERVAL_COMPUTED = 235 # mean interval between peaks in pixels, computed from the data

# ====================== CLASS DEFINITION ======================

class gauss_peak:
	"""
	Class to represent a Gaussian peak with its parameters and uncertainties.
	"""
	def __init__(self, A, mu, alpha, beta, gamma, A_err, mu_err, alpha_err, beta_err, gamma_err):
		self.A = A
		self.mu = mu
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.A_err = A_err
		self.mu_err = mu_err
		self.alpha_err = alpha_err
		self.beta_err = beta_err
		self.gamma_err = gamma_err

# ====================== FITTING FUNCTIONS DEFINITION ======================

# Define a parabolic fit function: f(x) = ax^2 + bx + c
def parabolic(x, a, b, c):
	"""
	Parabolic fit function: f(x) = ax^2 + bx + c
	"""
	return a * x**2 + b * x + c

# Define a Gaussian line function: f(x) = A * exp(- 0.5 * ((x - mu)^2) / (alpha * x + beta)^2) + gamma
def gauss_line(x, A, mu, alpha, beta, gamma):
	"""
	Gaussian line function: f(x) = A * exp(- 0.5 * ((x - mu)^2) / (alpha * x + beta)^2) + gamma
	"""
	return A * np.exp(- 0.5 * ((x - mu)**2) / ((alpha * x + beta)**2)) + gamma

# Function to fit a parabola to the data in a given interval and find the vertex (maximum point)
def parabolic_fit(x, y, y_err):
	
	# Need at least 3 points for a parabolic fit
	if len(x) < 3:
		return None 

	try:
		popt, pcov = curve_fit(parabolic, x, y, sigma=y_err, absolute_sigma=True)
		a, b, c = popt

		# Vertex of the parabola: x = -b/(2a)
		if a == 0:
			return None
		x_min = -b / (2 * a)
		y_min = parabolic(x_min, a, b, c)
		perr = np.sqrt(np.diag(pcov))
		sy_min = np.abs(x_min)*np.sqrt((perr[1]/(b))**2 + (perr[0]/(a))**2)
		return x_min, y_min, sy_min, popt, perr
	
	except Exception as e:
		print(f"Error in parabolic fit: {e}")
		return None

# Function to fit a Gaussian line to the data in a given interval and find the parameters
def gauss_line_fit(x, y, y_err = None, par0=None):
	
	# Need at least 5 points for a Gaussian fit
	if len(x) < 5:
		return None 

	if y_err is None:
		y_err = np.ones_like(y)

	try:
		popt, pcov = curve_fit(gauss_line, x, y, sigma=y_err, absolute_sigma=True, p0=par0)
		A, mu, alpha, beta, gamma = popt
		perr = np.sqrt(np.diag(pcov))

		result = gauss_peak(A, mu, alpha, beta, gamma, perr[0], perr[1], perr[2], perr[3], perr[4])
		
		return result
	
	except Exception as e:
		print(f"Error in Gaussian line fit: {e}")
		return None
	
# ====================== FUNCTIONS DEFINITION ======================

# Function to find local minima in a specific range of data
def get_min(x, y, y_err, start, len, parabolic = False):
	"""
	Find the minimum value of y in the interval [start, start + len] \n
	Return position - value of the minimum
	"""

	x = np.array(x)
	y = np.array(y)

	# Find the index of the first element in x that is greater than or equal to start
	start_index = np.where(x >= start)[0][0]
	try:
		end_index = np.where(x >= start + len)[0][0]
	except IndexError:
		end_index = np.size(x) - 1

	found_x = x[start_index:end_index]
	found_y = y[start_index:end_index]

	# First find the discrete minimum in the interval
	min_value = min(found_y)
	min_index = start_index + np.where(found_y == min_value)[0][0]
	min_position = x[min_index]
	min_std = y_err[min_index]

	if(min_index > 100):
		# Find new x and y values around the discrete minimum for a more accurate parabolic fit
		found_x = x[min_index - 60 : min_index + 60]
		found_y = y[min_index - 60 : min_index + 60]
		found_y_err = y_err[min_index - 60 : min_index + 60]

		if parabolic:
			# Parabolic fit in the interval to find a more accurate minimum
			fit_result = parabolic_fit(found_x, found_y, found_y_err)
		else: 
			fit_result = None

		if fit_result is not None:
			min_position, min_value, min_std, popt, perr = fit_result
			min_index = start_index + np.argmin(np.abs(found_x - min_position))		

	return min_position, min_value, min_std

# Function to find local maxima in a specific range of data
def get_max(x, y, y_err, start, len, parabolic = False):
	"""
	Find the maximum value of y in the interval [start, start + len] \n
	Return position - value of the maximum
	"""

	x = np.array(x)
	y = np.array(y)

	# Find the index of the first element in x that is greater than or equal to start
	start_index = np.where(x >= start)[0][0]
	try:
		end_index = np.where(x >= start + len)[0][0]
	except IndexError:
		end_index = np.size(x) - 1

	found_x = x[start_index:end_index]
	found_y = y[start_index:end_index]

	max_value = max(found_y)
	max_index = start_index + np.where(found_y == max_value)[0][0]
	max_position = x[max_index]

	return max_position, max_value

def get_all(distances, gray_values, y_err, parabolic = False):
	
	# Lists to store the positions and values of the minima and maxima
	max_values_position = []
	min_values_position = []
	max_values = []
	min_values = []
	min_std = []
	
	# Found the minimum value in the first 100 pixels (background) and store it in the list of minima
	min_variable = get_min(distances, gray_values, y_err, 0, 100, parabolic)
	min_values_position.append(min_variable[0])
	min_values.append(min_variable[1])
	min_std.append(min_variable[2])

	# Iteration over 13 gaussian peaks, with a step of 235 pixels (distance between peaks)
	# NO PARABOLIC FIT --> only to work with the background
	for i in range(0, PEAK_NUMBER):
		
		# Find the maximum in the interval [i*235, i*235 + 235]
		values = get_max(distances, gray_values, y_err, i*MEAN_INTERVAL_COMPUTED, MEAN_INTERVAL_COMPUTED)
		max_values.append(values[1])
		max_values_position.append(values[0])

		if(i <= 11):
			values = get_min(distances, gray_values, y_err, 130 + i*MEAN_INTERVAL_COMPUTED, MEAN_INTERVAL_COMPUTED, parabolic)
			min_values.append(values[1])
			min_values_position.append(values[0])
			min_std.append(values[2])

	# Found the minimum value in the last 100 pixels (background) and store it in the list of minima
	min_variable = get_min(distances, gray_values, y_err, 2950, 100, parabolic=False)
	min_values_position.append(min_variable[0])
	min_values.append(min_variable[1])
	min_std.append(min_variable[2])

	return np.array(max_values_position), np.array(max_values), np.array(min_values_position), np.array(min_values), np.array(min_std)

# ====================== MAIN CODE ======================

# Defining lists to store distances and gray values
distances = []
gray_values = []
GV_err = []

# Take the data from the file and store it in the lists
with open("./data/data_std_B_off_para.txt", "r") as f:
	next(f)  # Not take into account the first line (header)
	for line in f:
		parts = line.strip().split(",")
		if len(parts) == 3:
			try:
				distances.append(float(parts[0]))
				gray_values.append(float(parts[1]))
				GV_err.append(float(parts[2]))
			except ValueError:
				pass  # Ignore lines that cannot be converted to float

max_values_position, max_values, min_values_position, min_values, min_std = get_all(distances, gray_values, GV_err, parabolic = False)

# Finding the background values under the single peak
background_values = []
background_position = []
background_deviation = []

for i in range(0, PEAK_NUMBER):
	background_values.append((min_values[i] + min_values[i+1]) / 2)
	background_position.append((min_values_position[i] + min_values_position[i+1]) / 2)
	background_deviation.append(np.sqrt((min_std[i]**2 + min_std[i+1]**2) / 4))

background_mean = np.mean(background_values)
background_std = np.std(background_values)

# # Minimum and maximum plot with data
# plt.figure(figsize=(8, 5))
# plt.plot(distances, gray_values, label="Dati", alpha=0.7, marker='.', linestyle='-')
# plt.scatter(max_values_position, max_values, color='green', label="Massimi", zorder=5)
# plt.scatter(min_values_position, min_values, color='red', label="Minimi", zorder=5)
# #plt.scatter([peak.mu for peak in peaks], [peak.A for peak in peaks], color='blue', label="Massimi da fit", zorder=5)
# #plt.scatter([distances[i] for i in min_values_index], [gray_values[i] for i in min_values_index], color='red', label="Minimi", zorder=5)
# plt.xlabel("Distance (pixels)")
# plt.ylabel("Gray Value")
# plt.title("Gray Value vs Distance con Massimi Locali")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./plots/gray_values_massimi_minimi.png", dpi=150)

# Founding better parameters for the background(fittable with a guassian peak)
# THIS DATA IS TO BE USED ONLY FOR FITTING PROCEDURE, NOT FOR THE CALCULATION OF THE BACKGROUND
max_values_position, max_values, min_values_position, min_values, min_std = get_all(distances, gray_values, GV_err, parabolic = True)

peaks = []  # List for gauss_peak objects

# Defining a list of possible intervals for the Gaussian fit
max_intervals = np.array([166, 387, 608, 832.5, 1057.5, 1287, 1520, 1756, 1995.5, 2241, 2489, 2740, 2995]) - 38.5
min_intervals = np.array([107.5, 323, 544, 767.5, 996.5, 1226, 1459, 1695.5, 1939, 2178, 2425, 2676.5, 2935]) - 38.5


for i in range(len(max_intervals)):

	# Find the index of the first element in x that is greater than or equal to start
	start_index = np.where(np.array(distances) >= min_intervals[i])[0][0]
	try:
		end_index = np.where(np.array(distances) >= max_intervals[i])[0][0]
	except IndexError:
		end_index = np.size(distances) - 1

	# Fitting a Gaussian line to the data in the interval [i*235, i*235 + 235], removing the background
	# and store the parameters in the list of peaks
	try:
		peak = gauss_line_fit(distances[start_index:end_index], gray_values[start_index:end_index], y_err=GV_err[start_index:end_index], par0=[max_values[i] - min_values[i], max_values_position[i], 0.01, 60, min_values[i]])
		#print(f"Peak teor {i}: A = {max_values[i] - min_values[i]}, mu = {max_values_position[i]}, alpha = {0.1}, beta = {0.1}, gamma = {min_values[i]}")
		peaks.append(peak)
		
		# # Minimum and maximum plot with data
		# plt.figure(figsize=(8, 5))
		# plt.plot(distances[start_index:end_index], gray_values[start_index:end_index], label="Dati", alpha=0.7, linestyle='-')
		# plt.plot(np.linspace(distances[start_index], distances[end_index], 1000), 
		# 			gauss_line(np.linspace(distances[start_index], distances[end_index], 1000), peak.A, peak.mu, peak.alpha, peak.beta, peak.gamma),
		# 			color='red', label="Massimi da fit", zorder=5, linestyle='-')
		# # plt.fill_between(np.array(distances[start_index:end_index]), gray_values[start_index:end_index] - np.sqrt(np.abs(gray_values[start_index:end_index])),
		# # 				gray_values[start_index:end_index] + np.sqrt(np.abs(gray_values[start_index:end_index])), alpha=0.3, color='red', label="Errore", zorder=5)
		
		# plt.fill_between(np.array(distances[start_index:end_index]), np.array(gray_values[start_index:end_index]) - GV_err[start_index:end_index],
		# 				np.array(gray_values[start_index:end_index]) + GV_err[start_index:end_index], alpha=0.3, color='red', label="Errore", zorder=5)
		
		# #plt.scatter([distances[i] for i in min_values_index], [gray_values[i] for i in min_values_index], color='red', label="Minimi", zorder=5)
		# plt.xlabel("Distance (pixels)")
		# plt.ylabel("Gray Value")
		# plt.title("Gray Value vs Distance con Massimi Locali")
		# plt.legend()
		# plt.grid(True)
		# plt.tight_layout()
		# plt.show()
	except Exception as e:
		print(f"Error in fitting Gaussian line: {e}")


# SOME ANALYSIS ON THE FITTED PEAKS

# Finding the space difference between peaks positions
diff_peaks = []
diff_peaks_std = []

for i in range(len(peaks)-1):
	diff_peaks.append(peaks[i+1].mu - peaks[i].mu)
	diff_peaks_std.append(np.sqrt(peaks[i+1].mu_err**2 + peaks[i].mu_err**2))

diff_peaks_mean = np.mean(diff_peaks)
diff_peaks_std_mean = np.std(diff_peaks)

# Finding the FWHM of the peaks and the corresponding distances

fwhm = []  # List to store the FWHM of the peaks

for i, peak in enumerate(peaks):
	
	# FWHM is the distance between the two points where the function value is half of the maximum value
	half_max = ((peak.A + peak.gamma) + background_values[i]) / 2 # Half of the maximum value, considering the background
	
	# Find the two points where the function value is half of the maximum value
	x_fit = np.linspace(min_intervals[i]-6, max_intervals[i]+6, 5000)
	y_fit = gauss_line(x_fit, peak.A, peak.mu, peak.alpha, peak.beta, peak.gamma)

	# print(y_fit)
	# Find the indices of the points where the function value is half of the maximum value
	indices = np.where(np.abs(y_fit - half_max) < 0.1)[0]

	if len(indices) >= 2:
		fwhm.append(x_fit[indices[-1]] - x_fit[indices[0]])
	else:
		print(f"Peak {i}: FWHM not found")

fwhm_mean = np.mean(fwhm)
fwhm_std = np.std(fwhm)

convertion_factors = DELTA_RU / np.array(diff_peaks)
convertion_factors_mean = []

for i in range(len(convertion_factors) -1):
	convertion_factors_mean.append((convertion_factors[i] + convertion_factors[i+1])/2)

print("Mean convertion factors: ", convertion_factors_mean)
FWHM_NM = np.array(fwhm[1:-1]) * np.array(convertion_factors_mean) 

FWHM_NM_mean = np.mean(FWHM_NM)
FWHM_NM_std = np.std(FWHM_NM)

# Compute the resolving power
RV = LAMBDA / FWHM_NM
RV_mean = np.mean(RV)
RV_std = np.std(RV)

# PLOTTING PROCEDURE

# Plotting all gaussian fit
plt.figure(figsize=(8, 5))
plt.plot(distances, gray_values, label="Data", alpha=0.7, marker='.', linestyle='-', color = 'darkblue', linewidth = 0.5)	

# Adding point to the background only for visual operation
backG = np.zeros(len(background_position) + 2)
backG_positiion = np.zeros(len(background_position) + 2)
backG[0] = background_values[0]
backG[1:-1] = background_values
backG[-1] = background_values[-1]
backG_positiion[0] = distances[0]
backG_positiion[1:-1] = background_position
backG_positiion[-1] = distances[-1]

# Plotting the background
plt.fill_between(np.array(backG_positiion), 0, np.array(backG), alpha=0.6, color='gold', label="Background", zorder=5)

# Plotting the error region
plt.fill_between(np.array(distances), np.array(gray_values) - GV_err,
				np.array(gray_values) + GV_err, alpha=0.3, color='dodgerblue', label="Data standard deviation", zorder=5)


for i, peak in enumerate(peaks):
	# Defining the intervall for each peak
	x_min = min_intervals[i]
	x_max = max_intervals[i]
	x_fit = np.linspace(x_min, x_max, 1000)
	if i == 0:
		plt.plot(x_fit,
			 gauss_line(x_fit, peak.A, peak.mu, peak.alpha, peak.beta, peak.gamma),
			 label=f"Peak gaussian fit", zorder=5, linestyle='-', color='darkorange', linewidth=2.5)
	else:
		plt.plot(x_fit,
				gauss_line(x_fit, peak.A, peak.mu, peak.alpha, peak.beta, peak.gamma), zorder=5, linestyle='-', color='darkorange', linewidth = 2.5)

plt.xlabel("Distance (pixels)", fontsize=14)
plt.ylabel("Gray Value", fontsize=14)
plt.title("Peak gaussian fit", fontsize=14)
plt.legend(ncol = 2, fontsize=12)
plt.text(840, 230, r'$\lambda$ = 585.3 nm', fontsize=14)
plt.grid(True)

plt.xlim(-100, 3500)
plt.ylim(0, 255)
plt.tight_layout()
#plt.savefig("./plots/Boff/fit_gaussiani.png", dpi=150)


# Plotting the FWHM of the peaks in PIXELS
plt.figure(figsize=(8, 5))
plt.plot(np.linspace(0, np.size(fwhm)-1, np.size(fwhm)), fwhm , marker='o', linestyle=' ', color='darkred', label = "FWHM")
plt.fill_between(np.linspace(0, np.size(fwhm)-1, np.size(fwhm)), fwhm_mean - fwhm_std, fwhm_mean + fwhm_std, alpha=0.3, color='plum', label="Standard deviation", zorder=5)
plt.hlines(fwhm_mean, 0, np.size(fwhm)-1, colors='darkorange', lw = 2, linestyles='dashed', label=f"Mean", zorder=0)
plt.xlabel("Peak Index", fontsize=14)
plt.ylabel("FWHM (pixels)", fontsize=14)
plt.title("Full Width at Half Maximum", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.text(2.5, 56, f"Mean: {fwhm_mean:.0f} ± {fwhm_std:.0f} pixels", fontsize=14)
plt.ylim(53.5, 66)
plt.legend(loc = 'lower right', fontsize=12)
#plt.savefig("./plots/Boff/fwhm.png", dpi=150)	

# Plotting the FWHM of the peaks in NM
plt.figure(figsize=(8, 5))
plt.plot(np.linspace(0, np.size(FWHM_NM)-1, np.size(FWHM_NM)), FWHM_NM , marker='o', linestyle=' ', color='darkgreen', label="FWHM")
plt.fill_between(np.linspace(0, np.size(FWHM_NM)-1, np.size(FWHM_NM)), FWHM_NM_mean - FWHM_NM_std, FWHM_NM_mean + FWHM_NM_std, alpha=0.3, color='lightgreen', label="Standard deviation", zorder=5)
plt.hlines(FWHM_NM_mean, 0, np.size(FWHM_NM)-1, colors='green', lw = 2, linestyles='dashed', label=f"Mean", zorder=0)
plt.xlabel("Peak Index", fontsize=14)
plt.ylabel("FWHM (nm)", fontsize=14)
plt.title("Full Width at Half Maximum in nm", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.text(3, 0.0097, f"Mean: {FWHM_NM_mean:.4f} ± {FWHM_NM_std:.4f} nm", fontsize=14)
plt.legend(fontsize=12)
plt.ylim(0.00955, 0.0105)
plt.savefig("./plots/Boff/fwhm_nm.png", dpi=150)

# Plotting the distance between peaks
plt.figure(figsize=(8, 5))
plt.errorbar(np.linspace(0, np.size(diff_peaks)-1, np.size(diff_peaks)), diff_peaks, yerr=diff_peaks_std, label = "Distance between peaks", marker='o', linestyle=' ', color='teal')
plt.fill_between(np.linspace(0, np.size(diff_peaks)-1, np.size(diff_peaks)), diff_peaks_mean - diff_peaks_std_mean, diff_peaks_mean + diff_peaks_std_mean, alpha=0.3, color='lightseagreen', label="Standard deviation", zorder=5)
plt.hlines(diff_peaks_mean, 0, np.size(diff_peaks)-1, colors='darkcyan', lw = 2, linestyles='dashed', label=f"Mean", zorder=0)
plt.xlabel("Peak Index", fontsize=14)
plt.ylabel("Distance between peaks (pixels)", fontsize=14)
plt.title("Distance between peaks", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.text(6, 220, f"Mean: {diff_peaks_mean:.0f} ± {diff_peaks_std_mean:.0f} pixels", fontsize=14)
plt.legend(loc = 'upper left', fontsize=12)
#plt.savefig("./plots/Boff/distance_between_peaks.png", dpi=150)

# Only background plot
plt.figure(figsize=(8, 5))
plt.errorbar(background_position, background_values, yerr=background_deviation, label = "Background", marker='o', linestyle=' ', color='darkblue')
plt.hlines(background_mean, background_position[0], background_position[-1], colors='darkorange', lw = 2, linestyles='dashed', label=f"Mean", zorder=0)
plt.fill_between(background_position, background_mean - background_std, background_mean + background_std, alpha=0.3, color='gold', label="Standard deviation", zorder=5)
plt.xlabel("Distance (pixels)", fontsize=14)
plt.ylabel("Gray Value", fontsize=14)
plt.title("Background values", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.text(750, 21, f"Mean: {background_mean:.0f} ± {background_std:.0f} pixels", fontsize=14)
plt.legend(loc = 'lower right', fontsize=12)
#plt.savefig("./plots/Boff/background_values.png", dpi=150)

# Plotting the resolving power
plt.figure(figsize=(8, 5))
plt.plot(np.linspace(0, np.size(RV)-1, np.size(RV)), RV , marker='o', linestyle=' ', color='darkred', label = "Resolving power")
plt.fill_between(np.linspace(0, np.size(RV)-1, np.size(RV)), RV_mean - RV_std, RV_mean + RV_std, alpha=0.3, color='lightcoral', label="Standard deviation", zorder=5)
plt.hlines(RV_mean, 0, np.size(RV)-1, colors='red', linestyles='dashed', label=f"Mean", zorder=0, lw = 2)
plt.xlabel("Peak Index", fontsize=14)
plt.ylabel("Resolving Power", fontsize=14)
plt.title("Resolving Power", fontsize=14)
plt.text(2, 60500, f"Mean: {RV_mean:.0f} ± {RV_std:.0f}", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.ylim(55500, 62000)
plt.savefig("./plots/Boff/resolving_power.png", dpi=150)

# Printing all possible results
print("ALL DATA ANALYSIS RESULTS:")

print(np.array(convertion_factors_mean))
print("MEAN CONVERSTION FACTOR um -> pixels: ", 1000 * np.mean(DELTA_RU / np.array(diff_peaks)), "±", 1000 * np.std(DELTA_RU / np.array(diff_peaks)))
print("MEAN CONVERSTION FACTOR pixels -> um: ", np.mean(diff_peaks_mean) / (1000 * DELTA_RU), "±", diff_peaks_std_mean / (1000 * DELTA_RU ))
print(" ")

delta_x = DELTA_RU * 280 / LAMBDA

print("DELTA_RU / LAMBDA: ", DELTA_RU / LAMBDA)
print(f"DELTA_X: {delta_x} mm")
print("MEAN CONVERSTION FACTOR pixels -> um: ", np.mean(diff_peaks_mean) / (1000*delta_x), "±", diff_peaks_std_mean / (1000*delta_x))
print(" ")

print(" ")
for i in range(len(peaks)):
	print("Peak position (pixels): ", peaks[i].mu, "±", peaks[i].mu_err)

print(" ")
for i in range(len(diff_peaks)):
    print(f"Distance between peaks (pixels): {diff_peaks[i]} +- {diff_peaks_std[i]}")

print(" ")
for i in range(len(fwhm)):
    print(f"FWHM (pixels): {fwhm[i]}")

print(" ")
for i in range(len(FWHM_NM)):
	print(f"FWHM (nm): {FWHM_NM[i]}")

print(" ")
for i in range(len(background_values)):
	print(f"Background value {i}: {background_values[i]} +- {background_deviation[i]}")

print(" ")
for i in range(len(RV)):
	print(f"Resolving power {i}: {RV[i]}")

print(" ")
print("-------------------------------------------------")
print("MEAN VALUES:")
print("Distance between peaks (pixels): ", diff_peaks_mean, "±", diff_peaks_std_mean)
print("FWHM (pixels): ", fwhm_mean, "±", fwhm_std)
print("FWHM (nm): ", FWHM_NM_mean, "±", FWHM_NM_std)
print("Background values: ", background_mean, "±", background_std)
print("Resolving power: ", RV_mean, "±", RV_std)

plt.show()
