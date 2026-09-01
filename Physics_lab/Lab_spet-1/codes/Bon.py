
# PROGRAMM FOR DATA ANALYSIS OF GRAY VALUES VS DISTANCE - ZEEMAN EFFECT

# ====================== CLASS IMPORT ======================

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
from scipy.constants import constants

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
PEAK_NUMBER = 26 # number of peaks to be fitted
MEAN_INTERVAL_COMPUTED = 235 # mean interval between peaks in pixels, computed from the data
B_FIELD = (5030 + 4960)/2
B_FIELD_ERR = 0.02  # Assuming the error is the same for both measurements
HC = 1.98645e-16 # J*nm
MU_B = 9.274009994e-28 # J/G

DELTA_ZEEMAN_TEOR = (2*MU_B*B_FIELD*LAMBDA**2/HC)
DELTA_ZEEMAN_TEOR_ERR = DELTA_ZEEMAN_TEOR * B_FIELD_ERR

# Defining a list of possible intervals for the Gaussian fit
max_intervals = np.array([85, 169, 303, 387, 523, 609, 749, 835, 976 , 1067, 1205, 1292, 1437, 1526, 1674, 1762, 1912, 2004, 2151, 2248, 2405, 2499, 2655, 2752, 2915, 3008]) - 20
min_intervals = np.array([41, 123, 260, 342, 480, 563, 702, 784, 928, 1013, 1157, 1244, 1390, 1477, 1625, 1715, 1865, 1956, 2111, 2201, 2353, 2446, 2605, 2701, 2865, 2958]) - 20

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

	# Iteration over 26 gaussian peaks, with a step of 235 pixels (distance between peaks)
	# NO PARABOLIC FIT --> only to work with the background
	for i in range(np.size(max_intervals)):
		
		# Find the maximum in the interval [i*235, i*235 + 235]
		values = get_max(distances, gray_values, y_err, min_intervals[i], max_intervals[i] - min_intervals[i])
		max_values.append(values[1])
		max_values_position.append(values[0])

		if i < np.size(max_intervals) - 1:
			# Find the minimum in the interval
			values = get_min(distances, gray_values, y_err, max_intervals[i], min_intervals[i+1] - max_intervals[i], parabolic)
			min_values.append(values[1])
			min_values_position.append(values[0])
			min_std.append(values[2])


	return np.array(max_values_position), np.array(max_values), np.array(min_values_position), np.array(min_values), np.array(min_std)

# ====================== MAIN CODE ======================

# Defining lists to store distances and gray values
distances = []
gray_values = []
GV_err = []

# Take the data from the file and store it in the lists
with open("./data/data_std_B_on_para.txt", "r") as f:
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

# Getting the first background value position
background_values.append(min_values[1])
background_position.append(min_values_position[1] / 2)
background_deviation.append(min_std[1])

for i in range(1, 23, 2):
		background_values.append((min_values[i] + min_values[i+2]) / 2)
		background_position.append((min_values_position[i] + min_values_position[i+2]) / 2)
		background_deviation.append(np.sqrt((min_std[i]**2 + min_std[i+2]**2) / 4))

# Getting the last background value position
background_values.append(min_values[-2])
background_position.append(min_values_position[-2] + ( - min_values_position[-2] + distances[-1])/2)
background_deviation.append(min_std[-2])

# Finding the mean and standard deviation of the background values
background_mean = np.mean(background_values)
background_std = np.std(background_values)

# Minimum and maximum plot with data
plt.figure(figsize=(8, 5))
plt.plot(distances, gray_values, label="Dati", alpha=0.7, marker='.', linestyle='-')
plt.scatter(max_values_position, max_values, color='green', label="Massimi", zorder=5)
plt.scatter(min_values_position, min_values, color='red', label="Minimi", zorder=5)
#plt.scatter([peak.mu for peak in peaks], [peak.A for peak in peaks], color='blue', label="Massimi da fit", zorder=5)
#plt.scatter([distances[i] for i in min_values_index], [gray_values[i] for i in min_values_index], color='red', label="Minimi", zorder=5)
plt.xlabel("Distance (pixels)")
plt.ylabel("Gray Value")
plt.title("Gray Value vs Distance con Massimi Locali")
plt.legend()
plt.grid(True)
plt.tight_layout()
# # plt.savefig("./plots/gray_values_massimi_minimi.png", dpi=150)

# Founding better parameters for the background(fittable with a guassian peak)
# THIS DATA IS TO BE USED ONLY FOR FITTING PROCEDURE, NOT FOR THE CALCULATION OF THE BACKGROUND
max_values_position, max_values, min_values_position, min_values, min_std = get_all(distances, gray_values, GV_err, parabolic = True)

peaks = []  # List for gauss_peak objects

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
		if i != 25: 
			peak = gauss_line_fit(distances[start_index:end_index], gray_values[start_index:end_index], y_err=GV_err[start_index:end_index], par0=[max_values[i] - min_values[i], max_values_position[i], 0.01, 60, min_values[i]])
		else:
			peak = gauss_line_fit(distances[start_index:end_index], gray_values[start_index:end_index], y_err=GV_err[start_index:end_index], par0=[max_values[i] - min_values[i-1], max_values_position[i], 0.01, 60, min_values[i-1]])
		
		peaks.append(peak)

# 		# # Minimum and maximum plot with data
# 		# plt.figure(figsize=(8, 5))
# 		# plt.plot(distances[start_index:end_index], gray_values[start_index:end_index], label="Dati", alpha=0.7, linestyle='-')
# 		# plt.plot(np.linspace(distances[start_index], distances[end_index], 1000), 
# 		# 			gauss_line(np.linspace(distances[start_index], distances[end_index], 1000), peak.A, peak.mu, peak.alpha, peak.beta, peak.gamma),
# 		# 			color='red', label="Massimi da fit", zorder=5, linestyle='-')
# 		# # plt.fill_between(np.array(distances[start_index:end_index]), gray_values[start_index:end_index] - np.sqrt(np.abs(gray_values[start_index:end_index])),
# 		# # 				gray_values[start_index:end_index] + np.sqrt(np.abs(gray_values[start_index:end_index])), alpha=0.3, color='red', label="Errore", zorder=5)
		
# 		# plt.fill_between(np.array(distances[start_index:end_index]), np.array(gray_values[start_index:end_index]) - GV_err[start_index:end_index],
# 		# 				np.array(gray_values[start_index:end_index]) + GV_err[start_index:end_index], alpha=0.3, color='red', label="Errore", zorder=5)
		
# 		# #plt.scatter([distances[i] for i in min_values_index], [gray_values[i] for i in min_values_index], color='red', label="Minimi", zorder=5)
# 		# plt.xlabel("Distance (pixels)")
# 		# plt.ylabel("Gray Value")
# 		# plt.title("Gray Value vs Distance con Massimi Locali")
# 		# plt.legend()
# 		# plt.grid(True)
# 		# plt.tight_layout()
# 		# plt.show()
	except Exception as e:
		print(f"Error in fitting Gaussian line: {e}")


# SOME ANALYSIS ON THE FITTED PEAKS

# Finding the convertion function
mean_diff_peaks_NOZEMAN = []
variabile_diff = np.zeros(4)

print("PRINTING C1, C1', C2, C2' PEAKS POSITIONS:")
for i in range(0, len(peaks) - 5, 2):
	
	variabile_diff[0] = peaks[i+2].mu - peaks[i].mu
	variabile_diff[1] = peaks[i+3].mu - peaks[i+1].mu
	variabile_diff[2] = peaks[i+4].mu - peaks[i+2].mu
	variabile_diff[3] = peaks[i+5].mu - peaks[i+3].mu

	print("C1: ", variabile_diff[0], "C1': ", variabile_diff[1], "C2: ", variabile_diff[2], "C2': ", variabile_diff[3])
	
	mean_diff_peaks_NOZEMAN.append(np.mean(variabile_diff))

print(" ")
print("Mean differences (NO ZEEMAN):", np.array(mean_diff_peaks_NOZEMAN))
print(" ")

# Finding the space difference between shitted peaks positions
diff_peaks_zeeman = []
diff_peaks_std_zeeman = []

for i in range(0, len(peaks), 2):
	diff_peaks_zeeman.append(peaks[i+1].mu - peaks[i].mu)
	diff_peaks_std_zeeman.append(np.sqrt(peaks[i+1].mu_err**2 + peaks[i].mu_err**2))

diff_peaks_zeeman_mean = np.mean(diff_peaks_zeeman)
diff_peaks_zeeman_std_mean = np.std(diff_peaks_zeeman)

# Computing the convertion factor
convertion = DELTA_RU / np.array(mean_diff_peaks_NOZEMAN)

# Computing the landè g factor and its uncertainty

# Convertion factors computed with Boff
convertion_test = [0.00017669, 0.00017436, 0.0001718, 0.00016951, 0.00016696, 0.00016462,
              0.00016228, 0.00015979, 0.00015754, 0.000155, 0.00015236]

print("Convertion factors: ", convertion)
print("Diff: ", (convertion - convertion_test)*100/convertion)
print(" ")
diff_peaks_zeeman_nm = np.array(diff_peaks_zeeman[1:-1]) * np.array(convertion)
diff_peaks_zeeman_std_nm = np.array(diff_peaks_std_zeeman[1:-1]) * np.array(convertion)

diff_peaks_zeeman_mean_nm = np.mean(diff_peaks_zeeman_nm)
diff_peaks_zeeman_mean_std_nm = np.sqrt((np.std(diff_peaks_zeeman_nm))**2 + (0.02*diff_peaks_zeeman_mean_nm )**2)

g_lande = (HC * diff_peaks_zeeman_mean_nm) / (2 * MU_B * B_FIELD * LAMBDA**2)

g_lande_std = np.sqrt((diff_peaks_zeeman_mean_std_nm)**2 / (diff_peaks_zeeman_mean_nm)**2 + (B_FIELD_ERR)**2) * g_lande

print("G_lande: ", g_lande, " +- ", g_lande_std)
print("Diff peaks Zeeman pixel: ", diff_peaks_zeeman_mean, " +- ", diff_peaks_zeeman_std_mean)
print("Diff peaks Zeeman nm: ", diff_peaks_zeeman_mean_nm, " +- ", diff_peaks_zeeman_mean_std_nm)

# fwhm = []  # List to store the FWHM of the peaks

# for i, peak in enumerate(peaks):
	
# 	# FWHM is the distance between the two points where the function value is half of the maximum value
# 	half_max = ((peak.A + peak.gamma) + background_values[i]) / 2 # Half of the maximum value, considering the background
	
# 	# Find the two points where the function value is half of the maximum value
# 	x_fit = np.linspace(min_intervals[i]-6, max_intervals[i]+6, 5000)
# 	y_fit = gauss_line(x_fit, peak.A, peak.mu, peak.alpha, peak.beta, peak.gamma)

# 	# print(y_fit)
# 	# Find the indices of the points where the function value is half of the maximum value
# 	indices = np.where(np.abs(y_fit - half_max) < 0.1)[0]

# 	if len(indices) >= 2:
# 		fwhm.append(x_fit[indices[-1]] - x_fit[indices[0]])
# 	else:
# 		print(f"Peak {i}: FWHM not found")

# fwhm_mean = np.mean(fwhm)
# fwhm_std = np.std(fwhm)

# FWHM_NM = np.array(fwhm[:-1]) * (DELTA_RU / np.array(diff_peaks)) 

# FWHM_NM_mean = np.mean(FWHM_NM)
# FWHM_NM_std = np.std(FWHM_NM)

# # Compute the resolving power
# RV = LAMBDA / FWHM_NM
# RV_mean = np.mean(RV)
# RV_std = np.std(RV)

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
#plt.text(748, 230, f"B = {B_FIELD/10000:.2f} ± {B_FIELD_ERR*B_FIELD/10000:.2f} T", fontsize=14)
# plt.text(1110, 225, r'$\Delta M_J$ = -1', fontsize=14)
# plt.text(1290, 225, r'$\Delta M_J$ = +1', fontsize=14)
plt.grid(True)

plt.xlim(-100, 3500)
plt.ylim(0, 255)
plt.tight_layout()
# plt.savefig("./plots/fit_gaussiani.png", dpi=150)


# # Plotting the FWHM of the peaks in PIXELS
# plt.figure(figsize=(8, 5))
# plt.plot(np.linspace(0, np.size(fwhm)-1, np.size(fwhm)), fwhm , marker='o', linestyle=' ', color='purple', label = "FWHM")
# plt.fill_between(np.linspace(0, np.size(fwhm)-1, np.size(fwhm)), fwhm_mean - fwhm_std, fwhm_mean + fwhm_std, alpha=0.3, color='plum', label="Standard deviation", zorder=5)
# plt.hlines(fwhm_mean, 0, np.size(fwhm)-1, colors='magenta', linestyles='dashed', label=f"Mean", zorder=5)
# plt.xlabel("Peak Index")
# plt.ylabel("FWHM (pixels)")
# plt.title("Full Width at Half Maximum")
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.savefig("./plots/fwhm.png", dpi=150)	

# # Plotting the FWHM of the peaks in NM
# plt.figure(figsize=(8, 5))
# plt.plot(np.linspace(0, np.size(FWHM_NM)-1, np.size(FWHM_NM)), FWHM_NM , marker='o', linestyle=' ', color='darkgreen', label="FWHM")
# plt.fill_between(np.linspace(0, np.size(FWHM_NM)-1, np.size(FWHM_NM)), FWHM_NM_mean - FWHM_NM_std, FWHM_NM_mean + FWHM_NM_std, alpha=0.3, color='lightgreen', label="Standard deviation", zorder=5)
# plt.hlines(FWHM_NM_mean, 0, np.size(FWHM_NM)-1, colors='green', linestyles='dashed', label=f"Mean", zorder=5)
# plt.xlabel("Peak Index")
# plt.ylabel("FWHM (nm)")
# plt.title("Full Width at Half Maximum in nm")
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.savefig("./plots/fwhm_nm.png", dpi=150)

# Plotting the shift between Zeeman peaks in pixel
plt.figure(figsize=(8, 5))
plt.errorbar(np.linspace(0, np.size(diff_peaks_zeeman)-1, np.size(diff_peaks_zeeman)), diff_peaks_zeeman, label = "Distance between peaks", yerr=diff_peaks_std_zeeman, marker='o', linestyle=' ', color='teal')
plt.fill_between(np.linspace(0, np.size(diff_peaks_zeeman)-1, np.size(diff_peaks_zeeman)), diff_peaks_zeeman_mean - diff_peaks_zeeman_std_mean, diff_peaks_zeeman_mean + diff_peaks_zeeman_std_mean, alpha=0.3, color='lightseagreen', label="Standard deviation", zorder=5)
plt.hlines(diff_peaks_zeeman_mean, 0, np.size(diff_peaks_zeeman)-1, lw = 2, colors='darkcyan', linestyles='dashed', label=f"Mean", zorder=5)
plt.xlabel("Peak Index", fontsize=14)
plt.ylabel("Zeeman shift (pixels)", fontsize=14)
plt.title("Distance between peaks", fontsize=14)
plt.text(8, 83.5, f"Mean: {diff_peaks_zeeman_mean:.0f} ± {diff_peaks_zeeman_std_mean:.0f} pixels", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.legend(loc = 'upper left', fontsize = 12)
#plt.savefig("./plots/Bon_para/distance_between_peaks_pixels.png", dpi=150)

# Plotting the shift between Zeeman peaks in nm
plt.figure(figsize=(8, 5))
print(diff_peaks_zeeman_std_nm)
plt.errorbar(np.linspace(0, np.size(diff_peaks_zeeman_nm)-1, np.size(diff_peaks_zeeman_nm)), diff_peaks_zeeman_nm, label = "Distance between peaks", yerr=diff_peaks_zeeman_std_nm, marker='o', linestyle=' ', color='teal')
plt.fill_between(np.linspace(0, np.size(diff_peaks_zeeman_nm)-1, np.size(diff_peaks_zeeman_nm)), diff_peaks_zeeman_mean_nm - diff_peaks_zeeman_mean_std_nm, diff_peaks_zeeman_mean_nm + diff_peaks_zeeman_mean_std_nm, alpha=0.3, color='lightseagreen', label="Standard deviation", zorder=5)
plt.hlines(diff_peaks_zeeman_mean_nm, 0, np.size(diff_peaks_zeeman_nm)-1, colors='darkcyan', linestyles='dashed', label=f"Mean", zorder=5)
plt.xlabel("Peak Index", fontsize=14)
plt.ylabel("Zeeman shift (nm)", fontsize=14)
plt.title("Distance between peaks in nm", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.ylim(0.0141, 0.0154)
plt.legend(loc = 'lower right', fontsize = 12)
plt.text(1, 0.01434, f"Mean: {diff_peaks_zeeman_mean_nm:.4f} ± {diff_peaks_zeeman_mean_std_nm:.4f} nm", fontsize=12)
plt.text(1, 0.01427, r'$g_{jls}$:' f' {g_lande:.2f} ± {g_lande_std:.2f}', fontsize=12)
plt.text(1, 0.01420, f"Compatibility: {(1-g_lande)/g_lande_std:.1f}", fontsize=12)
plt.savefig("./plots/Bon_para/distance_between_peaks_nm.png", dpi=150)

# Only background plot
plt.figure(figsize=(8, 5))
plt.errorbar(background_position, background_values, yerr=background_deviation, label = "Background", marker='o', linestyle=' ', color='red')
plt.hlines(background_mean, background_position[0], background_position[-1], colors='magenta', linestyles='dashed', label=f"Mean", zorder=5)
plt.fill_between(background_position, background_mean - background_std, background_mean + background_std, alpha=0.3, color='plum', label="Standard deviation", zorder=5)
plt.xlabel("Distance (pixels)", fontsize=14)
plt.ylabel("Gray Value", fontsize=14)
plt.title("Background values", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=12, loc = 'lower right')
plt.text(750, 36, f"Mean: {background_mean:.0f} ± {background_std:.0f} pixels", fontsize=14)
#plt.savefig("./plots/Bon_para/background_values.png", dpi=150)

# # Plotting the resolving power
# plt.figure(figsize=(8, 5))
# plt.plot(np.linspace(0, np.size(RV)-1, np.size(RV)), RV , marker='o', linestyle=' ', color='darkred', label = "Resolving power")
# plt.fill_between(np.linspace(0, np.size(RV)-1, np.size(RV)), RV_mean - RV_std, RV_mean + RV_std, alpha=0.3, color='lightcoral', label="Standard deviation", zorder=5)
# plt.hlines(RV_mean, 0, np.size(RV)-1, colors='red', linestyles='dashed', label=f"Mean", zorder=5)
# plt.xlabel("Peak Index")
# plt.ylabel("Resolving Power")
# plt.title("Resolving Power")
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.savefig("./plots/resolving_power.png", dpi=150)

# print(RV_mean, " +- ", RV_std)
print("PRINT ALL DATA")
for i in range(len(background_position)):
	print(f"Background {i}: {background_values[i]:.2f} ± {background_deviation[i]:.2f} at position {background_position[i]:.2f} pixels")

print(" ")
for i in range(len(peaks)):
	print(f"Peak position {i}: mu = {peaks[i].mu:.2f} ± {peaks[i].mu_err:.2f}")

print(" ")
for i in range(len(diff_peaks_zeeman) - 2):
	print(f"Distance between zeeman peaks {i} and {i + 1}: {diff_peaks_zeeman[i]:.2f} ± {diff_peaks_std_zeeman[i]:.2f} pixels, {diff_peaks_zeeman_nm[i]:.4f} ± {diff_peaks_zeeman_std_nm[i]:.4f} nm")

print("---------------------------------------")
print("PRINT ALL INFORMATION")
print("Background mean: ", background_mean, " +- ", background_std)
print("Distance between peaks in pixels: ", diff_peaks_zeeman_mean, " +- ", diff_peaks_zeeman_std_mean)
print("Distance between peaks in nm: ", diff_peaks_zeeman_mean_nm, " +- ", diff_peaks_zeeman_mean_std_nm)
print("Landè g factor: ", g_lande, " +- ", g_lande_std)
print("Compatibility with theoretical value: ", (1-g_lande)/g_lande_std, " sigma")
print("Theoretical distance between peaks in nm: ", DELTA_ZEEMAN_TEOR, " +- ", DELTA_ZEEMAN_TEOR_ERR)
print("Compatibility with theoretical value: ", (DELTA_ZEEMAN_TEOR - diff_peaks_zeeman_mean_nm)/np.sqrt(DELTA_ZEEMAN_TEOR_ERR**2 + diff_peaks_zeeman_mean_std_nm**2), " sigma")
plt.show()
