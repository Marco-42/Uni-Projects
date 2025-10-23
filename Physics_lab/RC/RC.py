import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
import mplhep as hep
from cycler import cycler
import statistics
from scipy.odr import *

COLLUM = 4 # Number of columns in the data file

#============= HELPER FUNCTIONS =============
# Function to adapt a file to a common format and read data
def read_data(path):
	"""Read the file(path in input) and return three arrays: t (float), y (float), t_scale (float), and y_scale (float).

	The parser ignores lines starting with '#', replaces commas with
	decimal points, and takes the first two numeric columns found in the line.
	"""
	t = []
	y = []
	t_scale = []
	y_scale = []
	
	with open(path, 'r', encoding='utf-8') as f:
		
        # File adapted to a common format
		for line in f: # Take all the different lines
			line = line.strip()
			if not line or line.startswith('#'): # Ignore empty lines and comments
				continue
			# normalize decimal comma if presents
			line = line.replace(',', '.')
			# split on whitespace
			parts = line.split()

			# Extracting data from adapted file
			nums = []
			for p in parts:
				try:
					nums.append(float(p))
				except ValueError:
					continue
				if len(nums) >= COLLUM:
					break
			if len(nums) >= COLLUM:
				t.append(nums[0])
				y.append(nums[1])
				t_scale.append(nums[2])
				y_scale.append(nums[3])

    # Convert to numpy arrays and return
	return np.array(t, dtype=float), np.array(y, dtype=float), np.array(t_scale, dtype=float), np.array(y_scale, dtype=float)

#============= FITTING FUNCTIONS ============
# Exponential model function for odr
def odr_exp(beta, x):
	"""beta[0] * exp(beta[1] * x) + beta[2]"""
	return beta[0] * np.exp(beta[1] * x) + beta[2]

# Exponential model function for curve_fit
def exp_model(x, A, b, C):
    return A * np.exp(b * x) + C

# Fit function
def fit_exponential(t, y, t_error=None, y_error=None):
	"""Fit using ODR (errors on both axes) -  t, y, t_error, y_error.

	t, y, t_error, y_error can be numpy arrays. Fitting result are two vectors and a chi2 value:
	the first one contains the optimized parameters, while the second one contains
	their uncertainties.
	"""
	# reorder for increasing t for cleaner plot
	order = np.argsort(t)
	t_sorted = t[order]
	y_sorted = y[order]
	if t_error is not None:
		t_error = np.asarray(t_error)[order]
	if y_error is not None:
		y_error = np.asarray(y_error)[order]

	# initial parameters input: A ~ (max-min), C ~ min, b ~ -1 / (range_t)
	A0 = (np.max(y_sorted))
	C0 = 0
	b0 = -1000
	p0 = [A0, b0, C0]

	# Fitting datas with an exponential function using ODR (Orthogonal Distance Regression)
	# Create a model for fitting - Function the fitting method is able to read
	function_model = Model(odr_exp)

	# Create a RealData object using our initiated data from above
	graph_data = RealData(t_sorted, y_sorted, sx=t_error, sy=y_error)

	# Set up ODR with the model and data - ODR is the fitting method used to consider the errors both in x and y
	# beta0 are the input parameters
	odr = ODR(graph_data, function_model, beta0=p0)

	# Start the fitting method
	out = odr.run()

	# Get chi square from fit
	chi2 = out.sum_square

	# get parameters and parameters errors from fit 
	params = out.beta
	params_err = out.sd_beta

    # Return optimized parameters and their uncertainties
	return params, params_err, chi2

#============= PLOTTING FUNCTION ============
# Plotting function for the fit results
def plot_and_save(t, y, st, sy, popt, perr, outpath):
	
    # Taking points for the fit line
	t_fine = np.linspace(np.min(t), np.max(t), 400)
	y_fit = exp_model(t_fine, *popt)

    # Plotting
	plt.figure(figsize=(8, 5))
	plt.errorbar(t, y, xerr=st, yerr=sy, fmt='o', label='Datas', color='black', ms = 3, lw = 1.5)
	plt.plot(t_fine, y_fit, label='Fit: A exp(b t) + C', color='red')
	plt.xlabel('Time')
	plt.ylabel('Voltage (V)')
	plt.title('Exponential fit of RC data')
	plt.legend()

	# Taking parameters for annotation
	A, b, C = popt
	dA, db, dC = perr
	text = (
		f'A = {A:.3g} ± {dA:.3g}\n'
		f'b = {b:.3g} ± {db:.3g}\n'
		f'C = {C:.3g} ± {dC:.3g}'
	)

	plt.tight_layout()
	plt.savefig(outpath, dpi=150)
	print(f'Graph saved to: {outpath}')
	plt.show()

#============= MAIN FUNCTION ================
def main():
	# resolve the relative path to the data file in the same folder as the script
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(base_dir, 'data_RCt_BNC.txt')
	if not os.path.exists(data_path):
		raise FileNotFoundError(f'Data file not found: {data_path}')

	t, y, t_scale, y_scale = read_data(data_path)
	if t.size == 0:
		raise RuntimeError('No valid data read from file.')

	st = t_scale * 1e-6 * 2/(5*np.sqrt(24)) # Triangular distribution applied considering max error
	sy = np.sqrt(pow(y_scale/(10*np.sqrt(3)), 2) + pow(y*3/(np.sqrt(3)*100), 2)) # 1/10 reading error with max uniform distribution + 3% scale error with max uniform distribution
	t = t * 1e-6 # Convert microseconds to seconds

	popt, perr, chi2 = fit_exponential(t, y, t_error=st, y_error=sy)

	print('Fit parameters (A, b, C):')
	print(f'  A = {popt[0]:.6g} ± {perr[0]:.6g}')
	print(f'  b = {popt[1]:.6g} ± {perr[1]:.6g}')
	print(f'  C = {popt[2]:.6g} ± {perr[2]:.6g}')
	print("Chi-squared:", chi2)
	
	# save plot in the same folder
	outpath = os.path.join(base_dir, 'RC_exp_fit.png')
	plot_and_save(t, y, st, sy, popt, perr, outpath)

if __name__ == '__main__':
	main()


