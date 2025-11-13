# LAB HELPER FUNCTIONS
# This file contains helper functions for data reading and fitting in Physics Lab experiments.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
import mplhep as hep
from cycler import cycler
import statistics
from scipy.odr import *

# ============= GENERAL CONSTANTS ============

TO_K = 1e3  # Conversion factor to kilo units
TO_MU = 1e6  # Conversion factor to micro units
TO_N = 1e9  # Conversion factor to nano units
TO_P = 1e12  # Conversion factor to pico units

# ============= READ DATA FUNCTIONS ==========
# Functions to adapt a file to a common format and read data

# Data style x - y - x_scale - y_scale
def read_data_xy_scalexy(path):
	"""Read the file (path in input) and return four arrays: x (float), y (float), x_scale (float), and y_scale (float).

	The parser ignores lines starting with '#', replaces commas with
	decimal points, and takes the first four numeric columns found in the line.
	"""

	COLUMN = 4  # Number of columns in the data file

	x = []
	y = []
	x_scale = []
	y_scale = []

	with open(path, 'r', encoding='utf-8') as f:

		# File adapted to a common format
		for line in f:  # Take all the different lines
			line = line.strip()
			if not line or line.startswith('#'):  # Ignore empty lines and comments
				continue
			# normalize decimal comma if present
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
				if len(nums) >= COLUMN:
					break
			if len(nums) >= COLUMN:
				x.append(nums[0])
				y.append(nums[1])
				x_scale.append(nums[2])
				y_scale.append(nums[3])

	# Convert to numpy arrays and return
	return np.array(x, dtype=float), np.array(y, dtype=float), np.array(x_scale, dtype=float), np.array(y_scale, dtype=float)

# Data style x - y - z - z_error
def read_data_xyz_errorz(path):
	"""Read the file (path in input) and return four arrays: x (float), y (float), z (float), and z_error (float).

	The parser ignores lines starting with '#', replaces commas with
	decimal points, and takes the first four numeric columns found in the line.
	"""

	COLUMN = 4  # Number of columns in the data file

	x = []
	y = []
	z = []
	z_error = []

	with open(path, 'r', encoding='utf-8') as f:

		# File adapted to a common format
		for line in f:  # Take all the different lines
			line = line.strip()
			if not line or line.startswith('#'):  # Ignore empty lines and comments
				continue
			# normalize decimal comma if present
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
				if len(nums) >= COLUMN:
					break
			if len(nums) >= COLUMN:
				x.append(nums[0])
				y.append(nums[1])
				z.append(nums[2])
				z_error.append(nums[3])

	# Convert to numpy arrays and return
	return np.array(x, dtype=float), np.array(y, dtype=float), np.array(z, dtype=float), np.array(z_error, dtype=float)

# Data style x - y - sy
def read_data_xy_errory(path):
	"""Read the file (path in input) and return three arrays: x (float), y (float), and y_error (float).

	The parser ignores lines starting with '#', replaces commas with
	decimal points, and takes the first three numeric columns found in the line.
	"""

	COLUMN = 3  # Number of columns in the data file

	x = []
	y = []
	y_error = []

	with open(path, 'r', encoding='utf-8') as f:

		# File adapted to a common format
		for line in f:  # Take all the different lines
			line = line.strip()
			if not line or line.startswith('#'):  # Ignore empty lines and comments
				continue
			# normalize decimal comma if present
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
				if len(nums) >= COLUMN:
					break
			if len(nums) >= COLUMN:
				x.append(nums[0])
				y.append(nums[1])
				y_error.append(nums[2])

	# Convert to numpy arrays and return
	return np.array(x, dtype=float), np.array(y, dtype=float), np.array(y_error, dtype=float)

# Data style x - y
def read_data_xy(path):
	"""Read the file (path in input) and return two arrays: x (float), y (float).

	The parser ignores lines starting with '#', replaces commas with
	decimal points, and takes the first two numeric columns found in the line.
	"""

	COLUMN = 2  # Number of columns in the data file

	x = []
	y = []

	with open(path, 'r', encoding='utf-8') as f:

		# File adapted to a common format
		for line in f:  # Take all the different lines
			line = line.strip()
			if not line or line.startswith('#'):  # Ignore empty lines and comments
				continue
			# normalize decimal comma if present
			line = line.replace(',', '.')
			line = line.replace('(', '')
			line = line.replace(')', '')
			line = line.replace('dB', '')
			# split on whitespace
			parts = line.split()

			# Extracting data from adapted file
			nums = []
			for p in parts:
				try:
					nums.append(float(p))
				except ValueError:
					continue
				if len(nums) >= COLUMN:
					break
			if len(nums) >= COLUMN:
				x.append(nums[0])
				y.append(nums[1])

	# Convert to numpy arrays and return
	return np.array(x, dtype=float), np.array(y, dtype=float)

	COLUMN = 3  # Number of columns in the data file

	x = []
	y = []
	y_error = []

	with open(path, 'r', encoding='utf-8') as f:

		# File adapted to a common format
		for line in f:  # Take all the different lines
			line = line.strip()
			if not line or line.startswith('#'):  # Ignore empty lines and comments
				continue
			# normalize decimal comma if present
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
				if len(nums) >= COLUMN:
					break
			if len(nums) >= COLUMN:
				x.append(nums[0])
				y.append(nums[1])
				y_error.append(nums[2])

	# Convert to numpy arrays and return
	return np.array(x, dtype=float), np.array(y, dtype=float), np.array(y_error, dtype=float)

# ============= FUNCTIONS MODELS =============
# Model functions for fitting

# Exponential model function for odr
def odr_exp(beta, x):
	"""beta[0] * exp(beta[1] * x) + beta[2]"""
	return beta[0] * np.exp(beta[1] * x) + beta[2]

# Cos^2 fit function for odr
def odr_cos2(beta, x):
	"""beta[0] * cos^2(x)"""
	return beta[0] * (np.cos(x*np.pi/180))**2

# Linear model function for odr
def odr_linear(beta, x):
	"""beta[0] * x + beta[1]"""
	return beta[0] * x + beta[1]

def odr_shaper_module(beta, f):
	"""Module of the shaper characteristic function with different tau - beta[0] = tau1, beta[1] = tau2"""
	w = 2 * np.pi * f
	tau1 = beta[0]
	tau2 = beta[1]
	return tau1 * w / np.sqrt(((tau2**2)*w**2 + 1)*((tau1**2)*w**2 + 1))

def odr_shaper_module_semplified(beta, f):
	"""Module of the shaper characteristic function with same tau - beta[0] = tau, beta[1] = offset"""
	w = 2 * np.pi * f
	tau = beta[0]
	return beta[1]*tau * w / ((tau**2)*w**2 + 1)

def odr_inverting_oamp_module(beta, f):
	"""Module of the inverting oamp characteristic function """
	w = 2 * np.pi * f
	A = beta[0]
	return A / np.sqrt(1 + (w/(2*np.pi*beta[1]))**2)

# Exponential model function for curve_fit
def exp_model(x, A, b, C):
	"""A * exp(b * x) + C"""
	return A * np.exp(b * x) + C

# Linear model function for curve_fit
def linear_model(x, m, q):
	"""m * x + q"""
	return m * x + q

# Model function for the module of the shaper characteristic function
def shaper_module(f, tau1, tau2 = 0, offset = 0, semplified=False):
	"""Module of the shaper characteristic function with different tau if semplified is False, else with same tau
	The offset parameter is moltiplicative and used only in the semplified case"""
	w = 2 * np.pi * f
	if semplified:
		return offset * tau1 * w / ((tau1**2)*w**2 + 1)
	else: 
		return tau1 * w / np.sqrt(((tau2**2)*w**2 + 1)*((tau1**2)*w**2 + 1))

def inverting_oamp_module(f, A, f_t):
	"""Module of the inverting oamp characteristic function """
	w = 2 * np.pi * f
	return A / np.sqrt(1 + (w/(2 * np.pi * f_t))**2)

# Model function for cos^2
def cos2_model(x, A):
	"""A * cos^2(x)"""
	return A * (np.cos(x*np.pi/180))**2

# ============= FIT FUNCTIONS ================
# Functions to perform fits using ODR (errors on both axes)

# Exponential fit function
def fit_exponential(x, y, x_error=None, y_error=None, init0=None):
	"""Exponential fit using ODR (errors on both axes) -  x, y, x_error, y_error, init0.

	x, y, x_error, y_error can be numpy arrays, init0 can be a list of initial parameters. 
	Fitting result are four vectors and a chi2 value:
	the first one contains the optimized parameters, while the second one contains
	their uncertainties, the last two contain x and y residuals.
	"""
	# reorder for increasing x for cleaner plot
	order = np.argsort(x)
	x_sorted = x[order]
	y_sorted = y[order]
	if x_error is not None:
		x_error = np.asarray(x_error)[order]
	if y_error is not None:
		y_error = np.asarray(y_error)[order]

	# Fitting datas with an exponential function using ODR (Orthogonal Distance Regression)
	# Create a model for fitting - Function the fitting method is able to read
	function_model = Model(odr_exp)

	# Create a RealData object using our initiated data from above
	graph_data = RealData(x_sorted, y_sorted, sx=x_error, sy=y_error)

	# Set up ODR with the model and data - ODR is the fitting method used to consider the errors both in x and y
	# beta0 are the input parameters
	odr = ODR(graph_data, function_model, beta0=init0)

	# Start the fitting method
	out = odr.run()

	# Get chi square from fit
	chi2 = out.sum_square

	# get parameters and parameters errors from fit 
	params = out.beta
	params_err = out.sd_beta

	# Get residuals (orthogonal distances returned by ODR)
	y_residual = out.eps
	x_residual = np.sqrt(pow(out.delta, 2) + pow(out.eps, 2))

	# Return optimized parameters and their uncertainties
	return params, params_err, x_residual, y_residual, chi2

# Linear fit function
def fit_linear(x, y, x_error=None, y_error=None, init0=None):
	"""Linear fit using ODR (errors on both axes) -  x, y, x_error, y_error, init0.

	x, y, x_error, y_error can be numpy arrays, init0 can be a list of initial parameters. 
	Fitting result are four vectors and a chi2 value:
	the first one contains the optimized parameters, while the second one contains
	their uncertainties, the last two contain x and y residuals.
	"""
	# reorder for increasing x for cleaner plot
	order = np.argsort(x)
	x_sorted = x[order]
	y_sorted = y[order]
	if x_error is not None:
		x_error = np.asarray(x_error)[order]
	if y_error is not None:
		y_error = np.asarray(y_error)[order]

	# Fitting datas with an exponential function using ODR (Orthogonal Distance Regression)
	# Create a model for fitting - Function the fitting method is able to read
	function_model = Model(odr_linear)

	# Create a RealData object using our initiated data from above
	graph_data = RealData(x_sorted, y_sorted, sx=x_error, sy=y_error)

	# Set up ODR with the model and data - ODR is the fitting method used to consider the errors both in x and y
	# beta0 are the input parameters
	odr = ODR(graph_data, function_model, beta0=init0)

	# Start the fitting method
	out = odr.run()

	# Get chi square from fit
	chi2 = out.sum_square

	# get parameters and parameters errors from fit 
	params = out.beta
	params_err = out.sd_beta

	# Get residuals (orthogonal distances returned by ODR)
	y_residual = out.eps
	x_residual = np.sqrt(pow(out.delta, 2) + pow(out.eps, 2))

	# Return optimized parameters and their uncertainties
	return params, params_err, x_residual, y_residual, chi2

def fit_shaper_module(f, H, H_error=None, init0=None, semplified=False):
	"""Fit using ODR of the shaper characteristics function module(errors on y axis) -  freq, |H|, |H|_error, init0.

	f, H, H_error can be numpy arrays, init0 can be a list of initial parameters(tau1 and tau2). 
	Fitting result are four vectors and a chi2 value:
	the first one contains the optimized parameters, while the second one contains
	their uncertainties, the last two contain x and y residuals.

	semplified: boolean to choose if fit with same tau(true) or different tau(false)
	"""
	# reorder for increasing x for cleaner plot
	order = np.argsort(f)
	x_sorted = f[order]
	y_sorted = H[order]
	if H_error is not None:
		H_error = np.asarray(H_error)[order]

	# Fitting datas with an exponential function using ODR (Orthogonal Distance Regression)
	# Create a model for fitting - Function the fitting method is able to read
	if semplified and len(init0) == 2:
		function_model = Model(odr_shaper_module_semplified)
	elif not semplified and len(init0) == 2:
		function_model = Model(odr_shaper_module)

	# Create a RealData object using our initiated data from above
	graph_data = RealData(x_sorted, y_sorted, sy=H_error)

	# Set up ODR with the model and data - ODR is the fitting method used to consider the errors both in x and y
	# beta0 are the input parameters
	odr = ODR(graph_data, function_model, beta0=init0)

	# Start the fitting method
	out = odr.run()

	# Get chi square from fit
	chi2 = out.sum_square

	# get parameters and parameters errors from fit 
	params = out.beta
	params_err = out.sd_beta

	# Get residuals (orthogonal distances returned by ODR)
	y_residual = out.eps
	x_residual = np.sqrt(pow(out.delta, 2) + pow(out.eps, 2))

	# Return optimized parameters and their uncertainties
	return params, params_err, x_residual, y_residual, chi2

def fit_inverting_oamp_module(f, H, H_error=None, init0=None):
	"""Fit using ODR of the inverting OAMP characteristics function module(errors on y axis) -  freq, |H|, |H|_error, init0.

	f, H, H_error can be numpy arrays, init0 can be a list of initial parameters(A and f_t). 
	Fitting result are four vectors and a chi2 value:
	the first one contains the optimized parameters, while the second one contains
	their uncertainties, the last two contain x and y residuals.
	"""
	# reorder for increasing x for cleaner plot
	order = np.argsort(f)
	x_sorted = f[order]
	y_sorted = H[order]
	if H_error is not None:
		H_error = np.asarray(H_error)[order]

	# Fitting datas with an exponential function using ODR (Orthogonal Distance Regression)
	# Create a model for fitting - Function the fitting method is able to read
	function_model = Model(odr_inverting_oamp_module)

	# Create a RealData object using our initiated data from above
	graph_data = RealData(x_sorted, y_sorted, sy=H_error)

	# Set up ODR with the model and data - ODR is the fitting method used to consider the errors both in x and y
	# beta0 are the input parameters
	odr = ODR(graph_data, function_model, beta0=init0)

	# Start the fitting method
	out = odr.run()

	# Get chi square from fit
	chi2 = out.sum_square

	# get parameters and parameters errors from fit 
	params = out.beta
	params_err = out.sd_beta

	# Get residuals (orthogonal distances returned by ODR)
	y_residual = out.eps
	x_residual = np.sqrt(pow(out.delta, 2) + pow(out.eps, 2))

	# Return optimized parameters and their uncertainties
	return params, params_err, x_residual, y_residual, chi2

def fit_cos2(x, y, y_error=None, init0=None):
	"""Cos^2 fit using ODR (errors on y axis) -  x, y, y_error, init0.

	x, y, y_error can be numpy arrays, init0 can be a list of initial parameters. 
	Fitting result are four vectors and a chi2 value:
	the first one contains the optimized parameters, while the second one contains
	their uncertainties, the last two contain x and y residuals.
	"""
	# reorder for increasing x for cleaner plot
	order = np.argsort(x)
	x_sorted = x[order]
	y_sorted = y[order]
	if y_error is not None:
		y_error = np.asarray(y_error)[order]

	# Fitting datas with an exponential function using ODR (Orthogonal Distance Regression)
	# Create a model for fitting - Function the fitting method is able to read
	function_model = Model(odr_cos2)

	# Create a RealData object using our initiated data from above
	graph_data = RealData(x_sorted, y_sorted, sy=y_error)

	# Set up ODR with the model and data - ODR is the fitting method used to consider the errors both in x and y
	# beta0 are the input parameters
	odr = ODR(graph_data, function_model, beta0=init0)

	# Start the fitting method
	out = odr.run()

	# Get chi square from fit
	chi2 = out.sum_square

	# get parameters and parameters errors from fit 
	params = out.beta
	params_err = out.sd_beta

	# Get residuals (orthogonal distances returned by ODR)
	y_residual = out.eps
	x_residual = np.sqrt(pow(out.delta, 2) + pow(out.eps, 2))

	# Return optimized parameters and their uncertainties
	return params, params_err, x_residual, y_residual, chi2

# ============= ERROR FUNCTIONS ==============
# Functions to compute errors

# Voltage error function
def voltage_error(V, V_scale):
	"""Compute voltage error given voltage V and voltage scale V_scale."""
	return np.sqrt(pow(V_scale/(10*np.sqrt(3)), 2) + pow(V*3/(np.sqrt(3)*100), 2)) # 1/10 reading error with max uniform distribution + 3% scale error with max uniform distribution

# Time error function
def time_error(t_scale):
	"""Compute time error given time scale t_scale."""
	return t_scale * 2/(5*np.sqrt(24)) # Triangular distribution applied considering max error

# Resistance error function
def resistance_error(R, R_scale): # TODO
	"""Compute resistance error given resistance R."""
	if R_scale == 100e3 or R_scale == 10e3:
		return np.sqrt(pow(0.07*R/50, 2)/12 + pow(8*20, 2)/12)
	
# Capacitance error function
def capacitance_error(C, C_scale): # TODO
	"""Compute capacitance error given capacitance C."""
	if C_scale == 1e-9:
		return np.sqrt(pow(2.5*C/50, 2)/12 + pow(30*1e-12, 2)/12)

# ============= OTHER HELPERS ================

def compatibility(value1, error1, value2, error2):
	"""Compute compatibility between two values with their errors. - value1, error1, value2, error2."""
	diff = np.abs(value1 - value2)
	total_error = np.sqrt(error1**2 + error2**2)
	if total_error == 0:
		return np.inf  # Avoid division by zero
	return diff / total_error

def chi_squared(observed, expected, errors):
	"""Compute chi-square statistic - observed, expected, total errors."""
	return np.sum(((observed - expected) / errors) ** 2)
