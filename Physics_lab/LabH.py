# LABD HELPER FUNCTIONS
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

# ============= FUNCTIONS MODELS =============
# Exponential model function for odr
def odr_exp(beta, x):
	"""beta[0] * exp(beta[1] * x) + beta[2]"""
	return beta[0] * np.exp(beta[1] * x) + beta[2]

# Linear model function for odr
def odr_linear(beta, x):
	"""beta[0] * x + beta[1]"""
	return beta[0] * x + beta[1]

# Exponential model function for curve_fit
def exp_model(x, A, b, C):
    return A * np.exp(b * x) + C

# Linear model function for curve_fit
def linear_model(x, m, q):
	return m * x + q

# ============= FIT FUNCTIONS ================
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
	x_residual = out.delta
	y_residual = out.delta

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
	x_residual = out.delta
	y_residual = out.delta

	# Return optimized parameters and their uncertainties
	return params, params_err, x_residual, y_residual, chi2
