import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
import mplhep as hep
from cycler import cycler
import statistics
from scipy.odr import *

from Physics_lab import LabH as helper # Importing helper functions

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

#============= MAIN FUNCTION ================
def main():

	# resolve the relative path to the data file in the same folder as the script
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(base_dir, 'data_RCt_BNC.txt') # Data NO probe
	data_path_probe = os.path.join(base_dir, 'data_RCt_probe.txt') # Data WITH probe
	if not os.path.exists(data_path) or not os.path.exists(data_path_probe):
		raise FileNotFoundError(f'Data file not found: {data_path} or {data_path_probe}')

	# Reading data from file
	# Data style: time (t), voltage (y), time scale (t_scale), voltage scale (y_scale)
	t, y, t_scale, y_scale = helper.read_data_xy_scalexy(data_path)
	t_probe, y_probe, t_scale_probe, y_scale_probe = helper.read_data_xy_scalexy(data_path_probe)

	# Searching for possibile reading errors
	if t.size == 0 and t_probe.size == 0: 
		raise RuntimeError('No valid data read from file.')

	st = t_scale * 1e-6 * 2/(5*np.sqrt(24)) # Triangular distribution applied considering max error
	sy = np.sqrt(pow(y_scale/(10*np.sqrt(6)), 2) + pow(y*3/(np.sqrt(3)*100), 2)) # 1/10 reading error with max uniform distribution + 3% scale error with max uniform distribution
	sy_probe = np.sqrt(pow(y_scale_probe/(10*np.sqrt(6)), 2) + pow(y_probe*3/(np.sqrt(3)*100), 2)) # 1/10 reading error with max uniform distribution + 3% scale error with max uniform distribution
	t = t * 1e-6 # Convert microseconds to seconds
	t_probe = t_probe * 1e-6 # Convert microseconds to seconds

	# Searching for initial parameters
	# initial parameters input: A ~ (max-min), C ~ min, b ~ -1 / (range_t)
	A0 = (np.max(y))
	C0 = 0
	b0 = -1000
	p0 = [A0, b0, C0]

	popt, perr, x_residual, y_residual, chi2 = helper.fit_exponential(t, y, x_error=st, y_error=sy, init0=p0)
	popt_probe, perr_probe, x_residual_probe, y_residual_probe, chi2_probe = helper.fit_exponential(t_probe, y_probe, x_error=st, y_error=sy_probe, init0=p0)

	# Computing residuals
	
	# calculate the residuals error by quadratic sum using the variance theorem
	y_residual_err = np.sqrt(pow(sy, 2) + pow(perr[0]*np.exp(popt[1]*t), 2) + pow(popt[0]*t*np.exp(popt[1]*t)*perr[1], 2) + pow(perr[2], 2) + pow(popt[0]*popt[1]*np.exp(popt[1]*t)*st, 2))
	y_residual_probe_err = np.sqrt(pow(sy_probe, 2) + pow(perr_probe[0]*np.exp(popt_probe[1]*t_probe), 2) + pow(popt_probe[0]*t_probe*np.exp(popt_probe[1]*t_probe)*perr_probe[1], 2) + pow(perr_probe[2], 2) + pow(popt_probe[0]*popt_probe[1]*np.exp(popt_probe[1]*t_probe)*st, 2))

	# Computing the weighted mean of the residuals
	weighted_mean_y_residual = np.average(y_residual, weights=1/y_residual_err**2)
	weighted_mean_y_residual_std = np.sqrt(1 / np.sum(1/y_residual_err**2))
	weighted_mean_y_residual_probe = np.average(y_residual_probe, weights=1/y_residual_probe_err**2)
	weighted_mean_y_residual_std_probe = np.sqrt(1 / np.sum(1/y_residual_probe_err**2))

	# computing compatibility between weighted mean of residuals and 0
	r_residual = np.abs(weighted_mean_y_residual)/weighted_mean_y_residual_std
	r_residual_probe = np.abs(weighted_mean_y_residual_probe)/weighted_mean_y_residual_std_probe

	# Print fit results
	print('Fit parameters (A, b, C) - BNC:')
	print(f'  A = {popt[0]:.6g} ± {perr[0]:.6g}')
	print(f'  b = {popt[1]:.6g} ± {perr[1]:.6g}')
	print(f'  C = {popt[2]:.6g} ± {perr[2]:.6g}')
	print("Chi-squared:", chi2)

	print('Fit parameters (A, b, C) - Probe:')
	print(f'  A = {popt_probe[0]:.6g} ± {perr_probe[0]:.6g}')
	print(f'  b = {popt_probe[1]:.6g} ± {perr_probe[1]:.6g}')
	print(f'  C = {popt_probe[2]:.6g} ± {perr_probe[2]:.6g}')
	print("Chi-squared:", chi2_probe)
	
	# Computing Tau
	tau = -1 / popt[1]
	tau_err = perr[1] / (popt[1]**2) 
	tau_probe = -1 / popt_probe[1]
	tau_probe_err = perr_probe[1] / (popt_probe[1]**2)

	# Print Tau results
	print(f'Tau (BNC): {tau*1e6:.6g} ± {tau_err*1e6:.6g} µs')
	print(f'Tau (Probe): {tau_probe*1e6:.6g} ± {tau_probe_err*1e6:.6g} µs')

	# save plot in the same folder
	outpath = os.path.join(base_dir, 'RC_exp_fit.png')

	# Taking points for the fit line
	t_fine = np.linspace(np.min(t), np.max(t), 400)
	y_fit = helper.exp_model(t_fine, *popt)
	t_probe_fine = np.linspace(np.min(t_probe), np.max(t_probe), 400)
	y_fit_probe = helper.exp_model(t_probe_fine, *popt_probe)

    # Plotting
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(t * 1e6, y, xerr=st*1e6, yerr=sy, fmt='o', label='Datas BNC', color='black', ms = 3, lw = 1.6)
	ax[0].errorbar(t_probe * 1e6, y_probe, xerr=st*1e6, yerr=sy_probe, fmt='o', label='Datas Probe', color='dodgerblue', ms = 3, lw = 1.6)
	ax[0].plot(t_fine * 1e6, y_fit, label='Exponential fit BNC', color='red', lw = 1.2)
	ax[0].plot(t_probe_fine * 1e6, y_fit_probe, label='Exponential fit Probe', color='orange', lw = 1.2)
	ax[0].set_ylabel('Voltage (V)')
	ax[0].legend()
	ax[0].set_title('Exponential fit - RC data')
	ax[0].set_ylim(0.1, 1.6)
	ax[0].text(0, 0.35, r'$\tau_{{\,\text{{BNC}}}}$ = {e:.0f} $\pm$ {f:.0f} $\mu s$'.format(e=tau*1e6, f = tau_err*1e6), size=12)
	ax[0].text(0, 0.25, r'$\tau_{{\,\text{{Probe}}}}$ = {e:.0f} $\pm$ {f:.0f} $\mu s$'.format(e=tau_probe*1e6, f = tau_probe_err*1e6), size=12)

	ax[1].errorbar(t * 1e6 - 7, y_residual, yerr=y_residual_err, fmt='o', label='Residuals BNC', color='black', ms = 3, lw = 1.6)
	ax[1].errorbar(t_probe * 1e6 + 7, y_residual_probe, yerr=y_residual_probe_err, fmt='o', label='Residuals Probe', color='dodgerblue', ms = 3, lw = 1.6)
	ax[1].axhline(0, color='gray', linestyle='--', lw = 1.5)
	ax[1].set_xlabel(r'Time ($\mu$s)')
	ax[1].set_ylabel('Residuals (V)')
	ax[1].set_ylim(-0.08, 0.12)
	ax[1].legend(ncol=2)

	plt.tight_layout()
	plt.savefig(outpath, dpi=150)
	print(f'Graph saved to: {outpath}')
	plt.show()

if __name__ == '__main__':
	main()
