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
	data_path = os.path.join(base_dir, 'data_common.txt') # Data NO probe
	if not os.path.exists(data_path):
		raise FileNotFoundError(f'Data file not found: {data_path}')

	# Reading data from file
	# Data style: time (t), voltage (y), time scale (t_scale), voltage scale (y_scale)
	x, y, x_scale, y_scale = helper.read_data_xy_scalexy(data_path)

	# Searching for possibile reading errors
	if x.size == 0: 
		raise RuntimeError('No valid data read from file.')

	sx = np.sqrt(pow(x_scale/(10*np.sqrt(3)), 2) + pow(x*3/(np.sqrt(3)*100), 2)) # 1/10 reading error with max uniform distribution + 3% scale error with max uniform distribution
	sy = np.sqrt(pow(y_scale/(10*np.sqrt(3)), 2) + pow(y*3/(np.sqrt(3)*100), 2)) # 1/10 reading error with max uniform distribution + 3% scale error with max uniform distribution

	# Searching for initial parameters  
	m = 4.8
	q = 0
	p0 = [m, q]

	popt, perr, x_residual, y_residual, chi2 = helper.fit_linear(x, y, x_error=sx, y_error=sy, init0=p0)

	# Computing residuals
	
	# calculate the residuals error by quadratic sum using the variance theorem
	y_residual_err = np.sqrt(pow(sy, 2) + pow(perr[0]*x, 2) + pow(popt[0]*sx, 2) + pow(perr[1], 2))

	# Computing the weighted mean of the residuals
	weighted_mean_y_residual = np.average(y_residual, weights=1/y_residual_err**2)
	weighted_mean_y_residual_std = np.sqrt(1 / np.sum(1/y_residual_err**2))

	# computing compatibility between weighted mean of residuals and 0
	r_residual = np.abs(weighted_mean_y_residual)/weighted_mean_y_residual_std

	# Print fit results
	print('Fit parameters (m, q):')
	print(f'  m = {popt[0]:.6g} ± {perr[0]:.6g}')
	print(f'  q = {popt[1]:.6g} ± {perr[1]:.6g}')
	print("Chi-squared:", chi2)

	# Computing Tau
	tau = -1 / popt[1]
	tau_err = perr[1] / (popt[1]**2) 
	
	# save plot in the same folder
	outpath = os.path.join(base_dir, 'amp_differenze.png')

	# Taking points for the fit line
	t_fine = np.linspace(np.min(x), np.max(x), 400)
	y_fit = helper.linear_model(t_fine, *popt)

    # Plotting
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(x, y, xerr=sx, yerr=sy, fmt='o', label='Datas', color='black', ms = 3, lw = 1.6)
	ax[0].plot(t_fine, y_fit, label='Linear fit', color='red', lw = 1.2)
	ax[0].set_ylabel(r'$V_{\text{out}} \, (V)$')
	ax[0].legend()
	ax[0].set_title('Linear fit - VTC')
	#ax[0].text(0, 0.35, r'$\tau_{{\,\text{{BNC}}}}$ = {e:.0f} $\pm$ {f:.0f} $\mu s$'.format(e=tau*1e6, f = tau_err*1e6), size=12)

	ax[1].errorbar(x, y_residual, yerr=y_residual_err, fmt='o', label='Residuals BNC', color='black', ms = 3, lw = 1.6)
	ax[1].axhline(0, color='gray', linestyle='--', lw = 1.5)
	ax[1].set_xlabel(r'$V_{\text{in}} \, (V)$')
	ax[1].set_ylabel('Residuals (V)')
	ax[1].legend(ncol=2)

	plt.tight_layout()
	plt.savefig(outpath, dpi=150)
	print(f'Graph saved to: {outpath}')
	plt.show()

if __name__ == '__main__':
	main()
