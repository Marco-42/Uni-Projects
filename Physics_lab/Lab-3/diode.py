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

	# LINEAR FIT VD - ID 

	# resolve the relative path to the data file in the same folder as the script
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(base_dir, 'data_diode.txt') # Data NO probe
	if not os.path.exists(data_path):
		raise FileNotFoundError(f'Data file not found: {data_path}')

	# Reading data from file
	# Data style: Voltage (mV), current (muA), voltage_scale (x_scale), current scale (y_scale)
	x, y, x_scale, y_scale = helper.read_data_xy_scalexy(data_path)

	# Searching for possibile reading errors
	if x.size == 0: 
		raise RuntimeError('No valid data read from file.')

	print("VNC data: ")
	print("Vin - Vin error - Vout - Vout error")
	for i in range (len(x)):
		print(f"{helper.format_value_error(x[i], sx[i])}\t{helper.format_value_error(y[i], sy[i])}")

	# Searching for initial parameters  
	m = 4.8
	q = 0
	p0 = [m, q]

	popt, perr, x_residual, y_residual, chi2 = helper.fit_linear(x[:not_saturated], y[:not_saturated], x_error=sx[:not_saturated], y_error=sy[:not_saturated], init0=p0)

	# Computing residuals
	
	# calculate the residuals error by quadratic sum using the variance theorem
	y_residual_err = np.sqrt(pow(sy[:not_saturated], 2) + pow(perr[0]*x[:not_saturated], 2) + pow(popt[0]*sx[:not_saturated], 2) + pow(perr[1], 2))

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

	# Computing saturation input voltage
	saturation_input_voltage = [(saturated_value[0] - popt[1]) / popt[0], np.sqrt( (saturated_value[1]/popt[0])**2 + ((saturated_value[0] - popt[1]) * perr[0]/(popt[0]**2))**2 + (perr[1]/popt[0])**2 )]
	A_from_saturation = [saturated_value[0] / saturation_input_voltage[0], np.sqrt( (saturated_value[1]/saturation_input_voltage[0])**2 + (saturated_value[0] * saturation_input_voltage[1]/(saturation_input_voltage[0]**2))**2 )]

	# save plot in the same folder
	outpath = os.path.join(base_dir, 'A1.png')

	# Taking points for the fit line
	x_fine = np.linspace(np.min(x), np.max(x), 400)
	y_fit = helper.linear_model(x_fine, *popt)
	
    # Plotting
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(x, y, xerr=sx, yerr=sy, fmt='o', label='Data', color='black', ms = 3, lw = 1.6, zorder=3)
	ax[0].plot(x_fine, y_fit, label='Linear fit', color='firebrick', lw = 1.5)
	ax[0].axhline(saturated_value[0], color='darkorange', lw=1.5, label='Saturated voltage', linestyle='--')
	ax[0].set_ylabel(r'$V_{\text{out}} \, (V)$')
	ax[0].text(0.2, 25, r'$V_{{\text{{ sat}}}}$ = {a:.1f} $\pm$ {b:.1f} V'.format(a=saturated_value[0], b=saturated_value[1]), size=12)
	ax[0].text(0.2, 22.5, r'$|A|_{{\text{{ sat}}}}$ = {a:.1f} $\pm$ {b:.1f}'.format(a=A_from_saturation[0], b=A_from_saturation[1]), size=12)
	ax[0].text(0.2, 20, r'$|A|_{{\text{{ fit}}}}$ = {c:.2f} $\pm$ {d:.2f}'.format(c=popt[0], d=perr[0]), size=12)
	ax[0].text(0.2, 17.5, r'$q$ = {c:.2f} $\pm$ {d:.2f} V'.format(c=popt[1], d=perr[1]), size=12)
	ax[0].text(0.2, 15, r'$t_{{\text{{ q - 0}}}}$ = {c:.1f}'.format(c=helper.compatibility(popt[1], perr[1], 0, 0)), size=12)
	ax[0].text(0.2, 12.5, r'$\chi^2 \, / \, DOF$ = {c:.1f} / {d:.0f}'.format(c=chi2, d=not_saturated-2), size=12)
	ax[0].text(4.5, 12.5, r'$A_{{\text{{ exp}}}}$ = {a:.2f} $\pm$ {b:.2f}'.format(a=A_exp[0], b=A_exp[1]), size=12)
	ax[0].legend(loc='lower right')
	ax[0].set_title('Linear fit - VTC')
	ax[0].set_xlim(np.min(x)-0.4, np.max(x)+0.5)
	ax[0].set_ylim(-1, 33)
	#ax[0].text(0, 0.35, r'$\tau_{{\,\text{{BNC}}}}$ = {e:.0f} $\pm$ {f:.0f} $\mu s$'.format(e=tau*1e6, f = tau_err*1e6), size=12)

	ax[1].errorbar(x[:not_saturated], y_residual, yerr=y_residual_err, fmt='o', label='Residuals', color='black', ms = 3, lw = 1.6)
	ax[1].axhline(0, color='gray', linestyle='--', lw = 1.5, label = "Zero")
	ax[1].set_xlabel(r'$V_{\text{in}} \, (V)$')
	ax[1].set_ylabel('Residuals (V)')
	ax[1].legend(ncol=2, loc='upper left')
	ax[1].set_ylim(-1.4, 1.3)
	plt.tight_layout()
	plt.savefig(os.path.join(base_dir, 'VTC.png'), dpi=150)
	print(f'Graph saved to: {outpath}')
	
    plt.show()