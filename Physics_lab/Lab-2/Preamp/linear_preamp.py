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

# Variable to set
TO_MU = 1e6  # Conversion factor to micro units
Vin = [1.0, ...]    # Input voltage with scale (V) --> remember the sign
Rin = [5e3, ...]  # Input resistance with scale (Ohm)
Cf = [1e-9, ...]  # Feedback capacitance with scale (F)

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
	data_path = os.path.join(base_dir, 'data_preamp.txt') # Data NO probe
	if not os.path.exists(data_path):
		raise FileNotFoundError(f'Data file not found: {data_path}')

	# Reading data from file
	# Data style: time (t), voltage (V), time scale (t_scale), voltage scale (V_scale)
	t, V, t_scale, V_scale = helper.read_data_xy_scalexy(data_path)

	# Searching for possibile reading errors
	if t.size == 0: 
		raise RuntimeError('No valid data read from file.')

	# Computing errors
	st = TO_MU * helper.time_error(t_scale)  # Time error
	sV = helper.voltage_error(V, V_scale)  # Voltage error

	# Computing charge and its error
	Q = Vin[0]*t / Rin[0]  # Total input charge
	sQ = np.sqrt(pow(helper.voltage_error(Vin[0], Vin[1])*t/Rin[0], 2) + pow(st*Vin[0]/Rin[0], 2) + pow(helper.resistance_error(Rin[0], Rin[1])*Vin[0]*t/Rin[0]**2, 2))

	# Searching for initial parameters TODO  
	m = -1/Cf[0]
	q = 0
	p0 = [m, q]

	popt, perr, Q_residual, V_residual, chi2 = helper.fit_linear(Q, V, x_error=sQ, y_error=sV, init0=p0)

	# Computing residuals
	
	# calculate the residuals error by quadratic sum using the variance theorem
	V_residual_err = np.sqrt(pow(sV, 2) + pow(perr[0]*Q, 2) + pow(popt[0]*sQ, 2) + pow(perr[1], 2))

	# Computing the weighted mean of the residuals
	weighted_mean_V_residual = np.average(V_residual, weights=1/V_residual_err**2)
	weighted_mean_V_residual_std = np.sqrt(1 / np.sum(1/V_residual_err**2))

	# computing compatibility between weighted mean of residuals and 0
	r_residual = np.abs(weighted_mean_V_residual)/weighted_mean_V_residual_std

	# Print fit results
	print('Fit parameters (m, q):')
	print(f'  m = {popt[0]:.6g} ± {perr[0]:.6g}')
	print(f'  q = {popt[1]:.6g} ± {perr[1]:.6g}')
	print("Chi-squared:", chi2)

	# Computing Tau
	tau = -1 / popt[1]
	tau_err = perr[1] / (popt[1]**2) 
	
	# save plot in the same folder
	outpath = os.path.join(base_dir, 'preamp.png')

	# Taking points for the fit line
	Q_fine = np.linspace(np.min(Q), np.max(Q), 400)
	y_fit = helper.linear_model(Q_fine, *popt)

    # Plotting
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(Q, V, xerr=sQ, yerr=sV, fmt='o', label='Datas', color='black', ms = 3, lw = 1.6)
	ax[0].plot(Q_fine, y_fit, label='Linear fit', color='red', lw = 1.2)
	ax[0].set_ylabel(r'$V_{\text{out}} \, (V)$')
	ax[0].legend(loc='lower right')
	ax[0].set_title('Linear fit - Preamp Response')
	#ax[0].text(0, 0.35, r'$\tau_{{\,\text{{BNC}}}}$ = {e:.0f} $\pm$ {f:.0f} $\mu s$'.format(e=tau*1e6, f = tau_err*1e6), size=12)

	ax[1].errorbar(Q, V_residual, yerr=V_residual_err, fmt='o', label='Residuals', color='black', ms = 3, lw = 1.6)
	ax[1].axhline(0, color='gray', linestyle='--', lw = 1.5)
	ax[1].set_xlabel(r'$Q_{\text{in}} \, (V)$')
	ax[1].set_ylabel('Residuals (V)')
	ax[1].legend(ncol=2)
	#ax[1].set_ylim(-0.35, 0.5)
	plt.tight_layout()
	plt.savefig(outpath, dpi=150)
	print(f'Graph saved to: {outpath}')
	plt.show()

if __name__ == '__main__':
	main()
