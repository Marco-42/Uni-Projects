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
Vin = [-1.02, helper.voltage_error(1.02, 0.2)]    # Input voltage with scale (V) --> remember the sign
Rin = [55.646e3, helper.resistance_error(55.646e3, 100e3)]  # Input resistance with scale (Ohm)
C = [243e-12, helper.capacitance_error(243e-12, 1e-9)]  # Feedback capacitance with scale (F)
C_back = [22e-12, helper.capacitance_error(22e-12, 1e-9)]  # Background capacitance with scale (F)

# Computing Cf without background
Cf = [C[0] - C_back[0], np.sqrt(C[1]**2 + C_back[1]**2)]

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
	data_path = os.path.join(base_dir, 'data_lin_total.txt') # Data NO probe
	if not os.path.exists(data_path):
		raise FileNotFoundError(f'Data file not found: {data_path}')

	# Reading data from file
	# Data style: time (t), voltage (V), time scale (t_scale), voltage scale (V_scale)
	t, V, t_scale, V_scale = helper.read_data_xy_scalexy(data_path)

	# Convert time
	t = t/helper.TO_MU  # Convert time to seconds

	# Searching for possibile reading errors
	if t.size == 0: 
		raise RuntimeError('No valid data read from file.')

	# Computing errors
	st = helper.time_error(t_scale)/helper.TO_MU  # Time error
	sV = helper.voltage_error(V, V_scale)  # Voltage error

	# Computing charge and its error
	Q = -Vin[0]*t / Rin[0]  # Total input charge
	#sQ = np.sqrt(pow(Vin[1]*t/Rin[0], 2) + pow(st*Vin[0]/Rin[0], 2) + pow(Rin[1]*Vin[0]*t/Rin[0]**2, 2)) --> WITH TIME ERROR
	sQ = np.sqrt(pow(Vin[1]*t/Rin[0], 2) + pow(Rin[1]*Vin[0]*t/Rin[0]**2, 2))  # WITHOUT TIME ERROR

	# Searching for initial parameters
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

	# Computing Cfit - format [value, error]
	Cfit = [1 / popt[0], perr[0] / (popt[0]**2)]

	print(f'Computed Cf from fit: {Cfit[0]*helper.TO_P:.4f} ± {Cfit[1]*helper.TO_P:.4f} pF')

	# Computing compatibility between Cfit and Cf
	r_Cf = helper.compatibility(Cfit[0], Cfit[1], Cf[0], Cf[1])

	print(f'Compatibility between Cf from fit and Cf from datasheet: r = {r_Cf:.4f}')

	# save plot in the same folder
	outpath = os.path.join(base_dir, 'preamp.png')

	# Taking points for the fit line
	Q_fine = np.linspace(np.min(Q), np.max(Q), 400)
	y_fit = helper.linear_model(Q_fine, *popt)

    # Plotting
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(Q*helper.TO_N, V, xerr=sQ*helper.TO_N, yerr=sV, fmt='o', label='Data', color='black', ms = 3, lw = 1.6)
	ax[0].plot(Q_fine*helper.TO_N, y_fit, label='Linear fit', color='dodgerblue', lw = 1.2)
	ax[0].set_ylabel(r'$V_{\text{out}} \, (V)$')
	ax[0].legend(loc='lower right')
	ax[0].set_title('Linear fit - Preamp Response')
	ax[0].text(0.13, 1.2, r'$m_{{\,\text{{fit}}}}$ = {e:.1f} $\pm$ {f:.1f} $pC^{{\text{{-1}}}}$'.format(e=popt[0]/helper.TO_N, f = perr[0]/helper.TO_N), size=12)
	ax[0].text(0.13, 1, r'$q_{{\,\text{{fit}}}}$ = {e:.3f} $\pm$ {f:.3f} V'.format(e=popt[1], f = perr[1]), size=12)
	# ax[0].text(0.13, 0.8, r'$r_{{\,\text{{Cf}}}}$ = {e:.2f}'.format(e=r_Cf), size=12)
	ax[0].text(0.13, 0.78, r'$\chi^2 \, / \, DOF$ = {e:.1f} / {f:.0f}'.format(e=chi2, f=V_residual.size-2), size=12)

	ax[1].errorbar(Q*helper.TO_N, V_residual, yerr=V_residual_err, fmt='o', label='Residuals', color='black', ms = 3, lw = 1.6)
	ax[1].axhline(0, color='gray', linestyle='--', lw = 1.5)
	ax[1].set_xlabel(r'$Q_{\text{ in}} \, (nC)$')
	ax[1].set_ylabel('Residuals (V)')
	ax[1].legend(loc = 'upper left')
	#ax[1].set_ylim(-0.35, 0.5)
	plt.tight_layout()
	plt.savefig(outpath, dpi=150)
	print(f'Graph saved to: {outpath}')
	plt.show()

if __name__ == '__main__':
	main()
