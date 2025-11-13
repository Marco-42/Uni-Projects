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

# Variable to set, each vector in the form [value, error]
Vin = [1, helper.voltage_error(1, 0.2)]    # Input voltage with scale (V) 
R1 = [5.54e3, helper.resistance_error(5.54e3, 10e3)]  # Input resistance with scale (Ohm)
Rf = [26.57e3, helper.resistance_error(26.57e3, 100e3)]  # Second resistance with scale (Ohm)
A_exp = [Rf[0]/R1[0], np.sqrt( (Rf[1]/R1[0])**2 + (R1[1]*Rf[0]/(R1[0]**2))**2 )]  # Expected amplification

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

	# LINEAR FIT VTC

	# resolve the relative path to the data file in the same folder as the script
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(base_dir, 'data_VNC.txt') # Data NO probe
	if not os.path.exists(data_path):
		raise FileNotFoundError(f'Data file not found: {data_path}')

	# Reading data from file
	# Data style: time (t), voltage (y), time scale (t_scale), voltage scale (y_scale)
	x, y, x_scale, y_scale = helper.read_data_xy_scalexy(data_path)

	not_saturated = len(x) - 3  # number of not saturated points
	saturated_value = [np.max(y), helper.voltage_error(np.max(y), y_scale[len(y)-1])]  # value of the last not saturated point

	# Searching for possibile reading errors
	if x.size == 0: 
		raise RuntimeError('No valid data read from file.')

	sx = np.sqrt(pow(x_scale/(10*np.sqrt(3)), 2) + pow(x*3/(np.sqrt(3)*100), 2)) # 1/10 reading error with max uniform distribution + 3% scale error with max uniform distribution
	sy = np.sqrt(pow(y_scale/(10*np.sqrt(3)), 2) + pow(y*3/(np.sqrt(3)*100), 2)) # 1/10 reading error with max uniform distribution + 3% scale error with max uniform distribution

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
	ax[0].plot(x_fine, y_fit, label='Linear fit', color='firebrick', lw = 1.2)
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
	plt.savefig(outpath, dpi=150)
	print(f'Graph saved to: {outpath}')
	
	# BODE FIT
	
	# Getting data for the shaper frequency response
	# resolve the relative path to the data file in the same folder as the script
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(base_dir, 'data_oamp_bode.txt') # Data NO probe
	if not os.path.exists(data_path):
		raise FileNotFoundError(f'Data file not found: {data_path}')

	# Reading data from file
	# Data style: frequency (f), voltage (V), voltage scale (V_scale)
	f, Vout, Vout_scale = helper.read_data_xy_errory(data_path)

	
	# Searching for possibile reading errors
	if f.size == 0: 
		raise RuntimeError('No valid data read from file.')

	# Converting frequency to Hz
	f = f * 1e3  # kHz to Hz

	# Computing errors
	sVout = helper.voltage_error(Vout, Vout_scale)  # Voltage error

	# Computing module H
	H = Vout / Vin[0]
	#sH = np.sqrt( (sVout / Vin[0])**2 + (Vin[1]*Vout / Vin[0]**2)**2 )
	sH = H*np.sqrt(pow(Vout_scale/(Vout*10*np.sqrt(3)), 2) + pow(0.2/(Vin[0]*10*np.sqrt(3)), 2)+ pow(3/(np.sqrt(3)*100), 2))
	
	# Computing amplification in db
	A = 20 * np.log10(H)
	sA = 20 * sH /(np.log10(10) * H)

	# FITTING FOR INVERTING OAMP

	# Searching for initial parameters
	p02 = [A_exp[0], 460*helper.TO_K]  # Initial parameters: A and f_t

	popt2, perr2, f_residual2, V_residual2, chi22 = helper.fit_inverting_oamp_module(f, H, H_error=sH, init0=p02)

	# Computing residuals
	
	# calculate the residuals error
	V_residual_err2 = sH

	# Computing the weighted mean of the residuals
	weighted_mean_V_residual2 = np.average(V_residual2, weights=1/V_residual_err2**2)
	weighted_mean_V_residual_std2 = np.sqrt(1 / np.sum(1/V_residual_err2**2))

	# computing compatibility between weighted mean of residuals and 0
	r_residual = np.abs(weighted_mean_V_residual2)/weighted_mean_V_residual_std2

	# Print fit results
	print("")
	print('Fit parameters (A_fit, f_t):')
	print(f'  A_fit = {popt2[0]:.6g} ± {perr2[0]:.6g}')
	print(f'  f_t = {popt2[1]/helper.TO_K:.6g} ± {perr2[1]/helper.TO_K:.6g}')
	print("Chi-squared:", chi22)

	# Taking points for the fit line
	f_fine = np.linspace(np.min(f), np.max(f), 2000)
	y_fit = helper.inverting_oamp_module(f_fine, *popt2)
	y_fit_1sigma = helper.inverting_oamp_module(f_fine, *(popt2 + perr2))
	y_fit_m1sigma = helper.inverting_oamp_module(f_fine, *(popt2 - perr2))

	# Linear fitting of the four last points of bode graph (fit su log10(f))
	p0_linear = [-20/helper.TO_K, 100]  # Initial parameters: m and q
	logf_last = np.log10(f[-3:])
	popt_linear, perr_linear, f_residual_linear, V_residual_linear, chi2_linear = helper.fit_linear(logf_last, A[-3:], x_error=None, y_error=sA[-3:], init0=p0_linear)

	# Taking points for the fit line su griglia logaritmica
	f_fine_linear = np.logspace(np.log10(250*helper.TO_K), np.log10(np.max(f[-3:])), 400)
	y_fit_linear = helper.linear_model(np.log10(f_fine_linear), *popt_linear)

	print(" ")
	print('Linear fit of last three points (m, q):')
	print(f'  m = {popt_linear[0]:.6g} ± {perr_linear[0]:.6g}')
	print(f'  q = {popt_linear[1]:.6g} ± {perr_linear[1]:.6g}')
	
	saturated_small_freq = [4.8, helper.voltage_error(4.8, 1)]  # value of the last not saturated point at small frequency (V)
	saturated_amp = [20*np.log10(saturated_small_freq[0]/Vin[0]), 20*np.sqrt((saturated_small_freq[1]/Vin[0])**2 + (saturated_small_freq[0]*Vin[1]/Vin[0]**2)**2)/np.log(10)]

	# Computing cut-off frequency from linear fit
	f_cutoff = [((saturated_amp[0] - popt_linear[1]) / popt_linear[0]), (np.sqrt( (perr_linear[1]/popt_linear[0])**2 + ((saturated_amp[0] - popt_linear[1]) * perr_linear[0]/(popt_linear[0]**2))**2 + (saturated_amp[1]/popt_linear[0])**2 ))/np.log(10)]

	f_cutoff = [10**f_cutoff[0], np.log(10)*f_cutoff[1]*10**f_cutoff[0]]

	print(" ")
	print("Cut-off frequency from linear fit:")
	print(f'  f_c = {f_cutoff[0]/helper.TO_K:.6g} ± {f_cutoff[1]/helper.TO_K:.6g} KHz')

	# Plotting
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(f/helper.TO_K, H, yerr=sH, fmt='o', label='Data', color='black', ms = 3, lw = 1.6, zorder = 3)
	ax[0].plot(f_fine/helper.TO_K, y_fit, label='Fitted function', color='blue', lw = 1.2)
	#ax[0].plot(np.linspace(30*1e3, 300*1e3, 400)/helper.TO_K, shaper_module_asymptotic(np.linspace(30*1e3, 300*1e3, 400), popt[1]), label='Asymptotic function', color='deepskyblue', lw = 1.2, linestyle='--')
	ax[0].set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \, (V/V)$')
	ax[0].set_title('Inverting OAMP - Transfer function abs')
	ax[1].errorbar(f/helper.TO_K, V_residual2, yerr=V_residual_err2, fmt='o', label='Residuals', color='black', ms = 3, lw = 1.6, zorder = 3)
	ax[1].axhline(weighted_mean_V_residual2, color='gray', linestyle='--', lw = 1.5, label = 'Weighted mean')
	ax[0].text(50, 3, r'$A_{{\text{{ fit}}}}$ = {a:.2f} ± {b:.2f}'.format(a=popt2[0], b=perr2[0]), size=12)
	ax[0].text(50, 2.75, r'$f_{{\,\text{{ t}}}}$ = {a:.0f} ± {b:.0f} KHz'.format(a=popt2[1]/helper.TO_K, b = perr2[1]/helper.TO_K), size=12)
	ax[0].text(50, 2.5, r'$\chi^2 \, / \, DOF$ = {a:.1f} / {b:.0f}'.format(a=chi22, b=len(f)-2), size=12)
	#ax[1].fill_between(f/helper.TO_K, weighted_mean_V_residual - weighted_mean_V_residual_std, weighted_mean_V_residual + weighted_mean_V_residual_std, color='lightgray', alpha=0.5, label='Weighted mean ± 1 σ')
	ax[1].set_xlabel(r'$f \, (KHz)$')
	ax[1].set_ylabel('Residuals (V)')
	ax[1].legend(ncol=2)
	ax[0].legend(loc='upper right')
	ax[1].set_ylim(-0.23, 0.27)
	ax[1].text(140, 0.15, r'$t_{{\text{{ 0 - mean}}}}$ = {a:.2f}'.format(a=helper.compatibility(0, 0, weighted_mean_V_residual2, weighted_mean_V_residual_std2)), size=12)
	#ax[1].text(100, -0.015, r'$t_{{\text{{0 - mean}}}}$ = {a:.2f}'.format(a=helper.compatibility(0, 0, weighted_mean_V_residual, weighted_mean_V_residual_std)), size=12)
	plt.tight_layout()
	plt.savefig(os.path.join(base_dir, 'shaper_H_diff_tau.png'), dpi=150)
	#print(f'Graph saved to: {outpath}')

	# Plotting bode --> module in dB
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5), sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(f/helper.TO_K, A, yerr=sA, fmt='o', label='Data (dB)', color='black', ms=3, lw=1.6, zorder = 3)
	ax[0].plot(f_fine/helper.TO_K, 20*np.log10(y_fit), label='Fitted function (dB)', color='blue', lw=1.4, zorder = 2)
	ax[0].set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \, (db)$')
	ax[0].plot(f_fine_linear/helper.TO_K, y_fit_linear, label='Linear fit last 3 points', color='deepskyblue', lw=1.4, linestyle='--', zorder = 1)
	#ax[0].axvspan(f_low/helper.TO_K, f_high/helper.TO_K, color='gold', alpha=0.4, label='Bandwidth region')
	ax[0].axhline(20*np.log10(saturated_small_freq[0]/Vin[0]), color='darkorange', lw=1.5, label='Saturated voltage', linestyle='--', zorder = 0)
	ax[0].text(0.12, 12, r'$f_{{\text{{ t}}}}$ = {a:.0f} ± {b:.0f}'.format(a=popt2[1]/helper.TO_K, b=perr2[1]/helper.TO_K), size=12)
	ax[0].text(0.12, 11.2, r'$f_{{\text{{ intersection}}}}$ = {a:.0f} ± {b:.0f} KHz'.format(a=f_cutoff[0]/helper.TO_K, b=f_cutoff[1]/helper.TO_K), size=12)
	ax[0].text(0.12, 10.4, r'$t_{{\text{{ f}}}}$ = {a:.2f}'.format(a=helper.compatibility(popt2[1], perr2[1], f_cutoff[0], f_cutoff[1])), size=12)
	ax[0].text(0.12, 9.6, r'$m_{{\text{{ linear}}}}$ = {a:.1f} ± {b:.1f} db/decade'.format(a=popt_linear[0], b=perr_linear[0]), size=12)
	ax[0].legend(loc='lower left')
	ax[0].set_title('Inverting OAMP - Transfer function abs (dB)')
	ax[0].set_xscale('log')
	ax[0].set_xlim(0.07, 3000)
	#ax[1].text(0.15, -1.8, r'$t_{{\text{{0 - mean}}}}$ = {a:.2f}'.format(a=helper.compatibility(0, 0, weighted_mean_V_residual, weighted_mean_V_residual_std)), size=12)
	ax[1].errorbar(f/helper.TO_K, A - 20*np.log10(helper.inverting_oamp_module(f, *popt2)), yerr=sA, fmt='o', label='Residuals (dB)', color='black', ms=3, lw=1.6, zorder = 3)
	ax[1].axhline(weighted_mean_V_residual2, color='gray', linestyle='--', lw=1.5, label = 'Weighted mean')
	ax[1].set_xlabel(r'$f$ (kHz)')
	ax[1].set_ylabel('Residuals (dB)')
	ax[1].legend(ncol=2, loc='lower right')
	# ax[1].set_ylim(-2.8, 2)
	plt.tight_layout()
	plt.savefig(os.path.join(base_dir, 'shaper_H_diff_tau_dB.png'), dpi=150)

	# region - LTspice simulation data plotting

	# Getting data from LTspice simulation
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path0 = os.path.join(base_dir, 'LT_data.txt') # no probe
	if not os.path.exists(data_path0):
		raise FileNotFoundError(f'Data file not found: {data_path0}')

	# Reading data from file
	# Data style: frequency (f), amplification (dB)
	f0, amp0 = helper.read_data_xy(data_path0)

	# Searching for possibile reading errors
	if f0.size == 0: 
		raise RuntimeError('No valid data read from file.')

	# Plotting LTspice data along with experimental data and fit
	fig, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
	ax.errorbar(f/helper.TO_K, H, yerr = sH, fmt='o', label='Experimental Data', color='black', ms=3, lw=1.6, zorder = 3)
	ax.plot(f_fine/helper.TO_K, y_fit, label='Fitted function', color='dodgerblue', lw=1.5)
	ax.plot(f0/helper.TO_K, 10**(amp0/20), label='LTspice Simulation', color='darkorange', lw=1.5)
	# ax.plot(f_fine/helper.TO_K, helper.shaper_module(f_fine, tau1 = Tau1[0] + R1[0]*2e-12, tau2 = Tau2[0] + R2[0]*2e-12, semplified=False), label='Tau2 + 2 pF (dB)', color='purple', lw=1.2)

	# ax.plot(f2/helper.TO_K, amp2, label='LTspice Simulation 1 (dB)', color='green', lw=1.2)
	# ax.plot(f3/helper.TO_K, amp3, label='LTspice Simulation 2 (dB)', color='red', lw=1.2)
	ax.set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \,$ (V/V)')
	ax.legend(loc='upper right')
	ax.set_title('LTspice Simulation Comparison')
	ax.set_xlabel(r'$f$ (kHz)')

	# Plotting LTspice data along with experimental data and fit bode
	fig, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
	ax.errorbar(f/helper.TO_K, A, yerr = sA, fmt='o', label='Experimental Data', color='black', ms=3, lw=1.6, zorder = 3)
	ax.plot(f_fine/helper.TO_K, 20*np.log10(y_fit), label='Fitted function', color='dodgerblue', lw=1.5)
	ax.plot(f0/helper.TO_K, amp0, label='LTspice Simulation', color='darkorange', lw=1.5)
	# ax.plot(f2/helper.TO_K, amp2, label='LTspice Simulation 1 (dB)', color='green', lw=1.2)
	# ax.plot(f3/helper.TO_K, amp3, label='LTspice Simulation 2 (dB)', color='red', lw=1.2)
	ax.set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \, (db)$')
	ax.legend(loc='upper right')
	ax.set_xscale('log')
	ax.set_title('LTspice Simulation Comparison (dB)')
	ax.set_xlabel(r'$f$ (kHz)')
	# endregion

	plt.show()

if __name__ == '__main__':
	main()
