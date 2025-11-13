import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
import mplhep as hep
from cycler import cycler
import statistics
from scipy.odr import *
from scipy.interpolate import interp1d
from Physics_lab import LabH as helper # Importing helper functions

# Variable to set, each vector in the form [value, error]
Vin = [2, helper.voltage_error(2, 0.5)]    # Input voltage with scale (V) 
R1 = [147.92e3, helper.resistance_error(147.92e3, 100e3)]  # Input resistance with scale (Ohm)
R2 = [148.02e3, helper.resistance_error(148.02e3, 100e3)]  # Second resistance with scale (Ohm)
C1 = [90e-12, helper.capacitance_error(90e-12, 1e-9)]  # Feedback capacitance with scale (F)
C2 = [90e-12, helper.capacitance_error(90e-12, 1e-9)]  # Second capacitance with scale (F)
C_back = [22e-12, helper.capacitance_error(22e-12, 1e-9)]  # Background capacitance with scale (F)

# Computing C1 and C2 without background
C1_eff = [C1[0] - C_back[0], np.sqrt(pow(2.5*(C1[0]+C_back[0])/50, 2)/12 + 2*pow(30*1e-12, 2)/12)]
C2_eff = [C2[0] - C_back[0], np.sqrt(pow(2.5*(C2[0]+C_back[0])/50, 2)/12 + 2*pow(30*1e-12, 2)/12)]

print("Effective Capacitances:")
print(f"C1: {C1[0]*1e12:.2f} ± {C1[1]*1e12:.2f} pF")
print(f"C2: {C2[0]*1e12:.2f} ± {C2[1]*1e12:.2f} pF")
print(f'C1_noback: {C1_eff[0]*1e12:.2f} ± {C1_eff[1]*1e12:.2f} pF')
print(f'C2_noback: {C2_eff[0]*1e12:.2f} ± {C2_eff[1]*1e12:.2f} pF')

# Computing Tau1 and Tau2
Tau1 = [R1[0] * C1_eff[0], np.sqrt( (R1[1]*C1_eff[0])**2 + (C1_eff[1]*R1[0])**2 )]
Tau2 = [R2[0] * C2_eff[0], np.sqrt( (R2[1]*C2_eff[0])**2 + (C2_eff[1]*R2[0])**2 )]

print(" ")
print("Effective Time Constants:")
print(f'Tau1: {Tau1[0]*1e6:.2f} ± {Tau1[1]*1e6:.2f} µs')
print(f'Tau2: {Tau2[0]*1e6:.2f} ± {Tau2[1]*1e6:.2f} µs')

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

# Asintotic function definition for the module of shaper transfer function
def shaper_module_asymptotic(f, tau2):
	"""
	Asymptotic function for the shaper transfer function module - freq - tau2
	"""
	omega = 2 * np.pi * f
	return 1/(tau2*omega)

#============= MAIN FUNCTION ================
def main():

	# Getting data for the shaper frequency response
	# resolve the relative path to the data file in the same folder as the script
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(base_dir, 'data_shaper_bode.txt') # Data NO probe
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
	sH = H*np.sqrt(pow(Vout_scale/(Vout*10*np.sqrt(3)), 2) + pow(0.5/(Vin[0]*10*np.sqrt(3)), 2)+ pow(3/(np.sqrt(3)*100), 2))

	# Computing amplification in db
	A = 20 * np.log10(H)
	sA = 20 * sH /(np.log10(10) * H)

	# FITTING FOR SHAPER WITH DIFFERENT TAU

	# Searching for initial parameters
	p0 = [Tau1[0], Tau2[0]]

	popt, perr, f_residual, V_residual, chi2 = helper.fit_shaper_module(f, H, H_error=sH, init0=p0)

	# Computing residuals
	
	# calculate the residuals error
	V_residual_err = sH

	# Computing the weighted mean of the residuals
	weighted_mean_V_residual = np.average(V_residual, weights=1/V_residual_err**2)
	weighted_mean_V_residual_std = np.sqrt(1 / np.sum(1/V_residual_err**2))

	# computing compatibility between weighted mean of residuals and 0
	r_residual = np.abs(weighted_mean_V_residual)/weighted_mean_V_residual_std

	# Print fit results
	print("")
	print('Fit parameters (tau1, tau2):')
	print(f'  tau1 = {popt[0]*helper.TO_MU:.6g} ± {perr[0]*helper.TO_MU:.6g}')
	print(f'  tau2 = {popt[1]*helper.TO_MU:.6g} ± {perr[1]*helper.TO_MU:.6g}')
	print("Chi-squared:", chi2)

	# Computing the frequency of the max of the function
	f_max = 1/(2*np.pi*np.sqrt(popt[1] * popt[0]))
	f_max_err = np.sqrt((perr[0])**2/(2*popt[1]*popt[0]**3) + (perr[1])**2/(2*popt[0]*popt[1]**3))/(2*np.pi)

	# Taking points for the fit line
	f_fine = np.linspace(np.min(f), np.max(f), 5000)
	y_fit = helper.shaper_module(f_fine, *popt)
	y_fit_1sigma = helper.shaper_module(f_fine, *(popt + perr))
	y_fit_m1sigma = helper.shaper_module(f_fine, *(popt - perr))

	print(" ")
	print(f'Resonance frequency: f_max = {f_max:.2f} ± {f_max_err:.2f} Hz')

	# Searching for anomlies in capacity
	C2_fit = [popt[1]/R2[0], np.sqrt((perr[1]/R2[0])**2 + pow(R2[1]*popt[1]/R2[0]**2, 2))]
	C2_offset = [np.abs(C2_eff[0] - C2_fit[0]), np.sqrt(C2_eff[1]**2 + C2_fit[1]**2)]

	print(" ")
	print("Compatibility between tau1:", helper.compatibility(popt[0], perr[0], Tau1[0], Tau1[1]))
	print("Compatibility between tau2:", helper.compatibility(popt[1], perr[1], Tau2[0], Tau2[1]))
	print("Difference in tau2: ", (Tau2[0] - popt[1])*helper.TO_MU, " ± ", (np.sqrt(Tau2[1]**2 + perr[1]**2))*helper.TO_MU, " µs")
	print("Fit capacitance C2: ", C2_fit[0]*helper.TO_P, " ± ", C2_fit[1]*helper.TO_P, " pF")
	print("Disturbance capacitance: ", C2_offset[0]*helper.TO_P, " ± ", C2_offset[1]*helper.TO_P, " pF")
	
	# Searching the bandwidth limits
	f_grid = np.linspace(np.min(f), np.max(f), 5000)
	H_fit_grid = helper.shaper_module(f_grid, *popt)
	A_fit_grid = 20 * np.log10(H_fit_grid)
	A_max = np.max(A_fit_grid)
	A_6db = A_max - 6
	# Searching for frequencies where the module crosses Amax - 6dB
	mask = A_fit_grid >= A_6db
	if np.any(mask):
		idx = np.where(mask)[0]
		f_low = f_grid[idx[0]]
		f_high = f_grid[idx[-1]]

    # region - PLOTTING SHAPER WITH DIFFERENT TAU
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(f/helper.TO_K, H, yerr=sH, fmt='o', label='Data', color='black', ms = 3, lw = 1.6, zorder = 3)
	ax[0].plot(f_fine/helper.TO_K, y_fit, label='Fitted function', color='blue', lw = 1.2)
	#ax[0].plot(np.linspace(30*1e3, 300*1e3, 400)/helper.TO_K, shaper_module_asymptotic(np.linspace(30*1e3, 300*1e3, 400), popt[1]), label='Asymptotic function', color='deepskyblue', lw = 1.2, linestyle='--')
	ax[0].set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \, (V/V)$')
	ax[0].set_title('Shaper - Transfer function abs')
	#ax[0].text(0, 0.35, r'$\tau_{{\,\text{{BNC}}}}$ = {e:.0f} $\pm$ {f:.0f} $\mu s$'.format(e=tau*1e6, f = tau_err*1e6), size=12)
	ax[0].axvline(f_max/helper.TO_K, color = 'darkorange', label = 'Resonance frequency', lw = 1.5, ls = '--', zorder = 0)
	ax[0].axvspan(f_low/helper.TO_K, f_high/helper.TO_K, color='gold', alpha=0.4, label='Bandwidth region')
	#ax[0].axvspan(xmin=f_max/helper.TO_K - f_max_err/helper.TO_K, xmax=f_max/helper.TO_K + f_max_err/helper.TO_K, color='gold', alpha=0.5, label='Resonance frequency uncertainty', zorder=0)
	ax[0].text(400, 0.25, r'$f_{{\,\text{{t}}}}^{{\text{{sh}}}}$ = {a:.1f} ± {b:.1f} KHz'.format(a=f_max/helper.TO_K, b = f_max_err/helper.TO_K), size=12)
	ax[0].text(400, 0.22, r'$\tau_{{1}}$ = {a:.2f} ± {b:.2f} µs'.format(a=popt[0]*1e6, b = perr[0]*1e6), size=12)
	ax[0].text(400, 0.19, r'$\tau_{{2}}$ = {a:.2f} ± {b:.2f} µs'.format(a=popt[1]*1e6, b = perr[1]*1e6), size=12)
	ax[0].text(400, 0.16, r'$t_{{\tau}}$ = {:.1f}'.format(helper.compatibility(popt[0], perr[0], popt[1], perr[1])), size=12)
	ax[0].text(400, 0.13, r'$\chi^2 \, / \, DOF$ = {a:.1f} / {b:.0f}'.format(a=chi2, b=len(f)-2), size=12)
	ax[1].errorbar(f/helper.TO_K, V_residual, yerr=V_residual_err, fmt='o', label='Residuals', color='black', ms = 3, lw = 1.6, zorder = 3)
	ax[1].axhline(weighted_mean_V_residual, color='gray', linestyle='--', lw = 1.5, label = 'Weighted mean')
	#ax[1].fill_between(f/helper.TO_K, weighted_mean_V_residual - weighted_mean_V_residual_std, weighted_mean_V_residual + weighted_mean_V_residual_std, color='lightgray', alpha=0.5, label='Weighted mean ± 1 σ')
	ax[1].set_xlabel(r'$f \, (KHz)$')
	ax[1].set_ylabel('Residuals (V)')
	ax[1].legend(ncol=2)
	ax[0].legend(loc='upper right')
	ax[1].set_ylim(-0.026, 0.02)
	ax[1].text(100, -0.015, r'$t_{{\text{{0 - mean}}}}$ = {a:.2f}'.format(a=helper.compatibility(0, 0, weighted_mean_V_residual, weighted_mean_V_residual_std)), size=12)
	plt.tight_layout()
	plt.savefig(os.path.join(base_dir, 'shaper_H_diff_tau.png'), dpi=150)
	#print(f'Graph saved to: {outpath}')

	# Plotting bode --> module in dB
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5), sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(f/helper.TO_K, A, yerr=sA, fmt='o', label='Data (dB)', color='black', ms=3, lw=1.6, zorder = 3)
	ax[0].plot(f_fine/helper.TO_K, 20*np.log10(y_fit), label='Fitted function (dB)', color='blue', lw=1.2)
	ax[0].set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \, (db)$')
	ax[0].axvspan(f_low/helper.TO_K, f_high/helper.TO_K, color='gold', alpha=0.4, label='Bandwidth region')
	ax[0].legend(loc='upper right')
	ax[0].set_title('Shaper - Transfer function abs (dB)')
	ax[0].set_xscale('log')
	ax[0].text(0.097, -10, r'$f_{{\,\text{{t}}}}^{{\text{{sh}}}}$ = {a:.1f} ± {b:.1f} KHz'.format(a=f_max/helper.TO_K, b = f_max_err/helper.TO_K), size=12)
	ax[0].text(0.097, -12.5, r'$\tau_{{1}}$ = {a:.2f} ± {b:.2f} µs'.format(a=popt[0]*1e6, b = perr[0]*1e6), size=12)
	ax[0].text(0.097, -15, r'$\tau_{{2}}$ = {a:.2f} ± {b:.2f} µs'.format(a=popt[1]*1e6, b = perr[1]*1e6), size=12)
	ax[0].text(0.097, -17.5, r'$t_{{\tau}}$ = {:.1f}'.format(helper.compatibility(popt[0], perr[0], popt[1], perr[1])), size=12)
	ax[0].text(0.097, -20, r'$\chi^2/DOF$ = {a:.1f}/{b:.0f}'.format(a=chi2, b=len(f)-2), size=12)
	ax[0].axvline(f_max/helper.TO_K, color = 'darkorange', label = 'Resonance frequency', lw = 1.5, ls = '--', zorder = 0)
	ax[0].set_xlim(0.07, 3000)
	ax[1].text(0.15, -1.8, r'$t_{{\text{{0 - mean}}}}$ = {a:.2f}'.format(a=helper.compatibility(0, 0, weighted_mean_V_residual, weighted_mean_V_residual_std)), size=12)
	ax[1].errorbar(f/helper.TO_K, A - 20*np.log10(helper.shaper_module(f, *popt)), yerr=sA, fmt='o', label='Residuals (dB)', color='black', ms=3, lw=1.6, zorder = 3)
	ax[1].axhline(weighted_mean_V_residual, color='gray', linestyle='--', lw=1.5, label = 'Weighted mean')
	ax[1].set_xlabel(r'$f$ (kHz)')
	ax[1].set_ylabel('Residuals (dB)')
	ax[1].legend(ncol=2, loc='lower right')
	ax[1].set_ylim(-2.8, 2)
	plt.tight_layout()
	plt.savefig(os.path.join(base_dir, 'shaper_H_diff_tau_dB.png'), dpi=150)

	# endregion

	# region - PLOTTING SHAPER WITH SAME THEORITAL TAU

	print(" ")
	print("THEORITAL FIT WITH SAME TAU")

	# Computing theoritcal value for tau
	theoretical_tau = [(Tau1[0] + Tau2[0]) / 2,  np.sqrt( (Tau1[1]**2 + Tau2[1]**2))/2]

	print("Theoretical tau: ", theoretical_tau[0]*helper.TO_MU, " ± ", theoretical_tau[1]*helper.TO_MU, " µs")

	# Taking points for the fit line
	f_fine = np.linspace(np.min(f), np.max(f), 5000)
	y_fitD = helper.shaper_module(f_fine, tau1 = theoretical_tau[0], offset = 1, semplified=True)

	# Taking point for plotting
	yD = helper.shaper_module(f, tau1 = theoretical_tau[0], offset = 1, semplified=True)
	yD1sigma = helper.shaper_module(f_fine, tau1 = theoretical_tau[0] + theoretical_tau[1], offset = 1, semplified=True)
	yD_m1sigma = helper.shaper_module(f_fine, tau1 = theoretical_tau[0] - theoretical_tau[1], offset = 1, semplified=True)
	y_errD = np.abs(helper.shaper_module(f, tau1 = theoretical_tau[0] + theoretical_tau[1], offset = 1, semplified=True) - helper.shaper_module(f, tau1 = theoretical_tau[0] - theoretical_tau[1], offset = 1, semplified=True))/2


	# calculate the difference between data and theoretical value
	y_residualD = A - 20*np.log10(yD)
	y_residual_errD = np.sqrt(sA**2 + (20*y_errD/(yD*np.log(10)))**2)

	# Computing the weighted mean of the residuals
	weighted_mean_V_residualD = np.average(y_residualD, weights=1/y_residual_errD**2)
	weighted_mean_V_residual_stdD = np.sqrt(1 / np.sum(1/y_residual_errD**2))

	# computing compatibility between weighted mean of residuals and 0
	r_residualD = np.abs(weighted_mean_V_residualD)/weighted_mean_V_residual_stdD

	# Print chi squared result
	chi2_theoretical = helper.chi_squared(H, yD, np.sqrt(sH**2 + (y_errD)**2))
	print(" ")
	print('Chi square data - theoretical:')
	print("Chi-squared:", chi2_theoretical)

    # Plotting
	fig, ax2 = plt.subplots(1, 1, figsize=(6.5,6.5),sharex=True)
	ax2.errorbar(f/helper.TO_K, H, yerr=sH, fmt='o', label='Data', color='black', ms = 3, lw = 1.6, zorder = 3)
	ax2.plot(f_fine/helper.TO_K, y_fitD, label='Expected function', color='darkred', lw = 1.4, linestyle='--')
	ax2.plot(f_fine/helper.TO_K, y_fit, label='Fitted function diff tau', color='dodgerblue', lw = 1.4)
	#ax[0].fill_between(f_fine/helper.TO_K, yD_m1sigma, yD1sigma, color='red', alpha=0.5, label='Theoretical function ± 1 σ')
	ax2.set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \,$ (V/V)')

	ax2.legend(loc='upper right')
	ax2.set_title('Shaper - Expected Transfer function analysis')
	#ax[0].text(0, 0.35, r'$\tau_{{\,\text{{BNC}}}}$ = {e:.0f} $\pm$ {f:.0f} $\mu s$'.format(e=tau*1e6, f = tau_err*1e6), size=12)
	ax2.text(400, 0.25, r'$\tau_{{\text{{ exp}}}}$ = {a:.0f} ± {b:.0f} µs'.format(a=theoretical_tau[0]*helper.TO_MU, b=theoretical_tau[1]*helper.TO_MU), size=12)
	ax2.text(400, 0.23, r'$\tau_{{\text{{ exp1}}}}$ = {a:.0f} ± {b:.0f} µs'.format(a=Tau1[0]*helper.TO_MU, b=Tau1[1]*helper.TO_MU), size=12)
	ax2.text(400, 0.21, r'$\tau_{{\,\text{{ exp2}}}}$ = {a:.0f} ± {b:.0f} µs'.format(a=Tau2[0]*helper.TO_MU, b=Tau2[1]*helper.TO_MU), size=12)
	ax2.text(400, 0.19, r'$t_{{\,\text{{1}}}}$ = {a:.1f}'.format(a= helper.compatibility(popt[0], perr[0], Tau1[0], Tau1[1])), size=12)
	ax2.text(400, 0.17, r'$t_{{\text{{2}}}}$ = {:.1f}'.format(helper.compatibility(popt[1], perr[1], Tau2[0], Tau2[1])), size=12)
	ax2.text(400, 0.15, r'$\chi^2_{{\text{{ exp}}}} \, / \, DOF$ = {a:.1f} / {b:.0f}'.format(a=chi2_theoretical, b=len(f)-1), size=12)
	ax2.set_xlabel(r'$f$ (kHz)')
	# ax[1].axhline(0, color='gray', linestyle='--', lw = 1.5, label = "Ideal value")
	# ax[1].set_xlabel(r'$f \, (Hz)$')
	# ax[1].set_ylabel('Residuals (V)')
	# ax[1].legend(ncol=2)
	# ax[1].set_ylim(-0.01, 0.065)
	plt.tight_layout()
	plt.savefig(os.path.join(base_dir, 'shaper_H_same_tau.png'), dpi=150)
	#print(f'Graph saved to: {outpath}')

	# Plotting bode --> module in dB
	fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5), sharex=True, height_ratios=[2, 0.6])
	ax[0].errorbar(f/helper.TO_K, A, yerr=sA, fmt='o', label='Data (dB)', color='black', ms=3, lw=1.6, zorder = 3)
	ax[0].plot(f_fine/helper.TO_K, 20*np.log10(y_fit), label='Fitted function (dB)', color='dodgerblue', lw=1.4)
	ax[0].plot(f_fine/helper.TO_K, 20*np.log10(y_fitD), label='Expected function (dB)', color='darkred', lw=1.4, linestyle='--')
	ax[0].fill_between(f_fine/helper.TO_K, 20*np.log10(yD_m1sigma), 20*np.log10(yD1sigma), color='darkorange', alpha=0.4, label='Expected function ± 1 σ')
	ax[0].set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \, (db)$')
	#ax[0].axvspan(f_low/helper.TO_K, f_high/helper.TO_K, color='gold', alpha=0.4, label='Bandwidth region')
	ax[0].legend(loc='lower right', ncol = 2)
	ax[0].set_title('Shaper - Expected Transfer function analysis (dB)')
	ax[0].set_xscale('log')
	ax[0].text(0.097, -10, r'$C2_{{\text{{ exp}}}}$ = {a:.0f} ± {b:.0f} pF'.format(a= C2_eff[0]*1e12, b = C2_eff[1]*1e12), size=12)
	ax[0].text(0.097, -12.5, r'$C2_{{\text{{ fit}}}}$ = {a:.1f} ± {b:.1f} pF'.format(a=C2_fit[0]*1e12, b = C2_fit[1]*1e12), size=12)
	ax[0].text(0.097, -15, r'$C2_{{\text{{ offset}}}}$ = {a:.0f} ± {b:.0f} pF'.format(a=C2_offset[0]*1e12, b = C2_offset[1]*1e12), size=12)
	#ax[0].axvline(f_max/helper.TO_K, color = 'darkorange', label = 'Resonance frequency', lw = 1.5, ls = '--', zorder = 0)
	ax[0].set_xlim(0.07, 3000)
	ax[1].text(0.15, -2.4, r'$t_{{\text{{0 - mean}}}}$ = {a:.2f}'.format(a=helper.compatibility(0, 0, weighted_mean_V_residual, weighted_mean_V_residual_std)), size=12)
	ax[1].errorbar(f/helper.TO_K, y_residualD, yerr=y_residual_errD, fmt='o', label='Differences (dB)', color='black', ms=3, lw=1.6, zorder=3)
	ax[1].axhline(0, color='gray', linestyle='--', lw=1.5, label = 'Zero')
	#ax[1].axhspan(weighted_mean_V_residual - weighted_mean_V_residual_std, weighted_mean_V_residual + weighted_mean_V_residual_std, color='lightgray', alpha=0.5, label='Weighted mean ± 1 σ')
	ax[1].set_xlabel(r'$f$ (kHz)')
	ax[1].set_ylabel('Difference (dB)')
	ax[1].legend(ncol=2, loc='upper right')
	ax[1].set_ylim(-3.5, 3)
	plt.tight_layout()
	plt.savefig(os.path.join(base_dir, 'shaper_H_exp_tau_dB.png'), dpi=150)

	# endregion

	# region - LTspice simulation data plotting

	# Getting data from LTspice simulation
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path0 = os.path.join(base_dir, 'data_LTspice.txt') # no probe
	data_path1 = os.path.join(base_dir, 'data_LTspice1.txt') # no correction
	data_path2 = os.path.join(base_dir, 'data_LTspice2.txt') # +2 pF C2 -0.1 pF probe correction
	if not os.path.exists(data_path0):
		raise FileNotFoundError(f'Data file not found: {data_path0}')

	# Reading data from file
	# Data style: frequency (f), amplification (dB)
	f0, amp0 = helper.read_data_xy(data_path0)
	f1, amp1 = helper.read_data_xy(data_path1)
	f2, amp2 = helper.read_data_xy(data_path2)

	# Searching for possibile reading errors
	if f0.size == 0: 
		raise RuntimeError('No valid data read from file.')

	# Plotting LTspice data along with experimental data and fit
	fig, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
	ax.errorbar(f/helper.TO_K, H, yerr = sH, fmt='o', label='Experimental Data', color='black', ms=3, lw=1.6, zorder = 3)
	ax.plot(f_fine/helper.TO_K, y_fit, label='Fitted function', color='dodgerblue', lw=1.5)
	ax.plot(f0/helper.TO_K, 10**(amp0/20), label='LTspice Simulation no probe', color='darkorange', lw=1.5)
	ax.plot(f1/helper.TO_K, 10**(amp1/20), label='LTspice Simulation no correction', color='limegreen', lw=1.5)
	ax.plot(f2/helper.TO_K, 10**(amp2/20), label='LTspice Simulation with correction', color='blue', lw=1.5)
	ax.plot(f_fine/helper.TO_K, y_fitD, label='Expected function', color='firebrick', lw=1.5, linestyle='--')
	# ax.plot(f_fine/helper.TO_K, helper.shaper_module(f_fine, tau1 = Tau1[0] + R1[0]*2e-12, tau2 = Tau2[0] + R2[0]*2e-12, semplified=False), label='Tau2 + 2 pF (dB)', color='purple', lw=1.2)

	# ax.plot(f2/helper.TO_K, amp2, label='LTspice Simulation 1 (dB)', color='green', lw=1.2)
	# ax.plot(f3/helper.TO_K, amp3, label='LTspice Simulation 2 (dB)', color='red', lw=1.2)
	ax.set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \,$ (V/V)')
	ax.legend(loc='upper right')
	ax.set_title('Shaper - LTspice Simulation Comparison and correction')
	ax.set_xlabel(r'$f$ (kHz)')

	# Plotting LTspice data along with experimental data and fit bode
	fig, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
	ax.errorbar(f/helper.TO_K, A, yerr = sA, fmt='o', label='Experimental Data', color='black', ms=3, lw=1.6, zorder = 3)
	ax.plot(f_fine/helper.TO_K, 20*np.log10(y_fit), label='Fitted function', color='dodgerblue', lw=1.5)
	ax.plot(f0/helper.TO_K, amp0, label='LTspice Simulation no probe', color='darkorange', lw=1.5)
	ax.plot(f1/helper.TO_K, amp1, label='LTspice Simulation no correction', color='limegreen', lw=1.5)
	ax.plot(f2/helper.TO_K, amp2, label='LTspice Simulation with correction', color='blue', lw=1.5)
	ax.plot(f_fine/helper.TO_K, 20*np.log10(y_fitD), label='Expected function', color='firebrick', lw=1.5, linestyle='--')
	# ax.plot(f2/helper.TO_K, amp2, label='LTspice Simulation 1 (dB)', color='green', lw=1.2)
	# ax.plot(f3/helper.TO_K, amp3, label='LTspice Simulation 2 (dB)', color='red', lw=1.2)
	ax.set_ylabel(r'$|\, H \, | \, = \, \frac{V_{\text{out}}}{V_{\text{in}}} \, (db)$')
	ax.legend(loc='upper right')
	ax.set_xscale('log')
	ax.set_title('Shaper - LTspice Simulation Comparison and correction (dB)')
	ax.set_xlabel(r'$f$ (kHz)')
	# endregion

	plt.show()

if __name__ == '__main__':
	main()
