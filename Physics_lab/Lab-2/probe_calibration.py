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

	# Getting data from LTspice simulation
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path0 = os.path.join(base_dir, 'probe_LTspice.txt') # no probe
	data_path1 = os.path.join(base_dir, 'probe_LTspice1.txt') # no correction
	if not os.path.exists(data_path0):
		raise FileNotFoundError(f'Data file not found: {data_path0}')

	# Reading data from file
	# Data style: frequency (f), amplification (dB)
	f0, amp0 = helper.read_data_xy(data_path0)
	f1, amp1 = helper.read_data_xy(data_path1)
	
	# Searching for possibile reading errors
	if f0.size == 0: 
		raise RuntimeError('No valid data read from file.')

	# Plotting LTspice data along with experimental data and fit
	fig, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
	ax.plot(f0*helper.TO_MU, amp0, label='Compensated probe', color='darkorange', lw=1.5, zorder = 1)
	ax.plot(f1*helper.TO_MU, amp1, label='-0.1 pF compensation error', color='seagreen', lw=1.5, zorder = 0)
	# ax.plot(f_fine/helper.TO_K, helper.shaper_module(f_fine, tau1 = Tau1[0] + R1[0]*2e-12, tau2 = Tau2[0] + R2[0]*2e-12, semplified=False), label='Tau2 + 2 pF (dB)', color='purple', lw=1.2)

	# ax.plot(f2/helper.TO_K, amp2, label='LTspice Simulation 1 (dB)', color='green', lw=1.2)
	# ax.plot(f3/helper.TO_K, amp3, label='LTspice Simulation 2 (dB)', color='red', lw=1.2)
	ax.set_ylabel(r'$V_{\text{ out}}$ (V)')
	ax.legend(loc='lower right')
	ax.set_title('Probe - LTspice compensation simulation')
	ax.set_xlabel(r'$Time \, (\mu s)$')
	plt.savefig(os.path.join(base_dir, 'probe_compensation_simulation.png'), dpi=150)

	plt.show()

if __name__ == '__main__':
	main()
