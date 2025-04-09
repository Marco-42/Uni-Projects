import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool
import statistics
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib.ticker import LinearLocator
import matplotlib.scale as mscale

# settaggio globale grafici
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


# Function definition

# Function definition - Linear function
def fitf(x, m, q):
    """m  -  q"""
    return m*x + q

def fitf_norm(x, m, q):
    """m  -  q"""
    return (m*x + q)*2/np.pi

def fit_nolin_TS(x, f):
    return 1/np.sqrt(1 + (x/f)**2)

def fit_nolin_phase(x, f):
    return np.arctan(x/f)

def fit_nolin_phase_norm(x, f):
    return np.arctan(x/f)*2/np.pi

def computing_R_err(R):

    # Computing the error on R
    # Checking the R range
    if(R < 1000):
        return np.sqrt(pow(R*0.1/100, 2)/3 + pow(0.01*8, 2)/3)
    else:
        return np.sqrt(pow(R*0.07/100, 2)/3 + pow(0.1*8, 2)/3)

def computing_C_err(C):
    if(C < 10e-9 and C > 0):
        return np.sqrt(pow(C*0.01, 2)/3 + pow(0.01e-9*8, 2)/3)

def compatibility(x, y, x_err, y_err):
    """x, y, x_err, y_err"""
    return np.abs(x-y)/np.sqrt(x_err**2 + y_err**2)

# region - Local linear fit 

# Input file name
inputname1 = 'Local_fit.txt'

# Initial parameter values
m_init= -0.1
q_init = 1

# load sperimental data from file
data = np.loadtxt(inputname1).T
freq = np.array(data[0])
Vout = np.array(data[1])
Vin = np.array(data[2])
time = np.array(data[3])

# Computing phase
phase_not_norm = 2*np.pi*freq*time/1000
phase = phase_not_norm*2/np.pi

# Number of points to fit
N = len(freq)

# Assumed scale value
Vscale = 1  # V
Tscale = 25  # mus

# Assumed reading errors
letturaV = Vscale*2/(np.sqrt(24)*25)
#letturaT = Tscale*2/(np.sqrt(24)*25)
letturaT = Tscale*2/(np.sqrt(12)*25)
errscalaV = 1.5/100

# definition of errors array [x and y errorbars]
freq_err = np.zeros((N), dtype=np.float64)
eVout = np.sqrt((letturaV)**2 + ((errscalaV * Vout)**2))
eVin = np.sqrt((letturaV)**2 + ((errscalaV * Vin)**2))

# Computing phase errors
phase_err_not_norm = np.sqrt(2*np.pi*((freq_err*time)**2 + (freq*letturaT)**2))/1000
phase_err = phase_err_not_norm*2/np.pi

# defining the Transfer function 
TS = Vout/Vin

# defining the error of the Transfer function
TS_err = np.sqrt((eVout/Vin)**2 + (Vout*eVin/(Vin**2))**2)

# Limits for the fit function
x_min = min(freq)
x_max = max(freq)

# Proceding with linear fitting with LM algorithm

# fitting the Transfer function 
popt_TS, pcov_TS = curve_fit(fitf, freq, TS, p0=[m_init, q_init], method='lm', sigma=TS_err, absolute_sigma=True)

# Computing the residual 
residual_TS = TS - fitf(freq, *popt_TS)

# defining the x - axis for the plot
freq_fit = np.linspace(min(freq), max(freq), 1000)

# variables error and chi2
perr_TS = np.sqrt(np.diag(pcov_TS))
chisq_TS = np.sum((residual_TS/TS_err)**2)

# degrees of freedom
df = N - 2

# fitting parameters and errors
m_TS, q_TS = popt_TS 
em_TS, eq_TS = np.sqrt(np.diag(pcov_TS))

# Fitting the phase
popt_phase, pcov_phase = curve_fit(fitf_norm, freq, phase, p0=[m_init, q_init], method='lm', sigma=phase_err, absolute_sigma=True)

# Computing the residual
residual_phase = phase - fitf_norm(freq, *popt_phase)

# variables error and chi2
perr_phase = np.sqrt(np.diag(pcov_phase))
chisq_phase = np.sum((residual_phase/phase_err)**2)

# fitting parameters and errors
m_phase, q_phase = popt_phase
em_phase, eq_phase = np.sqrt(np.diag(pcov_phase))

# calculate the residuals error by quadratic sum using the variance theorem
#TS_residual_err = np.sqrt(pow(TS_err, 2) + pow(eq_TS, 2) + pow(freq*em_TS, 2))
#phase_residual_err = np.sqrt(pow(phase_err, 2) + pow(eq_phase, 2) + pow(freq*em_phase, 2))
TS_residual_err = TS_err
phase_residual_err = phase_err

# Computing the weighted mean of the residuals
weighted_mean_TS_residual = np.average(residual_TS, weights=1/TS_residual_err**2)
weighted_mean_TS_residual_std = np.sqrt(1 / np.sum(1/TS_residual_err**2))
weighted_mean_phase_residual = np.average(residual_phase, weights=1/phase_residual_err**2)
weighted_mean_phase_residual_std = np.sqrt(1 / np.sum(1/phase_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_TS = np.abs(weighted_mean_TS_residual)/weighted_mean_TS_residual_std
r_residual_phase = np.abs(weighted_mean_phase_residual)/weighted_mean_phase_residual_std

# Array of element for plotting the wheigted mean of residuals
TS_residual_array = np.full((N), weighted_mean_TS_residual)
phase_residual_array = np.full((N), weighted_mean_phase_residual)

# Computing the resonance frequency
f_TS = (1/np.sqrt(2) - q_TS)/m_TS
f_phase = (1/np.sqrt(2) - q_phase)/m_phase

# Computing the resonance frequency error
f_TS_err = np.sqrt((eq_TS/m_TS)**2 + ((1/np.sqrt(2) - q_TS)*em_TS/(m_TS**2))**2 + 2*((1/np.sqrt(2) - q_TS)/(m_TS**3))*pcov_TS[0, 1])
f_phase_err = np.sqrt((eq_phase/m_phase)**2 + ((1/np.sqrt(2) - q_phase)*em_phase/(m_phase**2))**2 + 2*((1/np.sqrt(2) - q_phase)/(m_phase**3))*pcov_phase[0, 1])

# Plotting the local fit
# defining the plot
fig, ax = plt.subplots(3, 1, figsize=(6.5, 9), sharex=True, constrained_layout=True, height_ratios=[2, 1, 1])

# defining points and fit function 
ax[0].errorbar(freq,TS,xerr=freq_err, yerr=TS_err, fmt='o', label=r'TS Data',ms=2,color='black')
ax[0].plot(freq_fit, fitf(freq_fit, *popt_TS), label='TS linear fit', linestyle='--', color='red')
ax[0].errorbar(freq,phase,xerr=freq_err, yerr=phase_err, fmt='o', label=r'Phase Data',ms=2,color='blue')
ax[0].plot(freq_fit, fitf_norm(freq_fit, *popt_phase), label='Phase linear fit', linestyle='--', color='blue')

ax[0].legend(loc='upper left', fontsize = 15)

# plotting the residual graph(in this case only y - residuals)
ax[1].errorbar(freq, residual_TS,yerr=TS_residual_err, fmt='o', label=r'TS residual',ms=2,color='black')
ax[2].errorbar(freq, residual_phase, yerr=phase_residual_err, fmt='o', label=r'Phase Residual', ms=2, color='blue')

# plotting the weighted mean of residuals
ax[1].plot(freq, TS_residual_array, linestyle='--', color='red')
ax[2].plot(freq, phase_residual_array, linestyle='--', color='blue')

R_ylim = np.std(residual_TS)*5 

# setting limit for y axis and the axis labels
ax[1].set_ylim([-0.05,0.05])
ax[0].set_ylim([0.2,0.8])
ax[0].set_ylabel(r'$A\, - \, \Phi \, normalized$', size = 15)
ax[1].set_ylabel(r'Residuals', size = 15)
ax[1].set_xlabel(r'$Frequency$ [$KHz$]', size = 15)
ax[2].set_ylabel(r'Phase Residuals', size=15)
ax[2].set_xlabel(r'$Frequency$ [$KHz$]', size=15)

# Dynamic text position for f_TS and f_phase
y_min, y_max = ax[0].get_ylim() 
text_y1 = y_max - 0.90* (y_max - y_min)  # vetical position under 90% 
text_y2 = y_max - 0.78* (y_max - y_min)  # vertical position under 78%

# Plotting some text
ax[0].text(0.95 * ax[0].get_xlim()[1], text_y1,  # Orizontal position 95% 
           r'$f_{{TS}}$ = {fts:.2f} $\pm$ {efts:.2f} $KHz$'.format(fts=f_TS, efts=f_TS_err),
           size=13, ha='right')  

ax[0].text(0.95 * ax[0].get_xlim()[1], text_y2,  # Orizontal position 95% 
           r'$f_{{phase}}$ = {fph:.2f} $\pm$ {efph:.2f} $KHz$'.format(fph=f_phase, efph=f_phase_err),
           size=13, ha='right') 

# aumatic layout configuration
fig.tight_layout(pad=1, w_pad=0.1, h_pad=0.3)

print(m_TS, " ", em_TS, " - ", q_TS, " ", eq_TS)
# show the plot to user 
plt.show()
            
# endregion

# region - Non linear fit
# Input file name
inputname2 = 'Bode.txt'

# Initial parameter values
a_init = 3.19

# load sperimental data from file
data = np.loadtxt(inputname2).T
freq = np.array(data[0])
Vout = np.array(data[1])
Vout_scale = np.array(data[2])
Vin = np.array(data[3])
time = np.array(data[4])
time_scale = np.array(data[5])

# Computing phase
phase_not_norm = 2*np.pi*freq*time/1000
phase = phase_not_norm*2/np.pi

# Number of points to fit
N = len(freq)

# Assumed reading errors
letturaV = Vout_scale*2/(np.sqrt(24)*25)
#letturaT = time_scale*2/(np.sqrt(24)*25)
letturaT = time_scale*2/(np.sqrt(12)*25)
errscalaV = 1.5/100

# definition of errors array [x and y errorbars]
freq_err = np.zeros((N), dtype=np.float64)
eVout = np.sqrt((1*2/(np.sqrt(24)*25))**2 + ((1.5*Vout/100)**2))
eVin = np.sqrt((letturaV)**2 + ((errscalaV * Vin)**2))

# Computing phase errors
phase_err_not_norm = np.sqrt(2*np.pi*((freq_err*time)**2 + (freq*letturaT)**2))/1000
phase_err = phase_err_not_norm*2/np.pi

# defining the Transfer function 
TS = Vout/Vin

# defining the error of the Transfer function
TS_err = TS*np.sqrt((letturaV/Vin)**2 + (letturaV/Vout)**2 + 2*(errscalaV*TS)**2)

# Limits for the fit function
x_min = min(freq)
x_max = max(freq)

# defining the x points for fit
freq_fit = np.linspace(x_min, x_max, 1000)

# fitting the Transfer function 
popt_TS, pcov_TS = curve_fit(fit_nolin_TS, freq, TS, p0=[a_init], method='lm', sigma=TS_err, absolute_sigma=True)

# Computing the residual 
residual_TS = TS - fit_nolin_TS(freq, *popt_TS)

# variables error and chi2
perr_TS = np.sqrt(np.diag(pcov_TS))
chisq_TS = np.sum((residual_TS/TS_err)**2)

# degrees of freedom
df = N - 2

# fitting parameters and errors
a_TS = popt_TS[0]
ea_TS = np.sqrt(pcov_TS[0, 0])

# Fitting the phase
popt_phase, pcov_phase = curve_fit(fit_nolin_phase_norm, freq, phase, p0=[a_init], method='lm', sigma=phase_err, absolute_sigma=True)

# Computing the residual
residual_phase = phase - fit_nolin_phase_norm(freq, *popt_phase)

# variables error and chi2
perr_phase = np.sqrt(np.diag(pcov_phase))
chisq_phase = np.sum((residual_phase/phase_err)**2)

# fitting parameters and errors
a_phase = popt_phase[0]
ea_phase = np.sqrt(pcov_phase[0, 0])

# calculate the residuals error by quadratic sum using the variance theorem
#TS_residual_err = np.sqrt(pow(TS_err, 2) + pow(eq_TS, 2) + pow(freq*em_TS, 2))
#phase_residual_err = np.sqrt(pow(phase_err, 2) + pow(eq_phase, 2) + pow(freq*em_phase, 2))
TS_residual_err = TS_err
phase_residual_err = phase_err

# Computing the weighted mean of the residuals
weighted_mean_TS_residual_nolin = np.average(residual_TS, weights=1/TS_residual_err**2)
weighted_mean_TS_residual_std_nolin = np.sqrt(1 / np.sum(1/TS_residual_err**2))
weighted_mean_phase_residual_nolin = np.average(residual_phase, weights=1/phase_residual_err**2)
weighted_mean_phase_residual_std_nolin = np.sqrt(1 / np.sum(1/phase_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_TS_nolin = np.abs(weighted_mean_TS_residual)/weighted_mean_TS_residual_std
r_residual_phase_nolin = np.abs(weighted_mean_phase_residual)/weighted_mean_phase_residual_std

# Array of element for plotting the wheigted mean of residuals
TS_residual_array_nolin = np.full((N), weighted_mean_TS_residual)
phase_residual_array_nolin = np.full((N), weighted_mean_phase_residual)

# Computing the resonance frequency
f_TS_nolin = a_TS
f_phase_nolin = a_phase

# Computing the resonance frequency error
f_TS_err = ea_TS
f_phase_err = ea_phase

# Plotting the local fit
# defining the plot
fig, ax = plt.subplots(3, 1, figsize=(6.5, 9), sharex=True, constrained_layout=True, height_ratios=[2, 1, 1])

ax[0].set_xscale('log')
ax[1].set_xscale('log')

# defining points and fit function 
ax[0].errorbar(freq,TS,xerr=freq_err, yerr=TS_err, fmt='o', label=r'TS Data',ms=2,color='black')
ax[0].plot(freq_fit, fit_nolin_TS(freq_fit, *popt_TS), label='TS linear fit', linestyle='--', color='red')
ax[0].errorbar(freq,phase,xerr=freq_err, yerr=phase_err, fmt='o', label=r'Phase Data',ms=2,color='blue')
ax[0].plot(freq_fit, fit_nolin_phase_norm(freq_fit, *popt_phase), label='Phase linear fit', linestyle='--', color='blue')

# plotting the residual graph(in this case only y - residuals)
ax[1].errorbar(freq, residual_TS,yerr=TS_residual_err, fmt='o', label=r'TS residual',ms=2,color='black')
ax[2].errorbar(freq, residual_phase, yerr=phase_residual_err, fmt='o', label=r'Phase Residual', ms=2, color='blue')

# plotting the weighted mean of residuals
ax[1].plot(freq, TS_residual_array_nolin, linestyle='--', color='red')
ax[2].plot(freq, phase_residual_array_nolin, linestyle='--', color='blue')

R_ylim = np.std(residual_TS)*5 

# setting limit for y axis and the axis labels
ax[0].set_ylabel(r'$A\, - \, \Phi \, normalized$', size = 15)
ax[1].set_ylabel(r'Residuals', size = 15)
ax[1].set_xlabel(r'$Frequency$ [$KHz$]', size = 15)
ax[2].set_ylabel(r'Phase Residuals', size=15)
ax[2].set_xlabel(r'$Frequency$ [$KHz$]', size=15)

# Dynamic text position for f_TS and f_phase
y_min, y_max = ax[0].get_ylim() 
text_y1 = y_max - 0.70* (y_max - y_min)  # vetical position under 70% 
text_y2 = y_max - 0.558* (y_max - y_min)  # vertical position under 55%

# Plotting some text
ax[0].text(0.75 * ax[0].get_xlim()[1], text_y1,  # Orizontal position 75% 
           r'$f_{{TS}}$ = {fts:.2f} $\pm$ {efts:.2f} $KHz$'.format(fts=a_TS, efts=ea_TS),
           size=13, ha='right')  

ax[0].text(0.75 * ax[0].get_xlim()[1], text_y2,  # Orizontal position 75% 
           r'$f_{{phase}}$ = {fph:.2f} $\pm$ {efph:.3f} $KHz$'.format(fph=a_phase, efph=ea_phase),
           size=13, ha='right') 

# aumatic layout configuration
fig.tight_layout(pad=1, w_pad=0.1, h_pad=0.3)

print(a_TS, " ", ea_TS, " - ", a_phase, " ", ea_phase)
# show the plot to user 
plt.show()

# endregion
