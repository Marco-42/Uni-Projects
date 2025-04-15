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

def fit_orr(x, q):
    return q

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
    if(C < 100e-9 and C > 0):
        return np.sqrt(pow(C*0.01, 2)/3 + pow(0.01e-9*8, 2)/3)

def compatibility(x, y, x_err, y_err):
    """x, y, x_err, y_err"""
    return np.abs(x-y)/np.sqrt(x_err**2 + y_err**2)

# Some variable definitions
F0 = (3.14+3.23)/2
eF0 = (3.23-3.14)/np.sqrt(12)

R = 3.284
C = 15e-9 
eR = computing_R_err(R)
eC = computing_C_err(C)

Ftheory = 1/(2*np.pi*R*C*1000000)
Ftheory_err = np.sqrt((eR/(C*R**2))**2 + (eC/(R*C**2))**2)/(2*np.pi*1000000)

# region - Local linear fit 

# Input file name
inputname1 = 'Local_fit_2.txt'

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
Tscale = 10  # mus

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
phase_err_not_norm = 2*np.pi*(freq*letturaT)/1000
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
chi2_TS = np.sum((residual_TS/TS_err)**2)


# degrees of freedom
df = N - 2

# fitting parameters and errors
m_TS, q_TS = popt_TS 
em_TS, eq_TS = np.sqrt(np.diag(pcov_TS))

# Fitting the phase
popt_phase, pcov_phase = curve_fit(fitf, freq, phase_not_norm, p0=[m_init, q_init], method='lm', sigma=phase_err_not_norm, absolute_sigma=True)

# Computing the residual
residual_phase = phase_not_norm - fitf(freq, *popt_phase)

# variables error and chi2
perr_phase = np.sqrt(np.diag(pcov_phase))
chi2_phase = np.sum((residual_phase/phase_err)**2)

# fitting parameters and errors
m_phase, q_phase = popt_phase*2/np.pi
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
f_phase = (0.5 - q_phase)/m_phase

# Computing the resonance frequency error
f_TS_err = np.sqrt((eq_TS/m_TS)**2 + ((1/np.sqrt(2) - q_TS)*em_TS/(m_TS**2))**2 + 2*((1/np.sqrt(2) - q_TS)/(m_TS**3))*pcov_TS[0, 1])
f_phase_err = np.sqrt((eq_phase/m_phase)**2 + ((1/np.sqrt(2) - q_phase)*em_phase/(m_phase**2))**2 + 2*((1/np.sqrt(2) - q_phase)/(m_phase**3))*pcov_phase[0, 1])

# Plotting the local fit
# defining the plot
fig, ax = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True, constrained_layout=True, height_ratios=[2, 0.5, 0.5])

# defining points and fit function 
ax[0].errorbar(freq,TS,xerr=freq_err, yerr=TS_err, fmt='o', label=r'Amp Data',ms=2,color='darkcyan', zorder = 1, lw =1.5)
ax[0].plot(freq_fit, fitf(freq_fit, *popt_TS), label='Amp linear fit', linestyle='--', color='darkorange', lw = 1, zorder = 0)
ax[0].errorbar(freq,phase,xerr=freq_err, yerr=phase_err, fmt='o', label=r'Phase Data',ms=2,color='black', zorder = 1, lw = 1.5)
ax[0].plot(freq_fit, fitf_norm(freq_fit, *popt_phase), label='Phase linear fit', linestyle='--', color='red', lw = 1, zorder = 0)

# Plotting the legend
ax[0].legend(loc='best', fontsize = 12)

# plotting the residual graph(in this case only y - residuals)
ax[1].errorbar(freq, residual_TS,yerr=TS_residual_err, fmt='o', label=r'Amp residual',ms=2,color='darkcyan', lw = 1.5, zorder = 1)
ax[2].errorbar(freq, residual_phase, yerr=phase_residual_err, fmt='o', label=r'Phase Residual', ms=2, color='black', lw = 1.5, zorder = 1)

# plotting the weighted mean of residuals
ax[1].plot(freq, TS_residual_array, linestyle='-', color='darkorange', zorder = 0)
ax[2].plot(freq, phase_residual_array, linestyle='-', color='red', zorder = 0)

R_ylim = np.std(residual_TS)*5 

# setting limit for y axis and the axis labels
ax[1].set_ylim([-0.05,0.05])
ax[2].set_ylim([-0.007,0.007])
ax[0].set_ylim([0.2,0.8])
ax[0].set_ylabel(r'$|A|\, - \, \Phi_{\text{norm}\rightarrow 1}$', size = 15)
ax[1].set_ylabel(r'$|A|_{\,\text{Residuals}}$', size = 15)
ax[1].set_xlabel(r'$Frequency$ [$KHz$]', size = 13)
ax[0].set_xlabel(r'$Frequency$ [$KHz$]', size = 13)
ax[2].set_xlabel(r'$Frequency$ [$KHz$]', size = 13)
ax[2].set_ylabel(r'$\Phi_{\,\text{Residuals}}$', size=15)
ax[2].set_xlabel(r'$Frequency$ [$KHz$]', size=13)

# Dynamic text position for f_TS and f_phase
y_min, y_max = ax[0].get_ylim() 
text_y1 = y_max - 0.65* (y_max - y_min)  # vetical position under 60% 
text_y2 = y_max - 0.73* (y_max - y_min)  # vertical position under 70%
text_y3 = y_max - 0.81* (y_max - y_min)  # vertical position under 80%
text_y4 = y_max - 0.89* (y_max - y_min)  # vertical position under 90%

# Plotting some text
ax[0].text(0.88 * ax[0].get_xlim()[1], text_y1,  # Orizontal position 88% 
           r'$f_{{Amp}}$ = {fts:.2f} $\pm$ {efts:.2f} $KHz$'.format(fts=f_TS, efts=f_TS_err),
           size=13)  

ax[0].text(0.88 * ax[0].get_xlim()[1], text_y2,  # Orizontal position 88% 
           r'$f_{{phase}}$ = {fph:.1f} $\pm$ {efph:.1f} $KHz$'.format(fph=f_phase, efph=f_phase_err),
           size=13) 

ax[0].text(0.88 * ax[0].get_xlim()[1], text_y3,  # Orizontal position 88% 
           r'$\chi^2_{{\text{{Amp}}}} \, / \, DOF$ = {fph:.1f} / {efph:.0f}'.format(fph=chi2_TS, efph=df),
           size=13) 

ax[0].text(0.88 * ax[0].get_xlim()[1], text_y4,  # Orizontal position 88% 
           r'$\chi^2_{{\text{{Phase}}}} \, / \, DOF$ = {fph:.1f} / {efph:.0f}'.format(fph=chi2_phase, efph=df),
           size=13) 

# aumatic layout configuration
#fig.tight_layout()

plt.savefig('local'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# show the plot to user 
#plt.show()
            
# endregion

# region - Non linear fit
# Input file name
inputname2 = 'Bode.txt'

# Initial parameter values
a_init = 3.27

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
#TS_err = TS*np.sqrt((letturaV/Vin)**2 + (letturaV/Vout)**2 + 2*(errscalaV*TS)**2)
# defining the error of the Transfer function
TS_err = np.sqrt((eVout/Vin)**2 + (Vout*eVin/(Vin**2))**2)

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
chi2_TS_nolin = np.sum((residual_TS/TS_err)**2)

# degrees of freedom
df = N - 1

# fitting parameters and errors
a_TS = popt_TS[0]
ea_TS = np.sqrt(pcov_TS[0, 0])

# Fitting the phase
popt_phase, pcov_phase = curve_fit(fit_nolin_phase_norm, freq, phase, p0=[a_init], method='lm', sigma=phase_err, absolute_sigma=True)

# Computing the residual
residual_phase = phase - fit_nolin_phase_norm(freq, *popt_phase)

# variables error and chi2
perr_phase = np.sqrt(np.diag(pcov_phase))
chi2_phase_nolin = np.sum((residual_phase/phase_err)**2)

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
r_residual_TS_nolin = compatibility(weighted_mean_TS_residual_nolin,0, weighted_mean_TS_residual_std_nolin, 0)
r_residual_phase_nolin = compatibility(weighted_mean_phase_residual_nolin, 0, weighted_mean_phase_residual_std_nolin, 0)

# Array of element for plotting the wheigted mean of residuals
TS_residual_array_nolin = np.full((N), weighted_mean_TS_residual)
phase_residual_array_nolin = np.full((N), weighted_mean_phase_residual)

# Computing the resonance frequency
f_TS_nolin = a_TS
f_phase_nolin = a_phase

# Computing the resonance frequency error
f_TS_nolin_err = ea_TS
f_phase_nolin_err = ea_phase

# Plotting the local fit
# defining the plot
fig, ax = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True, constrained_layout=True, height_ratios=[2, 0.7, 0.7])

ax[0].set_xscale('log')
ax[1].set_xscale('log')

# defining points and fit function 
ax[0].errorbar(freq,TS,xerr=freq_err, yerr=TS_err, fmt='o', label=r'Amp Data',ms=2,color='darkcyan', zorder = 1, lw =1.5)
ax[0].plot(freq_fit, fit_nolin_TS(freq_fit, *popt_TS), label='Amp fit', linestyle='--', color='darkorange', lw = 1, zorder = 0)
ax[0].errorbar(freq,phase,xerr=freq_err, yerr=phase_err, fmt='o', label=r'Phase Data',ms=2,color='black', zorder = 1, lw = 1.5)
ax[0].plot(freq_fit, fit_nolin_phase_norm(freq_fit, *popt_phase), label='Phase fit', linestyle='--', color='red', lw = 1, zorder = 0)

# plotting the residual graph(in this case only y - residuals)
ax[1].errorbar(freq, residual_TS,yerr=TS_residual_err, fmt='o', label=r'Amp residual',ms=2,color='darkcyan', lw = 1.5, zorder = 1)
ax[2].errorbar(freq, residual_phase, yerr=phase_residual_err, fmt='o', label=r'Phase Residual', ms=2,  color='black', lw = 1.5, zorder = 1)

# plotting the weighted mean of residuals
ax[1].plot(freq, TS_residual_array_nolin, linestyle='-', color='darkorange', zorder = 0)
ax[2].plot(freq, phase_residual_array_nolin, linestyle='-', color='red', zorder = 0)

R_ylim = np.std(residual_TS)*5 

# Plotting some text
ax[1].text(0.7, 0.03, r'$\mu_{{\,\text{{residual}}}}$ = {e:.3f} $\pm$ {f:.3f}'.format(e=weighted_mean_TS_residual_nolin, f = weighted_mean_TS_residual_std_nolin), size=12)
ax[1].text(30, 0.03, r'$r_{{\, \mu \, / \, 0}}$ = {e:.1f}'.format(e=r_residual_TS_nolin), size=12)
ax[2].text(0.7, 0.011, r'$\mu_{{\,\text{{residual}}}}$ = {e:.4f} $\pm$ {f:.4f}'.format(e=weighted_mean_phase_residual_nolin, f = weighted_mean_phase_residual_std_nolin), size=12)
ax[2].text(30, 0.011, r'$r_{{\, \mu \, / \, 0}}$ = {e:.1f}'.format(e=r_residual_phase_nolin), size=12)

# setting limit for y axis and the axis labels
ax[1].set_ylim([-0.04,0.06])
ax[2].set_ylim([-0.015,0.025])
ax[0].set_ylim([-0.2,1.02])
ax[0].set_ylabel(r'$|A|\, - \, \Phi_{\text{norm}\rightarrow 1}$', size = 15)
ax[1].set_ylabel(r'$|A|_{\,\text{Residuals}}$', size = 15)
ax[1].set_xlabel(r'$Frequency$ [$KHz$]', size = 13)
ax[0].set_xlabel(r'$Frequency$ [$KHz$]', size = 13)
ax[2].set_xlabel(r'$Frequency$ [$KHz$]', size = 13)
ax[2].set_ylabel(r'$\Phi_{\,\text{Residuals}}$', size=15)
ax[2].set_xlabel(r'$Frequency$ [$KHz$]', size=13)

# Plotting the legend
ax[0].legend(loc='best', fontsize = 12, ncol = 2)

# Dynamic text position for f_TS and f_phase
y_min, y_max = ax[0].get_ylim() 
text_y1 = y_max - 0.25* (y_max - y_min)  # vetical position under 91% 
text_y2 = y_max - 0.35* (y_max - y_min)  # vertical position under 91%
text_y3 = y_max - 0.45* (y_max - y_min)  # vertical position under 91%
text_y4 = y_max - 0.55* (y_max - y_min)  # vertical position under 91%

# Plotting some text
ax[0].text(23, text_y1, r'$f_{{Amp}}$ = {fts:.2f} $\pm$ {efts:.2f} $KHz$'.format(fts=a_TS, efts=ea_TS),size=13)  
ax[0].text(23, text_y2, r'$f_{{phase}}$ = {fph:.3f} $\pm$ {efph:.3f} $KHz$'.format(fph=a_phase, efph=ea_phase),size=13) 
ax[0].text(23, text_y3,  # Orizontal position 88% 
           r'$\chi^2_{{\text{{Amp}}}} \, / \, DOF$ = {fph:.1f} / {efph:.0f}'.format(fph=chi2_TS_nolin, efph=df),
           size=13) 
ax[0].text(23, text_y4,  # Orizontal position 88% 
           r'$\chi^2_{{\text{{Phase}}}} \, / \, DOF$ = {fph:.1f} / {efph:.0f}'.format(fph=chi2_phase_nolin, efph=df),
           size=13) 
# aumatic layout configuration
#fig.tight_layout()

# show the plot to user 
#plt.show()

plt.savefig('non_linear'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - Bode 

# Computing input data for Bode plot
log_freq = np.log10(freq)
log_TS = 20*np.log10(TS)

# Computing y errors for Bode plot
log_TS_err = 20*TS_err/(TS*np.log(10))

# Fitting the last parameters of plot
limit = 1.5 # Limit of frequency for bode validation
popt_log_TS, pcov_log_TS = curve_fit(fitf, log_freq[log_freq > limit], log_TS[log_freq > limit], p0=[-20, 20*np.log10(a_init)], method='lm', sigma=log_TS_err[log_freq > limit], absolute_sigma=True)

# Fitting the first parameters of plot to found the responce of data to low frequency(below the resonance frequency)
limit2 = -0.14
popt_A1, pcov_A1 = curve_fit(fit_orr, log_freq[log_freq < limit2], log_TS[log_freq < limit2], p0=[0], method='lm', sigma=log_TS_err[log_freq < limit2], absolute_sigma=True)

# Taking the parameters from fit
A1q = popt_A1[0]
eA1q = np.sqrt(np.diag(pcov_A1))[0]

# Computing the compatibility beetween 0 and A1q(the line has to be orrizontal)
r_A1 = compatibility(A1q, 0, eA1q, 0)

# Computing the residual 
residual_log_TS = log_TS[log_freq > limit] - fitf(log_freq[log_freq > limit], *popt_log_TS)
residual_A1 = log_TS[log_freq < limit2] - fit_orr(log_freq[log_freq < limit2], *popt_A1)

# defining the x - axis for the plot
log_freq_fit = np.linspace(min(log_freq), max(log_freq), 1000)

# variables error and chi2
perr_log_TS = np.sqrt(np.diag(pcov_log_TS))
chi2_log_TS = np.sum((residual_TS/TS_err)**2)

# degrees of freedom
df = N - 2

# fitting parameters and errors
m_log_TS, q_log_TS = popt_log_TS 
em_log_TS, eq_log_TS = np.sqrt(np.diag(pcov_log_TS))

# calculate the residuals error by quadratic sum using the variance theorem
#TS_residual_err = np.sqrt(pow(TS_err, 2) + pow(eq_TS, 2) + pow(freq*em_TS, 2))
#phase_residual_err = np.sqrt(pow(phase_err, 2) + pow(eq_phase, 2) + pow(freq*em_phase, 2))
TS_log_residual_err = log_TS_err[log_freq > limit]
A1_residual_err = log_TS_err[log_freq < limit2]

# Computing the weighted mean of the residuals
weighted_mean_log_TS_residual = np.average(residual_log_TS, weights=1/TS_log_residual_err**2)
weighted_mean_log_TS_residual_std = np.sqrt(1 / np.sum(1/TS_log_residual_err**2))
weighted_mean_A1_residual = np.average(residual_A1, weights=1/A1_residual_err**2)
weighted_mean_A1_residual_std = np.sqrt(1 / np.sum(1/A1_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_log_TS = np.abs(weighted_mean_log_TS_residual)/weighted_mean_log_TS_residual_std
r_residual_A1 = np.abs(weighted_mean_A1_residual)/weighted_mean_A1_residual_std

# Computing the resonance frequency
f_log_TS_bode1 = 10**(-q_log_TS/m_log_TS) # considering the fit's inclination
f_log_TS_bode2 = 10**(q_log_TS/20) # considering -20 as fit inclination
f_log_TS_bode3 = 10**((A1q - q_log_TS)/m_log_TS) # Considering the A1q value

# Computing errors on resonance frequency
ef_log_TS_bode1 = f_log_TS_bode1*np.log(10)*np.sqrt((eq_log_TS/m_log_TS)**2 + (em_log_TS*q_log_TS/(m_log_TS**2))**2)
ef_log_TS_bode2 = f_log_TS_bode2*np.log(10)*eq_log_TS/20
ef_log_TS_bode3 = f_log_TS_bode3*np.log(10)*np.sqrt((eq_log_TS/m_log_TS)**2 + (em_log_TS*(A1q - q_log_TS)/(m_log_TS**2))**2 + (eA1q/m_log_TS)**2)

# Computing the region of 1 sigma deviation for linear fit
y_fit_upper = fitf(log_freq_fit[log_freq_fit > limit-0.2]+0.1, m_log_TS - em_log_TS, q_log_TS-eq_log_TS)
y_fit_down = fitf(log_freq_fit[log_freq_fit > limit-0.2]+0.1, m_log_TS + em_log_TS, q_log_TS+eq_log_TS)

fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True, constrained_layout=True, height_ratios=[2, 0.7])

# defining points and fit function 
ax[0].errorbar(log_freq,log_TS,xerr=0, yerr = log_TS_err, fmt='o', label=r'Amp Data',ms=2,color='black', zorder = 2, lw =1.5)
ax[0].plot(log_freq_fit, fitf(log_freq_fit, *popt_log_TS), label=r'Linear fit $f \, > \, f_{Amp}$', linestyle='--', color='red', lw = 1, zorder = 1)
ax[0].plot(log_freq_fit, np.full(len(log_freq_fit), *popt_A1), label=r'Linear fit $f \, > \, f_{Amp}$', linestyle='--', color='darkorange', lw = 1, zorder = 1)
ax[0].fill_between(log_freq_fit[log_freq_fit > limit-0.2]+0.1, y_fit_down, y_fit_upper, color='gold', alpha=0.4, label = r'$\pm \, 1\sigma$ deviation', zorder = 0)

# plotting the residual graph(in this case only y - residuals)
ax[1].errorbar(log_freq[log_freq > limit], residual_log_TS,yerr=TS_log_residual_err, fmt='o', label=r'Amp residual',ms=2,color='black', lw = 1.5, zorder = 1)
ax[1].errorbar(log_freq[log_freq < limit2], residual_A1,yerr=A1_residual_err, fmt='o', label=r'TF residual',ms=2,color='black', lw = 1.5, zorder = 1)
ax[1].plot(log_freq_fit[log_freq_fit > limit-0.2]+0.1, np.full(len(log_freq_fit[log_freq_fit > limit-0.2]), weighted_mean_log_TS_residual), linestyle='--', color='red', zorder = 0)
ax[1].plot(log_freq_fit[log_freq_fit < limit2+0.2]-0.1, np.full(len(log_freq_fit[log_freq_fit < limit2+0.2]), weighted_mean_A1_residual), linestyle='--', color='darkorange', zorder = 0)

# Plotting some text
ax[1].text(0.6, 1.2, r'$\mu_{{\,\text{{residual}}}}$ = {e:.1f} $\pm$ {f:.1f}'.format(e=weighted_mean_log_TS_residual, f = weighted_mean_log_TS_residual_std), size=12)
ax[0].text(-0.35, -8, r'$q$ = {e:.1f} $\pm$ {f:.1f}'.format(e=A1q, f = eA1q), size=12)
ax[0].text(-0.35, -11, r'$r_{{\, q \, / \, 0}}$ = {e:.1f}'.format(e=r_A1), size=12)

# setting limit for y axis and the axis labels
ax[0].set_ylabel(r'$|A|\, [\, dB \, ]$', size = 15)
ax[1].set_ylabel(r'$log|A|_{\,\text{Residuals}}$', size = 15)
ax[1].set_xlabel(r'$log\,f$', size = 13)
ax[0].set_xlabel(r'$log\,f$', size = 13)

# Plotting the legend
ax[0].legend(loc='best', fontsize = 13, ncol = 2)

# Dynamic text position for f_TS and f_phase
y_min, y_max = ax[0].get_ylim() 
text_y1 = y_max - 0.60* (y_max - y_min)  # vetical position under 10% 
text_y2 = y_max - 0.68* (y_max - y_min)  # vertical position under 16%
text_y3 = y_max - 0.76* (y_max - y_min)  # vertical position under 22%
text_y4 = y_max - 0.84* (y_max - y_min)  # vertical position under 28%


# Plotting some text
ax[0].text(-0.35, text_y1, r'$f_{{Bode1}}$ = {fts:.1f} $\pm$ {efts:.1f} $KHz$'.format(fts=f_log_TS_bode1, efts=ef_log_TS_bode1),size=13)  
ax[0].text(-0.35, text_y2, r'$f_{{Bode2}}$ = {fph:.1f} $\pm$ {efph:.1f} $KHz$'.format(fph=f_log_TS_bode2, efph=ef_log_TS_bode2),size=13) 
ax[0].text(-0.35, text_y3, r'$f_{{Bode3}}$ = {fph:.1f} $\pm$ {efph:.1f} $KHz$'.format(fph=f_log_TS_bode3, efph=ef_log_TS_bode3),size=13) 
ax[0].text(-0.35, text_y4,  r'$\chi^2_{{\text{{Amp}}}} \, / \, DOF$ = {fph:.1f} / {efph:.0f}'.format(fph=chi2_log_TS, efph=df), size=13) 

ax[0].text(1.9, -18,  r'm = {fph:.1f} $\pm$ {efph:.1f}'.format(fph=m_log_TS, efph=em_log_TS), size=12) 
ax[0].text(1.9, -21, r'$r_{{\, m \, / \, 20}}$ = {e:.1f}'.format(e=compatibility(m_log_TS, -20, em_log_TS, 0)), size=12)
#fig.tight_layout()
#plt.show()

plt.savefig('bode'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - Linearized fit for TF

# Input data after linearization
TSmeno2 = 1/TS**2
TSmeno2err = 2*TS_err/(TS**3)

freq2 = freq**2

# Setting the TF^2 limit for the fit
limit3 = 90000
limit4 = 100

# Computing linear fit 
popt_meno2, pcov_meno2 = curve_fit(fitf, freq2[freq2 < limit3], TSmeno2[freq2 < limit3], p0=[1/a_init**2, 1], method='lm', sigma=TSmeno2err[freq2 < limit3], absolute_sigma=True)
popt_meno2_notall, pcov_meno2_notall = curve_fit(fitf, freq2[freq2 < limit4], TSmeno2[freq2 < limit4], p0=[1/a_init**2, 1], method='lm', sigma=TSmeno2err[freq2 < limit4], absolute_sigma=True)

# Taking the parameters from fit
m_meno2, q_meno2 = popt_meno2
em_meno2, eq_meno2 = np.sqrt(np.diag(pcov_meno2))
m_meno2_notall, q_meno2_notall = popt_meno2_notall
em_meno2_notall, eq_meno2_notall = np.sqrt(np.diag(pcov_meno2_notall))

# Computing the resonance frequency
f_meno2 = 1/np.sqrt(m_meno2)
f_meno2_notall = 1/np.sqrt(m_meno2_notall)
f_meno2_err = em_meno2/(2*m_meno2**(3/2))
f_meno2_notall_err = em_meno2_notall/(2*m_meno2_notall**(3/2))

# Computing compatibily between q_meno2_notall and 1
r_meno2_notall = compatibility(q_meno2_notall, 1, eq_meno2_notall, 0)

# Computing the residual 
residual_meno2_notall= TSmeno2 - fitf(freq2, *popt_meno2_notall)

# degrees of freedom
df = N - 2

# calculate the residuals error by quadratic sum using the variance theorem
TS_meno2_residual_err = TSmeno2err

# Computing chi square
chi2_meno2 = np.sum((residual_meno2_notall[freq2 < limit4]/TS_meno2_residual_err[freq2 < limit4])**2)

# Computing the weighted mean of the residuals
weighted_mean_meno2_residual = np.average(residual_meno2_notall, weights=1/TS_meno2_residual_err**2)
weighted_mean_meno2_residual_std = np.sqrt(1 / np.sum(1/TS_meno2_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_meno2 = compatibility(weighted_mean_meno2_residual, 0, weighted_mean_meno2_residual_std, 0)

# defining the x - axis for the plot
freq2_fit = np.linspace(min(freq2), max(freq2[freq2 < limit3]), 100000)

# Computing the region of 1 sigma deviation for linear fit
y_fit_upper = fitf(freq2_fit, m_meno2_notall - em_meno2_notall, q_meno2_notall-eq_meno2_notall)
y_fit_down = fitf(freq2_fit, m_meno2_notall + em_meno2_notall, q_meno2_notall+eq_meno2_notall)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True, constrained_layout=True, height_ratios=[2, 0.7])

# defining points and fit function 
ax[0].errorbar(freq2,TSmeno2,xerr=0, yerr = TSmeno2err, fmt='o', label=r'Amp Data',ms=2,color='black', zorder = 2, lw =1.5)
ax[0].plot(freq2_fit[freq2_fit < limit3], fitf(freq2_fit[freq2_fit < limit3], *popt_meno2_notall), label=r'Linear fit $f \, < \, f_{Amp}$', linestyle='--', color='blue', lw = 1, zorder = 1)
ax[0].fill_between(freq2_fit, y_fit_down, y_fit_upper, color='skyblue', alpha=0.4, label = r'$\pm \, 1\sigma$ deviation', zorder = 0)

# plotting the residual graph(in this case only y - residuals)
ax[1].errorbar(freq2, residual_meno2_notall,yerr=TS_meno2_residual_err, fmt='o', label=r'Amp residual',ms=2,color='black', lw = 1.5, zorder = 1)
ax[1].plot(freq2_fit, np.full(len(freq2_fit), weighted_mean_meno2_residual), linestyle='--', color='blue', zorder = 0)

# Plotting some text
ax[1].text(0.6, -2500, r'$\mu_{{\,\text{{residual}}}}$ = {e:.2f} $\pm$ {f:.2f}'.format(e=weighted_mean_meno2_residual, f = weighted_mean_meno2_residual_std), size=12)

# setting limit for y axis and the axis labels
ax[0].set_ylabel(r'$\frac{1}{A^2}$', size = 15)
ax[1].set_ylabel(r'$\frac{1}{A^2}_{\,\text{Residuals}}$', size = 15)
ax[1].set_xlabel(r'$f^2$ [$KHz^2$]', size = 13)
ax[0].set_xlabel(r'$f^2$ [$KHz^2$]', size = 13)

# Plotting the legend
ax[0].legend(loc='best', fontsize = 13)

# Plotting some text
ax[0].text(1.1, 1700, r'$f_{{Amp}}$ = {fts:.2f} $\pm$ {efts:.2f} $KHz$'.format(fts=f_meno2_notall, efts=f_meno2_notall_err),size=13)  
ax[0].text(1.1, 500, r'$q$ = {fph:.2f} $\pm$ {efph:.2f} $KHz$'.format(fph=q_meno2_notall, efph=eq_meno2_notall),size=13) 
ax[0].text(1.1, 250, r'$r_{{\text{{q / 1}}}}$ = {fph:.1f}'.format(fph=r_meno2_notall),size=13) 
ax[0].text(1.1, 900,  r'$\chi^2\, / \, DOF$ = {fph:.1f} / {efph:.0f}'.format(fph=chi2_meno2, efph=df), size=13) 

# setting log scale
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')

#fig.tight_layout()

plt.savefig('linear_A'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - Linearized fit for Phase

# Input data after linearization
phase_lin = np.tan(phase_not_norm)
phase_lin_err = phase_err_not_norm/np.cos(phase_not_norm)**2

# Setting the TF^2 limit for the fit
limit3 = 300
limit4 = 50

# Computing linear fit 
popt_phase_lin, pcov_phase_lin = curve_fit(fitf, freq[freq < limit3], phase_lin[freq < limit3], p0=[1/a_init, 0], method='lm', sigma=phase_lin_err[freq < limit3], absolute_sigma=True)
popt_phase_lin_notall, pcov_phase_lin_notall = curve_fit(fitf, freq[freq < limit4], phase_lin[freq < limit4], p0=[1/a_init, 0], method='lm', sigma=phase_lin_err[freq < limit4], absolute_sigma=True)

# Taking the parameters from fit
m_phase_lin, q_phase_lin = popt_phase_lin
em_phase_lin, eq_phase_lin = np.sqrt(np.diag(pcov_phase_lin))
m_phase_lin_notall, q_phase_lin_notall = popt_phase_lin_notall
em_phase_lin_notall, eq_phase_lin_notall = np.sqrt(np.diag(pcov_phase_lin_notall))

# Computing the resonance frequency
f_phase_lin = 1/m_phase_lin
f_phase_lin_notall = 1/m_phase_lin_notall
f_phase_lin_err = em_phase_lin/(m_phase_lin**2)
f_phase_lin_notall_err = em_phase_lin_notall/(m_phase_lin_notall**2)

# Computing compatibily between q_phase_lin_notall and 0
r_phase_lin_notall = compatibility(q_phase_lin_notall, 0, eq_phase_lin_notall, 0)

# Computing the residual 
residual_phase_lin_notall= phase_lin - fitf(freq, *popt_phase_lin_notall)

# degrees of freedom
df = N - 2

# calculate the residuals error by quadratic sum using the variance theorem
phase_lin_residual_err = phase_lin_err

# Computing chi square
chi2_phase_lin = np.sum((residual_phase_lin_notall[freq < limit4]/phase_lin_residual_err[freq < limit4])**2)

# Computing the weighted mean of the residuals
weighted_mean_phase_lin_residual = np.average(residual_phase_lin_notall, weights=1/phase_lin_residual_err**2)
weighted_mean_phase_lin_residual_std = np.sqrt(1 / np.sum(1/phase_lin_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_phase_lin = compatibility(weighted_mean_phase_lin_residual, 0, weighted_mean_phase_lin_residual_std, 0)

# defining the x - axis for the plot
freq_fit = np.linspace(min(freq), max(freq[freq < limit3]), 100000)

# Computing the region of 1 sigma deviation for linear fit
y_fit_upper = fitf(freq_fit, m_phase_lin_notall - em_phase_lin_notall, q_phase_lin_notall-eq_phase_lin_notall)
y_fit_down = fitf(freq_fit, m_phase_lin_notall + em_phase_lin_notall, q_phase_lin_notall+eq_phase_lin_notall)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True, constrained_layout=True, height_ratios=[2, 0.7])

# defining points and fit function 
ax[0].errorbar(freq,phase_lin,xerr=0, yerr = phase_lin_err, fmt='o', label=r'Phase Data',ms=2,color='black', zorder = 2, lw =1.5)
ax[0].plot(freq_fit[freq_fit < limit3], fitf(freq_fit[freq_fit < limit3], *popt_phase_lin_notall), label=r'Linear fit $f \, < \, f_{TF}$', linestyle='--', color='blue', lw = 1, zorder = 1)
#ax[0].fill_between(freq_fit, y_fit_down, y_fit_upper, color='skyblue', alpha=0.4, label = r'$\pm \, 1\sigma$ deviation', zorder = 0)

# plotting the residual graph(in this case only y - residuals)
ax[1].errorbar(freq, residual_phase_lin_notall,yerr=phase_lin_residual_err, fmt='o', label=r'TF residual',ms=2,color='black', lw = 1.5, zorder = 1)
ax[1].plot(freq_fit, np.full(len(freq_fit), weighted_mean_phase_lin_residual), linestyle='--', color='blue', zorder = 0)

# Plotting some text
ax[1].text(1, 50, r'$\mu_{{\,\text{{residual}}}}$ = {e:.4f} $\pm$ {f:.4f}'.format(e=weighted_mean_phase_lin_residual, f = weighted_mean_phase_lin_residual_std), size=12)


# setting limit for y axis and the axis labels
ax[0].set_ylabel(r'$tan(\Phi)$', size = 15)
ax[1].set_ylabel(r'$tan(\Phi)_{\,\text{Residuals}}$', size = 15)
ax[1].set_xlabel(r'$Frequency$ [$KHz$]', size = 13)
ax[0].set_xlabel(r'$Frequency$ [$KHz$]', size = 13)

# Plotting the legend
ax[0].legend(loc='best', fontsize = 13)

# Plotting some text
ax[0].text(1.1, 20, r'$f_{{TF}}$ = {fts:.3f} $\pm$ {efts:.3f} $KHz$'.format(fts=f_phase_lin_notall, efts=f_phase_lin_notall_err),size=13)  
#ax[0].text(1.1, 30, r'$q$ = {fph:.2f} $\pm$ {efph:.2f} $KHz$'.format(fph=q_phase_lin_notall, efph=eq_phase_lin_notall),size=13) 
#ax[0].text(1.1, 18, r'$r_{{\text{{q\, / \, 1}}}}$ = {fph:.1f}'.format(fph=r_phase_lin_notall),size=13) 
ax[0].text(1.1, 12,  r'$\chi^2 \, / \, DOF$ = {fph:.1f} / {efph:.0f}'.format(fph=chi2_phase_lin, efph=df), size=13) 

# setting log scale
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')

#fig.tight_layout()

plt.savefig('linear_phase'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - Results plot of resonance frequency

# Creating the list with frequency results
# puctual, local TF, local phase, TF non-linear, phase non-linear, bode1, bode2, bode3, linearized TF, linearized phase
result_freq = [F0, f_TS, f_phase, f_TS_nolin, f_phase_nolin, f_log_TS_bode1, f_log_TS_bode2, f_log_TS_bode3, f_meno2_notall, f_phase_lin_notall]
result_freq_err = [eF0, f_TS_err, f_phase_err, f_TS_nolin_err, f_phase_nolin_err, ef_log_TS_bode1, ef_log_TS_bode2, ef_log_TS_bode3, f_meno2_notall_err, f_phase_lin_notall_err]

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), sharex=True, constrained_layout=True)

# Defining the x axis for the plot
IDresult = np.linspace(1, len(result_freq), num = len(result_freq))

# defining points and fit function 
ax.errorbar(IDresult,result_freq,xerr=0, yerr = result_freq_err, fmt='o', label=r'Results', color='firebrick', zorder = 2, lw =1.5, markersize=6)
ax.axhline(y=Ftheory, color='black', linestyle='--', linewidth=1.7, label=r'$f_{\text{theor}}$')
plt.fill_between(np.linspace(0, len(result_freq)+1, num = len(result_freq)), Ftheory-Ftheory_err, Ftheory+Ftheory_err, color='silver', alpha=0.4, label = r'$f_{\text{theor - uniform distribution}}$')

# setting limit for y axis and the axis labels
ax.set_ylabel(r'$Cutoff \, \, frequency \, [KHz]$', size = 15)
ax.set_xlabel(r'$ID_{\text{measure}}$', size = 13)
ax.set_xlim([0 ,len(IDresult) + 1])
ax.set_ylim([2.0, 4.1])

# Plotting the legend
ax.legend(loc='upper left', fontsize = 13)

# Add text under each point
# puctual, local TF, local phase, TF non-linear, phase non-linear, bode1, bode2, bode3, linearized TF, linearized phase
text = [r'$A_{\text{Puct}}$', r'$A_{\text{loc}}$', r'$\Phi_{\text{loc}}$', r"$A_{\text{nl}}$",
        r"$\Phi_{\text{nl}}$", r"$Bode_1$", r"$Bode_2$", r"$Bode_3$",
        r"$A_{\text{lin}}$", r"$\Phi_{\text{lin}}$"]

ax.text(0.65, 3, text[0], fontsize=13)
ax.text(1.8, 3, text[1], fontsize=13)
ax.text(2.75, 3, text[2], fontsize=13)
ax.text(3.85, 3, text[3], fontsize=13)
ax.text(4.8, 3, text[4], fontsize=13)

ax.text(5.4, 2.1, text[5], fontsize=12)
ax.text(6.55, 2.1, text[6], fontsize=12)
ax.text(7.8, 2.1, text[7], fontsize=12)

ax.text(8.8, 3, text[8], fontsize=13)
ax.text(9.8, 3, text[9], fontsize=13)

#fig.tight_layout()

plt.savefig('result_f'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - Results plot of capacitance

# Creating the list with frequency results
# puctual, local TF, local phase, TF non-linear, phase non-linear, bode1, bode2, bode3, linearized TF, linearized phase
result_C = 1000/(2*R*np.array(result_freq)*np.pi)
result_C_err = 1000*np.sqrt((eR/(np.array(result_freq)*R**2))**2 + (result_freq_err/(R*(np.array(result_freq))**2))**2)/(2*np.pi)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), sharex=True, constrained_layout=True)

# Defining the x axis for the plot
IDresultC = np.linspace(1, len(result_C), num = len(result_C))

# defining points and fit function 
ax.errorbar(IDresultC,result_C,xerr=0, yerr = result_C_err, fmt='o', label=r'Results', color='darkblue', zorder = 2, lw =1.5, markersize=6)
ax.axhline(y=C*1000000000, color='black', linestyle='--', linewidth=1.7, label=r'$f_{\text{theor}}$')
plt.fill_between(np.linspace(0, len(result_C)+1, num = len(result_C)), (C-eC)*1000000000, (C+eC)*1000000000, color='silver', alpha=0.4, label = r'$f_{\text{theor - uniform distribution}}$')

# setting limit for y axis and the axis labels
ax.set_ylabel(r'$Capacitance \, [nF]$', size = 15)
ax.set_xlabel(r'$ID_{\text{measure}}$', size = 13)
ax.set_xlim([0 ,len(IDresultC) + 1])
ax.set_ylim([9, 22])

# Plotting the legend
ax.legend(loc='upper left', fontsize = 13)

# Add text under each point
# puctual, local TF, local phase, TF non-linear, phase non-linear, bode1, bode2, bode3, linearized TF, linearized phase
text = [r'$A_{\text{Puct}}$', r'$A_{\text{loc}}$', r'$\Phi_{\text{loc}}$', r"$A_{\text{nl}}$",
        r"$\Phi_{\text{nl}}$", r"$Bode_1$", r"$Bode_2$", r"$Bode_3$",
        r"$A_{\text{lin}}$", r"$\Phi_{\text{lin}}$"]

ax.text(0.65, 11, text[0], fontsize=13)
ax.text(1.8, 11, text[1], fontsize=13)
ax.text(2.75, 11, text[2], fontsize=13)
ax.text(3.85, 11, text[3], fontsize=13)
ax.text(4.8, 11, text[4], fontsize=13)

ax.text(5.4, 11, text[5], fontsize=12)
ax.text(6.55, 11, text[6], fontsize=12)
ax.text(7.8, 11, text[7], fontsize=12)

ax.text(8.8, 11, text[8], fontsize=13)
ax.text(9.8, 11, text[9], fontsize=13)

#fig.tight_layout()

plt.savefig('result_C'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion
