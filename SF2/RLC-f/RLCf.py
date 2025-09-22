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
from matplotlib.ticker import FuncFormatter

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

def fitf(x, A, B, C):
    """A -> offset, B -> OMEGA, C -> Q-value"""
    omega = 2.0 * np.pi * x * 1000  # input in kHz
    fitval = A / np.sqrt(1+C**2*(omega/B-B/omega)**2)
    return fitval

def fitf_phase(x, A, B, C):
    """A -> offset, B -> OMEGA, C -> Q-value"""
    omega = 2.0 * np.pi * x * 1000  # input in kHz
    fitval =  A + np.arctan(C*(-B/omega + omega/B))
    return fitval

def fitf_C(x, A, B, C):
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
    fitval = A / np.sqrt((1-omega**2/B**2)**2+(1/C**2)*omega**2/B**2)
    return fitval

def fitf_phase_C(x, A, B, C):
    return fitf_phase(x, A, B, C)

def fitchi2(i,j,k):
    x = freq
    y = T
    y_err = eT
    AA,BB,CC = A_chi[i],B_chi[j],C_chi[k]
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
#    residuals = (y -  fitf_R(x,AA,BB,CC))  # Seleziono fit su R
    residuals = (y -  fitf(x,AA,BB,CC))  # Seleziono fit su C
    chi2 = np.sum((residuals/y_err)**2)
    mappa[i,j,k] = chi2

def fitchi2_phase(i,j,k):
    x = freq
    y = phase
    y_err = ephase
    AA,BB,CC = A_chi[i],B_chi[j],C_chi[k]
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
#    residuals = (y -  fitf_R(x,AA,BB,CC))  # Seleziono fit su R
    residuals = (y -  fitf_phase(x,AA,BB,CC))  # Seleziono fit su C
    chi2 = np.sum((residuals/y_err)**2)
    mappa[i,j,k] = chi2

def fitchi2_C(i,j,k):
    x = freq
    y = T
    y_err = eT
    AA,BB,CC = A_chi[i],B_chi[j],C_chi[k]
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
#    residuals = (y -  fitf_R(x,AA,BB,CC))  # Seleziono fit su R
    residuals = (y -  fitf_C(x,AA,BB,CC))  # Seleziono fit su C
    chi2 = np.sum((residuals/y_err)**2)
    mappa[i,j,k] = chi2

def fitchi2_phase_C(i,j,k):
    x = freq
    y = phase
    y_err = ephase
    AA,BB,CC = A_chi[i],B_chi[j],C_chi[k]
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
#    residuals = (y -  fitf_R(x,AA,BB,CC))  # Seleziono fit su R
    residuals = (y -  fitf_phase_C(x,AA,BB,CC))  # Seleziono fit su C
    chi2 = np.sum((residuals/y_err)**2)
    mappa[i,j,k] = chi2

def profi2D(axis,matrix3D):
    if axis == 1 :
        mappa2D = np.array([[np.min(mappa[:,b,c]) for b in range(step)] for c in range(step)])
    if axis == 2 :
        mappa2D = np.array([[np.min(mappa[a,:,c]) for a in range(step)] for c in range(step)])
    if axis == 3 :
        mappa2D = np.array([[np.min(mappa[a,b,:]) for a in range(step)] for b in range(step)])
    return mappa2D

def profi1D(axis, mappa):
    if 1 in axis :
        mappa2D = np.array([[np.min(mappa[:,b,c]) for b in range(step)] for c in range(step)])
#        print('1')
        if 2 in axis:
            mappa1D = np.array([np.min(mappa2D[b,:]) for b in range(step)])
#            print('2')
        if 3 in axis:
            mappa1D = np.array([np.min(mappa2D[:,c]) for c in range(step)])
#            print('3')
    else :
#        print('2-3')
        mappa2D = np.array([[np.min(mappa[a,:,c]) for a in range(step)] for c in range(step)])
        mappa1D = np.array([np.min(mappa2D[a,:]) for a in range(step)])
    return mappa1D

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

def format_xticks(x, pos):
    return f'{x:.4g}'

# Number of bins for parameter scan
NI = 20
NJ = 20
NK = 20

# Some variable definitions

Rx = 12.07 #Ohm
RL = 11.58 #Ohm
Rg = 49.5 #Ohm
R = Rx + RL #Ohm
Rtot = R + Rg #Ohm
eRtot = np.sqrt(computing_R_err(R)**2 + (0.5)**2) #Ohm
C = 1.05e-9 #Farad
L = 445.4771548344286e-6 #Henry
eL = 0.8123200424814587e-6 #Henry

print("Rx = ", Rx, " +- ", computing_R_err(Rx), " Ohm")
print("RL = ", RL, " +- ", computing_R_err(RL), " Ohm")
print("R = Rx + RL = ", R, " +- ", computing_R_err(R), " Ohm")
print("C = ", C*1000000000, " +- ", computing_C_err(C)*1000000000, " F")
eR = computing_R_err(R)
eC = computing_C_err(C)

# Defining step for each parameter 
step = 100

#-------------------------------------------------------------
# RESISTANCE
#-------------------------------------------------------------

# region - Data acquisition
# Input file name
inputname1 = 'data.txt'

# load sperimental data from file
data = np.loadtxt(inputname1).T
freq = np.array(data[0])
Vin = np.array(data[1])
Vin_scale = np.array(data[2])
Vout = np.array(data[3])
Vout_scale = np.array(data[4])
time = np.array(data[5])

# Defining time scale
Tscale = 0.5 #microseconds

# Computing errors on each data type
# Assumed reading errors --> triangolar distribution applied as maximum error on 1/25 of division
letturaV_out = Vout_scale*2/(np.sqrt(24)*25)
letturaV_in = Vin_scale*2/(np.sqrt(24)*25)
letturaT = Tscale*2/(np.sqrt(24)*25)
errscalaV = 3/100

eVin = np.sqrt((letturaV_in)**2 + ((errscalaV * Vin)**2))
eVout = np.sqrt((letturaV_out)**2 + ((errscalaV * Vout)**2))

# Computing amplification and error of amplification
T = Vout/Vin 
eT = T*np.sqrt((letturaV_out/Vout)**2 + (letturaV_in/Vin)**2 + 2*(errscalaV**2))

# Computing phase and error of phase
phase = 2*np.pi*freq*time/1000 
ephase = 2*np.pi*freq*letturaT/1000

# Number of points
N = len(freq[freq > 0])

# endregion

# region - Plotting Vout and Vin distribution

fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5),sharex=True)
ax.errorbar(freq,Vin,yerr=eVin, fmt='o', label=r'$V_{\text{in}}$',ms=2, color = 'navy')
ax.errorbar(freq,Vout,yerr=eVout, fmt='o', label=r'$V_{\text{out}}$',ms=2.5, color = 'deepskyblue')
ax.legend(prop={'size': 14}, loc='best')
ax.set_ylabel(r'Voltage $[\, V \,]$', size = 13)
ax.set_xlabel(r'Frequency $[\, kHz \,]$', size = 13)

plt.savefig('Vin_Vout'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

# region - Plotting the amplification and phase distribution

fig, ax1 = plt.subplots(1, 1, figsize=(5.5, 5.5),sharex=True)

ax1.errorbar(freq,T,yerr=eT, fmt='o', label=r'Amplification data',ms=2,color='darkred')
ax1.set_ylabel(r'$|A| \, = \, \frac{V_{\text{out}}}{V_{\text{in}}}$', size = 13)
ax1.set_xlabel(r'Frequency $[\, kHz \, ]$', size = 13)
ax1.yaxis.set_ticks_position('left')
ax1.yaxis.set_label_position('left')
ax1.set_xlim(198, 266)

# instantiate a second Axes that shares the same x-axis
ax2 = ax1.twinx() 

ax2.set_ylabel(r'Phase', size = 13)
ax2.errorbar(freq, phase, yerr=ephase, fmt='o', label=r'$phase$',ms=2.5,color='orange')
ax2.tick_params(axis='y')
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.set_xlim(198, 266)

# Plotting the legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
plt.legend(lines, labels, loc='best', prop={'size': 14})


plt.savefig('Amp_phase_no_fit'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

#endregion 

# region - Amplification's non-linear fit

# Computing the initial parameter values from experimental data 
Q_init = np.sqrt(L/C)/R

Omega_init = 1/np.sqrt(L*C)
A_init = 1

# Computing the fitting using LM algorithm
popt_T, pcov_T = curve_fit(fitf, freq, T, p0=[A_init, Omega_init, Q_init], method='lm', sigma=eT, absolute_sigma=True)

# Computing the residual 
residual_T = T - fitf(freq, *popt_T)

# variables error and chi2
perr_T = np.sqrt(np.diag(pcov_T))
chi2_T = np.sum((residual_T/eT)**2)

# degrees of freedom
df = N - 3

# founding parameters and errors
# A --> offset, Omega --> 1/sqrt(LC), Q --> R*sqrt(C/L) - Q-value
# fitting parameters and errors
A_T, Omega_T, Q_T = popt_T #fit parameters
eA_T, eOmega_T, eQ_T = perr_T #fit errors

Q_T_R = Q_T 
eQ_T_R = eQ_T

# Computing resonance frequency and band interval
f_T = Omega_T/(2*np.pi)
f_T_err = eOmega_T/(2*np.pi)
delta_f_T = Omega_T/(2*np.pi*Q_T)
delta_f_T_err = np.sqrt((eOmega_T/Q_T)**2 + (Omega_T*eQ_T/Q_T**2)**2 - 2*(1/Q_T)*(Omega_T/Q_T**2)*pcov_T[1, 2])/(2*np.pi)

# calculate the residuals error by quadratic sum using the variance theorem
T_residual_err = eT

# Computing the weighted mean of the residuals
weighted_mean_T_residual = np.average(residual_T, weights=1/T_residual_err**2)
weighted_mean_T_residual_std = np.sqrt(1 / np.sum(1/T_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_T = compatibility(weighted_mean_T_residual,0, weighted_mean_T_residual_std, 0)

# Computing the array for fit points
fit_freq = np.linspace(min(freq), max(freq), 1000)

# Array of element for plotting the wheigted mean of residuals
T_residual_array = np.full(len(fit_freq), weighted_mean_T_residual)

# Computing the region of 1 sigma deviation for linear fit
y_fit_upper_T = fitf(fit_freq, *popt_T + perr_T)
y_fit_down_T = fitf(fit_freq, *popt_T - perr_T)

# endregion

# region - Phase's non-linear fit

# Computing the fitting using LM algorithm
popt_phase, pcov_phase = curve_fit(fitf_phase, freq, phase, p0=[A_init, Omega_init, Q_init], method='lm', sigma=ephase, absolute_sigma=True)

# Computing the residual 
residual_phase = phase - fitf_phase(freq, *popt_phase)

# variables error and chi2
perr_phase = np.sqrt(np.diag(pcov_phase))
chi2_phase = np.sum((residual_phase/ephase)**2)

# degrees of freedom
df = N - 3

# founding parameters and errors
# A --> offset, Omega --> 1/sqrt(LC), Q --> R*sqrt(C/L) - Q-value
# fitting parameters and errors
A_phase, Omega_phase, Q_phase = popt_phase #fit parameters
eA_phase, eOmega_phase, eQ_phase = perr_phase #fit errors

Q_phase_R = Q_phase
eQ_phase_R = eQ_phase

# Computing resonance frequency and band interval
f_phase = Omega_phase/(2*np.pi)
f_phase_err = eOmega_phase/(2*np.pi)
delta_f_phase = Omega_phase/(2*np.pi*Q_phase)
delta_f_phase_err = np.sqrt((eOmega_phase/Q_phase)**2 + (Omega_phase*eQ_phase/Q_phase**2)**2 - 2*(1/Q_phase)*(Omega_phase/Q_phase**2)*pcov_phase[1, 2])/(2*np.pi)

# calculate the residuals error by quadratic sum using the variance theorem
phase_residual_err = ephase

# Computing the weighted mean of the residuals
weighted_mean_phase_residual = np.average(residual_phase, weights=1/phase_residual_err**2)
weighted_mean_phase_residual_std = np.sqrt(1 / np.sum(1/phase_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_phase = compatibility(weighted_mean_phase_residual,0, weighted_mean_phase_residual_std, 0)

# Array of element for plotting the wheigted mean of residuals
phase_residual_array = np.full(len(fit_freq), weighted_mean_phase_residual)

# Computing the region of 1 sigma deviation for linear fit
y_fit_upper = fitf_phase(fit_freq, *popt_phase + perr_phase)
y_fit_down = fitf_phase(fit_freq, *popt_phase - perr_phase)

# endregion

# region - Amplification residual plot

# Computing the region of 1 sigma deviation for weighted mean fit
y_fit_upper = np.full(len(fit_freq), weighted_mean_T_residual + weighted_mean_T_residual_std)
y_fit_down = np.full(len(fit_freq), weighted_mean_T_residual - weighted_mean_T_residual_std)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 6),sharex=True)
ax.errorbar(freq,residual_T,yerr=T_residual_err, fmt='o', label=r'Amplification residuals',ms=3,color='black', zorder = 2, lw = 1.5)
ax.plot(fit_freq, T_residual_array, label=r'Residual weighted mean', color='blue', lw = 1.5, zorder = 1)
ax.fill_between(fit_freq, y_fit_down, y_fit_upper, color='silver', alpha=0.5, label = r'Weighted mean $1\sigma$ deviation', zorder = 0, lw = 0)

ax.set_ylim(-0.023, 0.0305)

# Plotting some text about residuals weighted mean and compatibility with zero
ax.text(210, -0.0165, r'$\mu_{{\,\text{{residual}}}}$ = {e:.3f} $\pm$ {f:.3f} $mV$'.format(e=weighted_mean_T_residual, f = weighted_mean_T_residual_std), size=15)
ax.text(210, -0.019, r'$r_{{\, \mu \, / \, 0}}$ = {e:.1f}'.format(e= r_residual_T), size=15)

ax.set_ylabel(r'$Residual$', size = 17)
ax.set_xlabel(r'Frequency $[\, kHz \, ]$', size = 17)
ax.legend(prop={'size': 15}, loc='upper left')

plt.savefig('amp_residuals'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()
# endregion

# region - Phase residual plot

# Computing the region of 1 sigma deviation for weighted mean fit
y_fit_upper = np.full(len(fit_freq), weighted_mean_phase_residual + weighted_mean_phase_residual_std)
y_fit_down = np.full(len(fit_freq), weighted_mean_phase_residual - weighted_mean_phase_residual_std)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 6),sharex=True)
ax.errorbar(freq,residual_phase,yerr=phase_residual_err, fmt='o', label=r'Phase residuals',ms=3,color='black', zorder = 2, lw = 1.5)
ax.plot(fit_freq, phase_residual_array, label=r'Residual weighted mean', color='tab:red', lw = 1.5, zorder = 1)
ax.fill_between(fit_freq, y_fit_down, y_fit_upper, color='gold', alpha=0.5, label = r'Weighted mean $1\sigma$ deviation', zorder = 0, lw = 0)

ax.set_ylim(-0.045, 0.05)

# Plotting some text about residuals weighted mean and compatibility with zero
ax.text(210, -0.036, r'$\mu_{{\,\text{{residual}}}}$ = {e:.3f} $\pm$ {f:.3f} $mV$'.format(e=weighted_mean_phase_residual, f = weighted_mean_phase_residual_std), size=15)

ax.set_ylabel(r'$Residual$', size = 17)
ax.set_xlabel(r'Frequency $[\, kHz \, ]$', size = 17)
ax.legend(prop={'size': 15}, loc='upper left')

plt.savefig('phase_residuals'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()
# endregion

# region - Manual chi square minimization for amplification

# Define the interval for parameter limits
NSI = 3 # number of sigma for chi square map
A0, A1 = A_T - NSI * eA_T, A_T + NSI * eA_T # A = A
B0, B1 = Omega_T - NSI * eOmega_T, Omega_T + NSI * eOmega_T # B = Omega
C0, C1 = Q_T - NSI * eQ_T, Q_T + NSI * eQ_T # C = Q value

# Computing chi square map 

# array for each parameter
A_chi = np.linspace(A0,A1,step)
B_chi = np.linspace(B0,B1,step)
C_chi = np.linspace(C0,C1,step)

# 3d matrix for chi square map
mappa = np.zeros((step,step,step))

item = [(i,j,k) for i in range(step) for j in range(step) for k in range(step)]

# assign the global pool
pool = multiprocessing.pool.ThreadPool(100)

# issue top level tasks to pool and wait
pool.starmap(fitchi2, item, chunksize=10)

# close the pool
pool.close()

# return the chi square 3D map
mappa = np.asarray(mappa)            

# founding the minimum of chi square map and his position
chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa),mappa.shape)

chi2_min_T = chi2_min

# computing residual and chi square value using the manual chi square minimization
residui_chi2 = T - fitf(freq,A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]])

chisq_res = np.sum((residui_chi2/eT)**2)

# endregion

# region - Chi square profilation for amplification [Omega and Q-value]

# Computing 2D profilation on Omega and Q-value
chi2D = profi2D(1,mappa)

# Computing 1D profilation 
prof_B = profi1D([1,3],mappa)
prof_C = profi1D([1,2],mappa)   
prof_A = profi1D([2,3],mappa)    

# Computing the 1 sigma deviation error
lvl = chi2_min+1. #2.3 for 2 sigma, 3.8 for 3 sigma
diff_B = abs(prof_B-lvl)
diff_C = abs(prof_C-lvl)
diff_A = abs(prof_A-lvl)

B_dx = np.argmin(diff_B[B_chi<Omega_T]) 
B_sx = np.argmin(diff_B[B_chi>Omega_T])+len(diff_B[B_chi<Omega_T]) 
C_dx = np.argmin(diff_C[C_chi<Q_T])
C_sx = np.argmin(diff_C[C_chi>Q_T])+len(diff_C[C_chi<Q_T])
A_dx = np.argmin(diff_A[A_chi<A_T])
A_sx = np.argmin(diff_A[A_chi>A_T])+len(diff_A[A_chi<A_T])


# Computing non symmetric error on Parameters
errA = A_chi[argchi2_min[0]]-A_chi[A_dx]
errAA = A_chi[A_sx]-A_chi[argchi2_min[0]]
errB = B_chi[argchi2_min[1]]-B_chi[B_dx] 
errBB = B_chi[B_sx] -B_chi[argchi2_min[1]]
errC = C_chi[argchi2_min[2]]-C_chi[C_dx]
errCC = C_chi[C_sx]-C_chi[argchi2_min[2]]


# Printing fit results 
print("============== BEST FIT AMPLIFICATION - R ==========================")
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]] * 1e-3, e=errB * 1e-3, f=errBB* 1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) '.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC))
print(r'chisq = {m:.2f}'.format(m=np.min(mappa)))
print("====================================================================")

# Extracting the best parameters
BT, eBTdx, eBTsx = B_chi[argchi2_min[1]], errB, errBB
AT, eATdx, eATsx = A_chi[argchi2_min[0]], errA, errAA
CT, eCTdx, eCTsx = C_chi[argchi2_min[2]], errC, errCC

# Plotting the chi square map
cmap = mpl.colormaps['viridis'].reversed()
level = np.linspace(np.min(chi2D),np.max(chi2D),100)
line_c = 'gray'


# Plot definition
fig, ax = plt.subplots(2, 2, figsize=(8, 8),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
fig.suptitle(r'$\chi^2 \left(\omega_0, Q \right)$ - Amplification - R', size = 20)

# defining smooth plot of chi square values in function of two parameteres
im = ax[0,1].contourf(B_chi,C_chi,chi2D, levels=level, cmap=cmap) 

# plotting the legend's bar of chi square value(and associated color)
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 

# Plotting constant chi-square elliptical orbit at Xmin + 1, + 2.3, + 3.8
CS = ax[0, 1].contour(B_chi, C_chi, chi2D, levels=[chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
                      linewidths=2, colors=['red','darkorange', 'saddlebrown'], alpha=1, linestyles='-')

# Add labels directly to the legend
handles, _ = CS.legend_elements()
labels = [r'$\chi^2_{{min}} + 1$', r'$\chi^2_{{min}} + 2.3$', r'$\chi^2_{{min}} + 3.8$']

# lotting chi square value of elliptical orbits
ax[0,1].clabel(CS, inline=True, fontsize=11, fmt='%0.1f')

# plotting deviation bars of parameters
ax[0,1].plot([B0,B1],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B0,B1],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_sx],B_chi[B_sx]],[C0,C1],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_dx],B_chi[B_dx]],[C0,C1],color=line_c, ls='dashed', lw = 1.5)

# plotting best parameters
ax[0,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[C0,C1], ls='dashed', lw = 1.5, color = 'black')
ax[0,1].plot([B0,B1],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black', label = 'Best fit')

#plotting the parabolic curve for Q(also standard dev and best parameter)
ax[0,0].plot(prof_B,C_chi,ls='-')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_sx],C_chi[B_sx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_dx],C_chi[B_dx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black')
ax[0,0].set_xticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])
#ax[0,0].text(chi2_min - 7, C_chi[argchi2_min[2]],r'{e:.1f}'.format(e=C_chi[argchi2_min[2]]), color='k',alpha=1, fontsize=10)
#ax[0,0].text(chi2_min - 7, C_chi[B_dx],r'{e:.1f}'.format(e=C_chi[B_dx] - C_chi[argchi2_min[2]]), color='b',alpha=1, fontsize=10)
#ax[0,0].text(chi2_min - 7, C_chi[B_sx],r'{e:.1f}'.format(e=C_chi[B_sx] - C_chi[argchi2_min[2]]), color='r',alpha=1, fontsize=10)

# plotting the parabolic curve for omega(also standard dev and best parameter)
ax[1,1].plot(B_chi,prof_C)
ax[1,1].plot([B_chi[C_sx],B_chi[C_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[C_dx],B_chi[C_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[int(chi2_min-2),int(chi2_min+10)], ls='dashed', lw = 1.5, color = 'black')
#ax[1,1].text(B_chi[argchi2_min[1]]-400, chi2_min - 6.4,r'{e:.4e} $Hz$'.format(e=B_chi[argchi2_min[1]]), color='k',alpha=1, fontsize=10)
#ax[1,1].text(B_chi[C_dx]-620, chi2_min +4.3,r'{e:.0e} $Hz$'.format(e=-B_chi[argchi2_min[1]] + B_chi[C_dx]), color='b',alpha=1, fontsize=10)
#ax[1,1].text(B_chi[C_sx]+80, chi2_min +4.3,r'{e:.0e} $Hz$'.format(e= -B_chi[argchi2_min[1]] + B_chi[C_sx]), color='r',alpha=1, fontsize=10)
ax[1,1].set_yticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])

# Plotting chi square + 1, + 2.3, + 3.8 level in the lateral graph ax[0, 0]
ax[0, 0].vlines(x=chi2_min + 1, ymin=min(C_chi), ymax=max(C_chi), colors='red', linestyles='-', linewidth=1.2)
ax[0, 0].vlines(x=chi2_min + 2.3, ymin=min(C_chi), ymax=max(C_chi), colors='darkorange', linestyles='-', linewidth=1.2)
ax[0, 0].vlines(x=chi2_min + 3.8, ymin=min(C_chi), ymax=max(C_chi), colors='saddlebrown', linestyles='-', linewidth=1.2)

# Plotting chi square + 1, + 2.3, + 3.8 level in the lateral graph ax[1, 1]
ax[1, 1].hlines(y=chi2_min + 1, xmin=min(B_chi), xmax=max(B_chi), colors='red', linestyles='-', linewidth=1.2)
ax[1, 1].hlines(y=chi2_min + 2.3, xmin=min(B_chi), xmax=max(B_chi), colors='darkorange', linestyles='-', linewidth=1.2)
ax[1, 1].hlines(y=chi2_min + 3.8, xmin=min(B_chi), xmax=max(B_chi), colors='saddlebrown', linestyles='-', linewidth=1.2)

ax[1, 1].set_xticks(np.linspace(ax[1, 1].get_xlim()[0], ax[1, 1].get_xlim()[1], num=5)) 
ax[1, 1].xaxis.set_major_formatter(FuncFormatter(format_xticks))

# setting axis labels
ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$Q \, - \, value$', fontsize = 22)
ax[0,0].set_xlabel(r'$\chi^2_{{min}}$', fontsize = 22)
ax[1,1].set_xlabel(r'$\Omega [ \, Hz \, ]$', fontsize = 22, labelpad=15)
ax[1,1].set_ylabel(r'$\chi^2_{{min}}$', fontsize = 22)
ax[0,1].set_xlabel(r'$\Omega [ \, Hz \, ]$', fontsize = 22)
ax[0,1].set_ylabel(r'$Q \, - \, value$', fontsize = 22)
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10))
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

# adding some text
ax[0, 1].text(B_chi[argchi2_min[1]]+100, C_chi[argchi2_min[2]]-1.35,r'$\chi^2_{{best}}$ = {g:.1f}'.format(g=chi2_min), size=18)

# plotting legend
ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.5, 0.95), loc='upper center', fontsize=20)

# Change the size of ticks for 2D plots
ax[0, 0].tick_params(axis='x', labelsize=11)
ax[0, 0].tick_params(axis='y', labelsize=11)

ax[0, 1].tick_params(axis='x', labelsize=11)
ax[0, 1].tick_params(axis='y', labelsize=11)

ax[1, 1].tick_params(axis='x', labelsize=11)
ax[1, 1].tick_params(axis='y', labelsize=11)

plt.savefig('chi_square_map_amp_R'+'.png',
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

# region - Plotting amplification with fit

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 6),sharex=True)
ax.errorbar(freq,T,yerr=eT, fmt='o', label=r'Amplification data',ms=3,color='black', zorder = 2, lw = 1.5)
ax.plot(fit_freq, fitf(fit_freq, *popt_T), label=r'Amplification fit', color='blue', lw = 1.2, zorder = 1)
#ax.fill_between(fit_freq, y_fit_down_T, y_fit_upper_T, color='skyblue', alpha=0.4, label = r'$\pm \, 1\sigma$ deviation', zorder = 0, lw = 0)
ax.axvspan(f_T/1000 - delta_f_T/2000, f_T/1000 + delta_f_T/2000, alpha=0.4, color='skyblue', label = 'Bandwidth', zorder = 0)
ax.axvline(f_T/1000, color = 'darkorange', label = 'Resonance frequency', lw = 1.5, ls = '--', zorder = 0)

ax.text(189, 0.25, r'$\Omega$ = {d:.1f} - {e:.1f} + {f:.1f} kHz'.format(d=B_chi[argchi2_min[1]]/1000, e=errB/1000, f=errBB/1000), size=13)
ax.text(189, 0.23,r'$Q$ = {g:.1f} - {h:.1f} + {n:.1f}'.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC), size=13)
ax.text(189, 0.21,r'$B$ = {a:.3f} - {b:.3f} + {c:.3f}'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA), size=13)
ax.text(189, 0.19,r'$f_R$ = {a:.1f} $\pm$ {b:.1f} KHz'.format(a=f_T/1000,b = f_T_err/1000), size=13)
ax.text(189, 0.17,r'$\Delta f_R$ = {a:.1f} $\pm$ {b:.1f} KHz'.format(a=delta_f_T/1000,b = delta_f_T_err/1000), size=13)
ax.text(189, 0.15, r'$\chi^2$/DOF = {m:.1f} / {i:.0f}'.format(m=chi2_min, i = df), size=13)

ax.set_ylabel(r'$|A| \, = \, \frac{V_{\text{out}}}{V_{\text{in}}}$', size = 13)
ax.set_xlabel(r'Frequency $[\, kHz \, ]$', size = 13)
ax.set_xlim(185, 266)
ax.set_ylim(0.055, 0.38)
ax.legend(prop={'size': 14}, loc='upper left', frameon=False).set_zorder(2)

plt.savefig('amp_R'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

# region - Manual chi square minimization for phase

# Define the interval for parameter limits
NSI = 3 # number of sigma for chi square map
A0, A1 = A_phase - NSI * eA_phase, A_phase + NSI * eA_phase # A = A
B0, B1 = Omega_phase - NSI * eOmega_phase, Omega_phase + NSI * eOmega_phase # B = Omega
C0, C1 = Q_phase - NSI * eQ_phase, Q_phase + NSI * eQ_phase # C = Q value

# Computing chi square map 

# array for each parameter
A_chi = np.linspace(A0,A1,step)
B_chi = np.linspace(B0,B1,step)
C_chi = np.linspace(C0,C1,step)

# 3d matrix for chi square map
mappa = np.zeros((step,step,step))

item = [(i,j,k) for i in range(step) for j in range(step) for k in range(step)]

# assign the global pool
pool = multiprocessing.pool.ThreadPool(100)

# issue top level tasks to pool and wait
pool.starmap(fitchi2_phase, item, chunksize=10)

# close the pool
pool.close()

# return the chi square 3D map
mappa = np.asarray(mappa)            

# founding the minimum of chi square map and his position
chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa),mappa.shape)

chi2_min_phase = chi2_min

# computing residual and chi square value using the manual chi square minimization
residui_chi2 = phase - fitf_phase(freq,A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]])

chisq_res = np.sum((residui_chi2/eT)**2)

# endregion

# region - Chi square profilation for phase [Omega and Q-value]

# Computing 2D profilation on Omega and Q-value
chi2D = profi2D(1,mappa)

# Computing 1D profilation 
prof_B = profi1D([1,3],mappa)
prof_C = profi1D([1,2],mappa)   
prof_A = profi1D([2,3],mappa)    

# Computing the 1 sigma deviation error
lvl = chi2_min+1. #2.3 for 2 sigma, 3.8 for 3 sigma
diff_B = abs(prof_B-lvl)
diff_C = abs(prof_C-lvl)
diff_A = abs(prof_A-lvl)

B_dx = np.argmin(diff_B[B_chi<Omega_phase]) 
B_sx = np.argmin(diff_B[B_chi>Omega_phase])+len(diff_B[B_chi<Omega_phase]) 
C_dx = np.argmin(diff_C[C_chi<Q_phase])
C_sx = np.argmin(diff_C[C_chi>Q_phase])+len(diff_C[C_chi<Q_phase])
A_dx = np.argmin(diff_A[A_chi<A_phase])
A_sx = np.argmin(diff_A[A_chi>A_phase])+len(diff_A[A_chi<A_phase])


# Computing non symmetric error on Parameters
errA = A_chi[argchi2_min[0]]-A_chi[A_dx]
errAA = A_chi[A_sx]-A_chi[argchi2_min[0]]
errB = B_chi[argchi2_min[1]]-B_chi[B_dx] 
errBB = B_chi[B_sx] -B_chi[argchi2_min[1]]
errC = C_chi[argchi2_min[2]]-C_chi[C_dx]
errCC = C_chi[C_sx]-C_chi[argchi2_min[2]]


# Printing fit results 
print("============== BEST FIT PHASE - R ==================================")
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]] * 1e-3, e=errB * 1e-3, f=errBB* 1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) '.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC))
print(r'chisq = {m:.2f}'.format(m=np.min(mappa)))
print("====================================================================")

# Extracting the best parameters
Bphase, eBphasedx, eBphasesx = B_chi[argchi2_min[1]], errB, errBB
Aphase, eAphasedx, eAphasesx = A_chi[argchi2_min[0]], errA, errAA
Cphase, eCphasedx, eCphasesx = C_chi[argchi2_min[2]], errC, errCC

# Plotting the chi square map
cmap = mpl.colormaps['viridis'].reversed()
level = np.linspace(np.min(chi2D),np.max(chi2D),100)
line_c = 'gray'


# Plot definition
fig, ax = plt.subplots(2, 2, figsize=(8, 8),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
fig.suptitle(r'$\chi^2 \left(\omega_0, Q \right)$ - Phase - R', size = 20)

# defining smooth plot of chi square values in function of two parameteres
im = ax[0,1].contourf(B_chi,C_chi,chi2D, levels=level, cmap=cmap) 

# plotting the legend's bar of chi square value(and associated color)
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 

# Plotting constant chi-square elliptical orbit at Xmin + 1, + 2.3, + 3.8
CS = ax[0, 1].contour(B_chi, C_chi, chi2D, levels=[chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
                      linewidths=2, colors=['red','darkorange', 'saddlebrown'], alpha=1, linestyles='-')

# Add labels directly to the legend
handles, _ = CS.legend_elements()
labels = [r'$\chi^2_{{min}} + 1$', r'$\chi^2_{{min}} + 2.3$', r'$\chi^2_{{min}} + 3.8$']

# lotting chi square value of elliptical orbits
ax[0,1].clabel(CS, inline=True, fontsize=11, fmt='%0.1f')

# plotting deviation bars of parameters
ax[0,1].plot([B0,B1],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B0,B1],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_sx],B_chi[B_sx]],[C0,C1],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_dx],B_chi[B_dx]],[C0,C1],color=line_c, ls='dashed', lw = 1.5)

# plotting best parameters
ax[0,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[C0,C1], ls='dashed', lw = 1.5, color = 'black')
ax[0,1].plot([B0,B1],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black', label = 'Best fit')

#plotting the parabolic curve for Q(also standard dev and best parameter)
ax[0,0].plot(prof_B,C_chi,ls='-')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_sx],C_chi[B_sx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_dx],C_chi[B_dx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black')
ax[0,0].set_xticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])
#ax[0,0].text(chi2_min - 10, C_chi[argchi2_min[2]],r'{e:.1f}'.format(e=C_chi[argchi2_min[2]]), color='k',alpha=1, fontsize=10)
#ax[0,0].text(chi2_min - 10, C_chi[B_dx],r'{e:.1f}'.format(e=C_chi[B_dx] - C_chi[argchi2_min[2]]), color='b',alpha=1, fontsize=10)
#ax[0,0].text(chi2_min - 10, C_chi[B_sx],r'{e:.1f}'.format(e=C_chi[B_sx] - C_chi[argchi2_min[2]]), color='r',alpha=1, fontsize=10)

# plotting the parabolic curve for omega(also standard dev and best parameter)
ax[1,1].plot(B_chi,prof_C)
ax[1,1].plot([B_chi[C_sx],B_chi[C_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[C_dx],B_chi[C_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[int(chi2_min-2),int(chi2_min+10)], ls='dashed', lw = 1.5, color = 'black')
#ax[1,1].text(B_chi[argchi2_min[1]]-120, chi2_min - 6.4,r'{e:.4e} $Hz$'.format(e=B_chi[argchi2_min[1]]), color='k',alpha=1, fontsize=10)
#ax[1,1].text(B_chi[C_dx]-280, chi2_min +4.3,r'{e:.0e} $Hz$'.format(e=-B_chi[argchi2_min[1]] + B_chi[C_dx]), color='b',alpha=1, fontsize=10)
#ax[1,1].text(B_chi[C_sx]+20, chi2_min +4.3,r'{e:.0e} $Hz$'.format(e= -B_chi[argchi2_min[1]] + B_chi[C_sx]), color='r',alpha=1, fontsize=10)
ax[1,1].set_yticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])

# Plotting chi square + 1, + 2.3, + 3.8 level in the lateral graph ax[0, 0]
ax[0, 0].vlines(x=chi2_min + 1, ymin=min(C_chi), ymax=max(C_chi), colors='red', linestyles='-', linewidth=1.2)
ax[0, 0].vlines(x=chi2_min + 2.3, ymin=min(C_chi), ymax=max(C_chi), colors='darkorange', linestyles='-', linewidth=1.2)
ax[0, 0].vlines(x=chi2_min + 3.8, ymin=min(C_chi), ymax=max(C_chi), colors='saddlebrown', linestyles='-', linewidth=1.2)

# Plotting chi square + 1, + 2.3, + 3.8 level in the lateral graph ax[1, 1]
ax[1, 1].hlines(y=chi2_min + 1, xmin=min(B_chi), xmax=max(B_chi), colors='red', linestyles='-', linewidth=1.2)
ax[1, 1].hlines(y=chi2_min + 2.3, xmin=min(B_chi), xmax=max(B_chi), colors='darkorange', linestyles='-', linewidth=1.2)
ax[1, 1].hlines(y=chi2_min + 3.8, xmin=min(B_chi), xmax=max(B_chi), colors='saddlebrown', linestyles='-', linewidth=1.2)

ax[1, 1].set_xticks(np.linspace(ax[1, 1].get_xlim()[0], ax[1, 1].get_xlim()[1], num=5)) 
ax[1, 1].xaxis.set_major_formatter(FuncFormatter(format_xticks))

# setting axis labels
ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$Q \, - \, value$', fontsize = 22)
ax[0,0].set_xlabel(r'$\chi^2_{{min}}$', fontsize = 22)
ax[1,1].set_xlabel(r'$\Omega [ \, Hz \, ]$', fontsize = 22, labelpad=15)
ax[1,1].set_ylabel(r'$\chi^2_{{min}}$', fontsize = 22)
ax[0,1].set_xlabel(r'$\Omega [ \, Hz \, ]$', fontsize = 22)
ax[0,1].set_ylabel(r'$Q \, - \, value$', fontsize = 22)
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10))
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

# adding some text
ax[0, 1].text(B_chi[argchi2_min[1]]+30, C_chi[argchi2_min[2]]-0.35,r'$\chi^2_{{best}}$ = {g:.1f}'.format(g=chi2_min), size=18)

# plotting legend
ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.5, 0.95), loc='upper center', fontsize=20)

# Change the size of ticks for 2D plots
ax[0, 0].tick_params(axis='x', labelsize=11)
ax[0, 0].tick_params(axis='y', labelsize=11)

ax[0, 1].tick_params(axis='x', labelsize=11)
ax[0, 1].tick_params(axis='y', labelsize=11)

ax[1, 1].tick_params(axis='x', labelsize=11)
ax[1, 1].tick_params(axis='y', labelsize=11)

plt.savefig('chi_square_map_phase_R'+'.png',
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

# region - Plotting phase with fit

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 6),sharex=True)
ax.errorbar(freq,phase,yerr=ephase, fmt='o', label=r'Phase data',ms=3,color='black', zorder = 2, lw = 1.5)
ax.plot(fit_freq, fitf_phase(fit_freq, *popt_phase), label=r'Phase fit', color='tab:red', lw = 1.2, zorder = 1)
#ax.fill_between(fit_freq, y_fit_down, y_fit_upper, color='skyblue', alpha=0.4, label = r'$\pm \, 1\sigma$ deviation', zorder = 0, lw = 0)
ax.axvspan(f_phase/1000 - delta_f_phase/2000, f_T/1000 + delta_f_phase/2000, alpha=0.4, color='gold', label = 'Bandwidth', zorder = 0)
ax.axvline(f_phase/1000, color = 'darkgreen', label = 'Resonance frequency', lw = 1.5, ls = '--', zorder = 0)

ax.text(193, 0.3, r'$\Omega$ = {d:.1f} - {e:.1f} + {f:.1f} kHz'.format(d=B_chi[argchi2_min[1]]/1000, e=errB/1000, f=errBB/1000), size=13)
ax.text(193, 0.1,r'$Q$ = {g:.1f} - {h:.1f} + {n:.1f}'.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC), size=13)
ax.text(193, -0.1,r'$D$ = {a:.3f} - {b:.3f} + {c:.3f}'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA), size=13)
ax.text(193, -0.3,r'$f_R$ = {a:.2f} $\pm$ {b:.2f} KHz'.format(a=f_phase/1000,b = f_phase_err/1000), size=13)
ax.text(193, -0.5,r'$\Delta f_R$ = {a:.1f} $\pm$ {b:.1f} KHz'.format(a=delta_f_phase/1000,b = delta_f_phase_err/1000), size=13)
ax.text(193, -0.7, r'$\chi^2$/DOF = {m:.1f} / {i:.0f}'.format(m=chi2_min_phase, i = df), size=13)

ax.set_ylabel(r'$\Delta \Phi \, = \, 2 \, \pi \, f \, \Delta T$', size = 13)
ax.set_xlabel(r'Frequency $[\, kHz \, ]$', size = 13)
ax.set_xlim(189, 265)
#ax.set_ylim(0.055, 0.38)
ax.legend(prop={'size': 14}, loc='upper left', frameon=False).set_zorder(2)

plt.savefig('image_4'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

#-------------------------------------------------------------
# ANALYSIS ON L AND R
#-------------------------------------------------------------

# region - Computing L and R from experimental data considering C METRIX

L_exp_T = 1/(C*Omega_T**2)
L_exp_phase = 1/(C*Omega_phase**2)
eL_exp_T = np.sqrt((eC/((Omega_T**2)**C**2))**2 + (2*eOmega_T/(C*Omega_T**3))**2)
eL_exp_phase = np.sqrt((eC/((Omega_phase**2)**C**2))**2 + (2*eOmega_phase/(C*Omega_phase**3))**2)
R_exp_T = 1/(C*Q_T*Omega_T)
R_exp_phase = 1/(C*Q_phase*Omega_phase)
eR_exp_T = np.sqrt((eC/(Omega_T*Q_T*C**2))**2 + (eQ_T/(C*Omega_T*Q_T**2))**2 + (eOmega_T/(C*Q_T*Omega_T**2))**2)
eR_exp_phase = np.sqrt((eC/(Omega_phase*Q_phase*C**2))**2 + (eQ_phase/(C*Omega_phase*Q_phase**2))**2 + (eOmega_phase/(C*Q_phase*Omega_phase**2))**2)

# Printing fit results
print("============== BEST FIT L and R - Amplification ====================")
print(r'L = {a:.3f} +-{b:.1f} muH'.format(a=L_exp_T*1000000,b=eL_exp_T*1000000))
print(r'R = {d:.3f} +- {e:.1f} Ohm'.format(d=R_exp_T,e=eR_exp_T))
print("====================================================================")
print(" ")
print("============== BEST FIT L and R - Phase ============================")
print(r'L = {a:.3f} +- {b:.1f} muH'.format(a=L_exp_phase*1000000,b=eL_exp_phase*1000000))
print(r'R = {d:.3f} +- {e:.1f} Ohm'.format(d=R_exp_phase,e=eR_exp_phase))
print("====================================================================")
print(" ")
print("Compatibility L-amp with L-theory: ", compatibility(L_exp_T, L, eL_exp_T, eL))
print("Compatibility L-phase with L-theory: ", compatibility(L_exp_phase, L, eL_exp_phase, eL))
print("Compatibility L-amp with L-phase: ", compatibility(L_exp_T, L_exp_phase, eL_exp_T, eL_exp_phase))
print(" ")
print("Compatibility R-amp with R-theory: ", compatibility(R_exp_T, R, eR_exp_T, eR))
print("Compatibility R-phase with R-theory: ", compatibility(R_exp_phase, R, eR_exp_phase, eR))
print("Compatibility R-amp with R-phase: ", compatibility(R_exp_T, R_exp_phase, eR_exp_T, eR_exp_phase))

# endregion

# region - Studying total resistance
Vin_off = 3.96
eVin_off = np.sqrt((2/(np.sqrt(24)*25))**2 + ((errscalaV * Vin_off*2/np.sqrt(24))**2))
Vin_on = 1.66 
eVin_on = np.sqrt((0.5*2/(np.sqrt(24)*25))**2 + ((errscalaV * Vin_on*2/np.sqrt(24))**2))

print(" ")
print("V_off = ", Vin_off, "+-", eVin_off)
print("V_on = ", Vin_on, "+-", eVin_on)
print(" ")
eRg = 0.5
Rtot_exp = Rg/(1-Vin_on/Vin_off)
eRtot_exp = np.sqrt((Vin_off*eRg/(Vin_off - Vin_on))**2 + (Rg*Vin_off*eVin_on/(Vin_off - Vin_on)**2)**2 + (Rg*Vin_on*eVin_off/(Vin_off - Vin_on)**2)**2)

Rtot_exp_nog = Rtot_exp - Rg
eRtot_exp_nog = np.sqrt((eRtot_exp)**2 + (eRg)**2)

print(" ")
print("============== R TOT ===============================================")
print(r'Rtot_exp = {a:.3f} +- {b:.1f} Ohm'.format(a=Rtot_exp,b=eRtot_exp))
print(r'Rtot_theory = {a:.3f} +- {b:.1f} Ohm'.format(a=Rtot,b=computing_R_err(Rtot)))
print(r'Rtot_exp_nog = {a:.3f} +- {b:.2f} Ohm'.format(a=Rtot_exp_nog,b=eRtot_exp_nog))
print("Compatibility Rtot with R amplification: ", compatibility(Rtot_exp_nog, R_exp_T, eRtot_exp_nog, eR_exp_T))
print("Compatibility Rtot with R phase: ", compatibility(Rtot_exp_nog, R_exp_phase, eRtot_exp_nog, eR_exp_phase))
print("Compatibility Rtot with R theory: ", compatibility(Rtot_exp, Rtot, eRtot_exp, computing_R_err(Rtot)))
print("====================================================================")

# endregion 

# region - Studying Voltage divider effects

# Computing the voltage divider effect on the amplification

H_fromRtot_exp = Rx/(Rtot_exp_nog)
eH_fromRtot_exp = np.sqrt((eRtot_exp_nog*Rx/(Rtot_exp_nog)**2)**2 + (computing_R_err(Rx)/(Rtot_exp_nog))**2)

AA = Rtot-Rg
eAA = np.sqrt(eRtot**2 + eRg**2)
H_theory = Rx/AA
eH_theory = np.sqrt((eAA*Rx/(AA)**2)**2 + (computing_R_err(Rx)/(AA))**2)

H_exp = A_T
eH_exp = eA_T

print(" ")
print("============== VOLTAGE DIVIDER EFFECT ==============================")
print(r'H_fromRtot_exp = {a:.3f} +- {b:.3f} Ohm'.format(a=H_fromRtot_exp,b=eH_fromRtot_exp))
print(r'H_exp = {a:.3f} +- {b:.3f} Ohm'.format(a=H_exp,b=eH_exp))
print(r'H_theory = {a:.3f} +- {b:.3f} Ohm'.format(a=H_theory,b=eH_theory))
print("Compatibility H from experimental Rtot with H amplification: ", compatibility(H_fromRtot_exp, H_exp, eH_fromRtot_exp, eH_exp))
print("Compatibility H theory with H amplification: ", compatibility(H_theory, H_exp, eH_theory, eH_exp))
print("====================================================================")
print(" ")
# endregion

#-------------------------------------------------------------
# CAPACITANCE
#-------------------------------------------------------------

# region - Data acquisition for capacitance

# Input file name
inputname1 = 'dataC.txt'

# load sperimental data from file
data = np.loadtxt(inputname1).T
freq = np.array(data[0])
Vin = np.array(data[1])
Vin_scale = np.array(data[2])
Vout = np.array(data[3])
Vout_scale = np.array(data[4])
time = np.array(data[5])

# Defining time scale
Tscale = 0.5 #microseconds

# Computing errors on each data type
# Assumed reading errors --> triangolar distribution applied as maximum error on 1/25 of division
letturaV_out = Vout_scale*2/(np.sqrt(24)*25)
letturaV_in = Vin_scale*2/(np.sqrt(24)*25)
letturaT = Tscale*2/(np.sqrt(24)*25)
errscalaV = 1.5/100

eVin = np.sqrt((letturaV_in)**2 + ((errscalaV * Vin)**2))
eVout = np.sqrt((letturaV_out)**2 + ((errscalaV * Vout)**2))

# Computing amplification and error of amplification
T = Vout/Vin 
eT = T*np.sqrt((eVout/Vout)**2 + (eVin/Vin)**2 + 2*(errscalaV**2))

# Computing phase and error of phase
phase = 2*np.pi*freq*time/1000 
ephase = 2*np.pi*freq*letturaT/1000

# Number of points
N = len(freq[freq > 0])

# endregion

# region - Amplification's non-linear fit

# Computing the initial parameter values from experimental data 
Q_init = np.sqrt(L/C)/R

Omega_init = 1/np.sqrt(L*C)
A_init = 1

# Computing the fitting using LM algorithm
popt_T, pcov_T = curve_fit(fitf_C, freq, T, p0=[A_init, Omega_init, Q_init], method='lm', sigma=eT, absolute_sigma=True)

# Computing the residual 
residual_T = T - fitf_C(freq, *popt_T)

# variables error and chi2
perr_T = np.sqrt(np.diag(pcov_T))
chi2_T = np.sum((residual_T/eT)**2)

# degrees of freedom
df = N - 3

# founding parameters and errors
# A --> offset, Omega --> 1/sqrt(LC), Q --> R*sqrt(C/L) - Q-value
# fitting parameters and errors
A_T, Omega_T, Q_T = popt_T #fit parameters
eA_T, eOmega_T, eQ_T = perr_T #fit errors

Q_T_C = Q_T 
eQ_T_C = eQ_T

# Computing resonance frequency and band interval
f_T_C = Omega_T/(2*np.pi)
f_T_C_err = eOmega_T/(2*np.pi)
delta_f_T_C = Omega_T/(2*np.pi*Q_T)
delta_f_T_C_err = np.sqrt((eOmega_T/Q_T)**2 + (Omega_T*eQ_T/Q_T**2)**2 - 2*(1/Q_T)*(Omega_T/Q_T**2)*pcov_T[1, 2])/(2*np.pi)

# calculate the residuals error by quadratic sum using the variance theorem
T_residual_err = eT

# Computing the weighted mean of the residuals
weighted_mean_T_residual = np.average(residual_T, weights=1/T_residual_err**2)
weighted_mean_T_residual_std = np.sqrt(1 / np.sum(1/T_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_T = compatibility(weighted_mean_T_residual,0, weighted_mean_T_residual_std, 0)

# Computing the array for fit points
fit_freq = np.linspace(min(freq), max(freq), 1000)

# Array of element for plotting the wheigted mean of residuals
T_residual_array = np.full(len(fit_freq), weighted_mean_T_residual)

# Computing the region of 1 sigma deviation for linear fit
y_fit_upper_T = fitf_C(fit_freq, *popt_T + perr_T)
y_fit_down_T = fitf_C(fit_freq, *popt_T - perr_T)

# endregion

# region - Phase's non-linear fit

# Computing the fitting using LM algorithm
popt_phase, pcov_phase = curve_fit(fitf_phase_C, freq, phase, p0=[A_init, Omega_init, Q_init], method='lm', sigma=ephase, absolute_sigma=True)

# Computing the residual 
residual_phase = phase - fitf_phase_C(freq, *popt_phase)

# variables error and chi2
perr_phase = np.sqrt(np.diag(pcov_phase))
chi2_phase = np.sum((residual_phase/ephase)**2)

# degrees of freedom
df = N - 3

# founding parameters and errors
# A --> offset, Omega --> 1/sqrt(LC), Q --> R*sqrt(C/L) - Q-value
# fitting parameters and errors
A_phase, Omega_phase, Q_phase = popt_phase #fit parameters
eA_phase, eOmega_phase, eQ_phase = perr_phase #fit errors

Q_phase_C = Q_phase
eQ_phase_C = eQ_phase

# Computing resonance frequency and band interval
f_phase_C = Omega_phase/(2*np.pi)
f_phase_C_err = eOmega_phase/(2*np.pi)
delta_f_phase_C = Omega_phase/(2*np.pi*Q_phase)
delta_f_phase_C_err = np.sqrt((eOmega_phase/Q_phase)**2 + (Omega_phase*eQ_phase/Q_phase**2)**2 - 2*(1/Q_phase)*(Omega_phase/Q_phase**2)*pcov_phase[1, 2])/(2*np.pi)

# calculate the residuals error by quadratic sum using the variance theorem
phase_residual_err = ephase

# Computing the weighted mean of the residuals
weighted_mean_phase_residual = np.average(residual_phase, weights=1/phase_residual_err**2)
weighted_mean_phase_residual_std = np.sqrt(1 / np.sum(1/phase_residual_err**2))

# computing compatibility between weighted mean of residuals and 0
r_residual_phase = compatibility(weighted_mean_phase_residual,0, weighted_mean_phase_residual_std, 0)

# Array of element for plotting the wheigted mean of residuals
phase_residual_array = np.full(len(fit_freq), weighted_mean_phase_residual)

# Computing the region of 1 sigma deviation for linear fit
y_fit_upper_phase = fitf_phase_C(fit_freq, *popt_phase + perr_phase)
y_fit_down_phase = fitf_phase_C(fit_freq, *popt_phase - perr_phase)

# endregion

# region - Manual chi square minimization for amplification

# Define the interval for parameter limits
NSI = 3 # number of sigma for chi square map
A0, A1 = A_T - NSI * eA_T, A_T + NSI * eA_T # A = A
B0, B1 = Omega_T - NSI * eOmega_T, Omega_T + NSI * eOmega_T # B = Omega
C0, C1 = Q_T - NSI * eQ_T, Q_T + NSI * eQ_T # C = Q value

# Computing chi square map 

# array for each parameter
A_chi = np.linspace(A0,A1,step)
B_chi = np.linspace(B0,B1,step)
C_chi = np.linspace(C0,C1,step)

# 3d matrix for chi square map
mappa = np.zeros((step,step,step))

item = [(i,j,k) for i in range(step) for j in range(step) for k in range(step)]

# assign the global pool
pool = multiprocessing.pool.ThreadPool(100)

# issue top level tasks to pool and wait
pool.starmap(fitchi2_C, item, chunksize=10)

# close the pool
pool.close()

# return the chi square 3D map
mappa = np.asarray(mappa)            

# founding the minimum of chi square map and his position
chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa),mappa.shape)

chi2_min_T = chi2_min

# computing residual and chi square value using the manual chi square minimization
residui_chi2 = T - fitf_C(freq,A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]])

chisq_res = np.sum((residui_chi2/eT)**2)

# endregion

# region - Chi square profilation for amplification [Omega and Q-value]

# Computing 2D profilation on Omega and Q-value
chi2D = profi2D(1,mappa)

# Computing 1D profilation 
prof_B = profi1D([1,3],mappa)
prof_C = profi1D([1,2],mappa)   
prof_A = profi1D([2,3],mappa)    

# Computing the 1 sigma deviation error
lvl = chi2_min+1. #2.3 for 2 sigma, 3.8 for 3 sigma
diff_B = abs(prof_B-lvl)
diff_C = abs(prof_C-lvl)
diff_A = abs(prof_A-lvl)

B_dx = np.argmin(diff_B[B_chi<Omega_T]) 
B_sx = np.argmin(diff_B[B_chi>Omega_T])+len(diff_B[B_chi<Omega_T]) 
C_dx = np.argmin(diff_C[C_chi<Q_T])
C_sx = np.argmin(diff_C[C_chi>Q_T])+len(diff_C[C_chi<Q_T])
A_dx = np.argmin(diff_A[A_chi<A_T])
A_sx = np.argmin(diff_A[A_chi>A_T])+len(diff_A[A_chi<A_T])


# Computing non symmetric error on Parameters
errA = A_chi[argchi2_min[0]]-A_chi[A_dx]
errAA = A_chi[A_sx]-A_chi[argchi2_min[0]]
errB = B_chi[argchi2_min[1]]-B_chi[B_dx] 
errBB = B_chi[B_sx] -B_chi[argchi2_min[1]]
errC = C_chi[argchi2_min[2]]-C_chi[C_dx]
errCC = C_chi[C_sx]-C_chi[argchi2_min[2]]


# Printing fit results 
print("============== BEST FIT AMPLIFICATION - C ==========================")
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]] * 1e-3, e=errB * 1e-3, f=errBB* 1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) '.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC))
print(r'chisq = {m:.2f}'.format(m=np.min(mappa)))
print("====================================================================")

# Extracting the best parameters
BT, eBTdx, eBTsx = B_chi[argchi2_min[1]], errB, errBB
AT, eATdx, eATsx = A_chi[argchi2_min[0]], errA, errAA
CT, eCTdx, eCTsx = C_chi[argchi2_min[2]], errC, errCC

# Plotting the chi square map
cmap = mpl.colormaps['viridis'].reversed()
level = np.linspace(np.min(chi2D),np.max(chi2D),100)
line_c = 'gray'


# Plot definition
fig, ax = plt.subplots(2, 2, figsize=(8, 8),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
fig.suptitle(r'$\chi^2 \left(\omega_0, Q \right)$ - Amplification - C')

# defining smooth plot of chi square values in function of two parameteres
im = ax[0,1].contourf(B_chi,C_chi,chi2D, levels=level, cmap=cmap) 

# plotting the legend's bar of chi square value(and associated color)
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 

# setting the bar label
cbar.set_label(r'$\chi^2$',rotation=360, size = 17)

# Plotting constant chi-square elliptical orbit at Xmin + 1, + 2.3, + 3.8
CS = ax[0, 1].contour(B_chi, C_chi, chi2D, levels=[chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
                      linewidths=2, colors=['red','darkorange', 'saddlebrown'], alpha=1, linestyles='-')

# Add labels directly to the legend
handles, _ = CS.legend_elements()
labels = [r'$\chi^2_{{min}} + 1$', r'$\chi^2_{{min}} + 2.3$', r'$\chi^2_{{min}} + 3.8$']

# lotting chi square value of elliptical orbits
ax[0,1].clabel(CS, inline=True, fontsize=11, fmt='%0.1f')

# plotting deviation bars of parameters
ax[0,1].plot([B0,B1],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B0,B1],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_sx],B_chi[B_sx]],[C0,C1],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_dx],B_chi[B_dx]],[C0,C1],color=line_c, ls='dashed', lw = 1.5)

# plotting best parameters
ax[0,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[C0,C1], ls='dashed', lw = 1.5, color = 'black')
ax[0,1].plot([B0,B1],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black', label = 'Best fit')

#plotting the parabolic curve for Q(also standard dev and best parameter)
ax[0,0].plot(prof_B,C_chi,ls='-')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_sx],C_chi[B_sx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_dx],C_chi[B_dx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black')
ax[0,0].set_xticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])
ax[0,0].text(chi2_min - 7, C_chi[argchi2_min[2]],r'{e:.1f}'.format(e=C_chi[argchi2_min[2]]), color='k',alpha=1, fontsize=10)
ax[0,0].text(chi2_min - 7, C_chi[B_dx],r'{e:.1f}'.format(e=C_chi[B_dx] - C_chi[argchi2_min[2]]), color='b',alpha=1, fontsize=10)
ax[0,0].text(chi2_min - 7, C_chi[B_sx],r'{e:.1f}'.format(e=C_chi[B_sx] - C_chi[argchi2_min[2]]), color='r',alpha=1, fontsize=10)

# plotting the parabolic curve for omega(also standard dev and best parameter)
ax[1,1].plot(B_chi,prof_C)
ax[1,1].plot([B_chi[C_sx],B_chi[C_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[C_dx],B_chi[C_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[int(chi2_min-2),int(chi2_min+10)], ls='dashed', lw = 1.5, color = 'black')
ax[1,1].text(B_chi[argchi2_min[1]]-400, chi2_min - 6.4,r'{e:.4e} $Hz$'.format(e=B_chi[argchi2_min[1]]), color='k',alpha=1, fontsize=10)
ax[1,1].text(B_chi[C_dx]-710, chi2_min +4.3,r'{e:.0e} $Hz$'.format(e=-B_chi[argchi2_min[1]] + B_chi[C_dx]), color='b',alpha=1, fontsize=10)
ax[1,1].text(B_chi[C_sx]+80, chi2_min +4.3,r'{e:.0e} $Hz$'.format(e= -B_chi[argchi2_min[1]] + B_chi[C_sx]), color='r',alpha=1, fontsize=10)
ax[1,1].set_yticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])

# Plotting chi square + 1, + 2.3, + 3.8 level in the lateral graph ax[0, 0]
ax[0, 0].vlines(x=chi2_min + 1, ymin=min(C_chi), ymax=max(C_chi), colors='red', linestyles='-', linewidth=1.2)
ax[0, 0].vlines(x=chi2_min + 2.3, ymin=min(C_chi), ymax=max(C_chi), colors='darkorange', linestyles='-', linewidth=1.2)
ax[0, 0].vlines(x=chi2_min + 3.8, ymin=min(C_chi), ymax=max(C_chi), colors='saddlebrown', linestyles='-', linewidth=1.2)

# Plotting chi square + 1, + 2.3, + 3.8 level in the lateral graph ax[1, 1]
ax[1, 1].hlines(y=chi2_min + 1, xmin=min(B_chi), xmax=max(B_chi), colors='red', linestyles='-', linewidth=1.2)
ax[1, 1].hlines(y=chi2_min + 2.3, xmin=min(B_chi), xmax=max(B_chi), colors='darkorange', linestyles='-', linewidth=1.2)
ax[1, 1].hlines(y=chi2_min + 3.8, xmin=min(B_chi), xmax=max(B_chi), colors='saddlebrown', linestyles='-', linewidth=1.2)

# setting axis labels
ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$Q \, - \, value$', fontsize = 15)
ax[0,0].set_xlabel(r'$\chi^2_{{min}}$', fontsize = 16)
ax[1,1].set_xlabel(r'$\Omega [ \, Hz \, ]$', fontsize = 16, labelpad=15)
ax[1,1].set_ylabel(r'$\chi^2_{{min}}$', fontsize = 16)
ax[0,1].set_xlabel(r'$\Omega [ \, Hz \, ]$', fontsize = 16)
ax[0,1].set_ylabel(r'$Q \, - \, value$', fontsize = 15)
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10))
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

# adding some text
ax[0, 1].text(B_chi[argchi2_min[1]]+100, C_chi[argchi2_min[2]]-1.1,r'$\chi^2_{{best}}$ = {g:.1f}'.format(g=chi2_min), size=14)

# plotting legend
ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.5, 0.95), loc='upper center', fontsize=16)

# Change the size of ticks for 2D plots
ax[0, 0].tick_params(axis='x', labelsize=11)
ax[0, 0].tick_params(axis='y', labelsize=11)

ax[0, 1].tick_params(axis='x', labelsize=11)
ax[0, 1].tick_params(axis='y', labelsize=11)

ax[1, 1].tick_params(axis='x', labelsize=11)
ax[1, 1].tick_params(axis='y', labelsize=11)

plt.savefig('chi_square_map_amp_C'+'.png',
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

# region - Plotting amplification with fit and residuals

# Computing the region of 1 sigma deviation for weighted mean fit
y_fit_upper = np.full(len(fit_freq), weighted_mean_T_residual + weighted_mean_T_residual_std)
y_fit_down = np.full(len(fit_freq), weighted_mean_T_residual - weighted_mean_T_residual_std)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 1])

ax[0].errorbar(freq,T,yerr=eT, fmt='o', label=r'Amplification data',ms=3,color='black', zorder = 2, lw = 1.5)
ax[0].plot(fit_freq, fitf_C(fit_freq, *popt_T), label=r'Amplification fit', color='blue', lw = 1.2, zorder = 1)
ax[0].fill_between(fit_freq, y_fit_down_T, y_fit_upper_T, color='skyblue', alpha=0.4, label = r'$\pm \, 1\sigma$ deviation', zorder = 0, lw = 0)
ax[0].text(210, 4, r'$\Omega$ = {d:.1f} - {e:.1f} + {f:.1f} kHz'.format(d=B_chi[argchi2_min[1]]/1000, e=errB/1000, f=errBB/1000), size=13)
ax[0].text(210, 2,r'$Q$ = {g:.1f} - {h:.1f} + {n:.1f}'.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC), size=13)
ax[0].text(210, 0,r'$A$ = {a:.3f} - {b:.3f} + {c:.3f}'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA), size=13)
ax[0].text(235, 4,r'$f_R$ = {a:.1f} $\pm$ {b:.1f} KHz'.format(a=f_T_C/1000,b = f_T_C_err/1000), size=13)
ax[0].text(235, 2,r'$\Delta f_R$ = {a:.1f} $\pm$ {b:.1f} KHz'.format(a=delta_f_T_C/1000,b = delta_f_T_C_err/1000), size=13)
ax[0].text(235, 0, r'$\chi^2$/DOF = {m:.1f} / {i:.0f}'.format(m=chi2_min, i = df), size=13)
ax[0].set_xlim(206, 260)
ax[0].set_ylim(-2, 21)

ax[0].set_ylabel(r'$|A| \, = \, \frac{V_{\text{out}}}{V_{\text{in}}}$', size = 13)
ax[0].set_xlabel(r'Frequency $[\, kHz \, ]$', size = 13)
ax[0].legend(prop={'size': 14}, loc='upper left')

ax[1].errorbar(freq,residual_T,yerr=T_residual_err, fmt='o', label=r'Amplification residuals',ms=3,color='black', zorder = 2, lw = 1.5)
ax[1].plot(fit_freq, T_residual_array, label=r'Residual weighted mean', color='blue', lw = 1.5, zorder = 1)
ax[1].fill_between(fit_freq, y_fit_down, y_fit_upper, color='silver', alpha=0.5, label = r'Weighted mean $1\sigma$ deviation', zorder = 0, lw = 0)

# Plotting some text about residuals weighted mean and compatibility with zero
ax[1].text(210, -1.2, r'$\mu_{{\,\text{{residual}}}}$ = {e:.2f} $\pm$ {f:.2f} $mV$'.format(e=weighted_mean_T_residual, f = weighted_mean_T_residual_std), size=13)
ax[1].text(235, -1.2, r'$r_{{\, \mu \, / \, 0}}$ = {e:.2f}'.format(e= r_residual_T), size=13)
ax[1].set_ylim(-1.6, 1.2)

ax[1].set_ylabel(r'$Residual$', size = 13)
ax[1].set_xlabel(r'Frequency $[\, kHz \, ]$', size = 13)

plt.savefig('amp_C'+'.png',
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - Manual chi square minimization for phase

# Define the interval for parameter limits
NSI = 3 # number of sigma for chi square map
A0, A1 = A_phase - NSI * eA_phase, A_phase + NSI * eA_phase # A = A
B0, B1 = Omega_phase - NSI * eOmega_phase, Omega_phase + NSI * eOmega_phase # B = Omega
C0, C1 = Q_phase - NSI * eQ_phase, Q_phase + NSI * eQ_phase # C = Q value

# Computing chi square map 

# array for each parameter
A_chi = np.linspace(A0,A1,step)
B_chi = np.linspace(B0,B1,step)
C_chi = np.linspace(C0,C1,step)

# 3d matrix for chi square map
mappa = np.zeros((step,step,step))

item = [(i,j,k) for i in range(step) for j in range(step) for k in range(step)]

# assign the global pool
pool = multiprocessing.pool.ThreadPool(100)

# issue top level tasks to pool and wait
pool.starmap(fitchi2_phase_C, item, chunksize=10)

# close the pool
pool.close()

# return the chi square 3D map
mappa = np.asarray(mappa)            

# founding the minimum of chi square map and his position
chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa),mappa.shape)

chi2_min_phase = chi2_min

# computing residual and chi square value using the manual chi square minimization
residui_chi2 = phase - fitf_phase_C(freq,A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]])

chisq_res = np.sum((residui_chi2/eT)**2)

# endregion

# region - Chi square profilation for phase [Omega and Q-value]

# Computing 2D profilation on Omega and Q-value
chi2D = profi2D(1,mappa)

# Computing 1D profilation 
prof_B = profi1D([1,3],mappa)
prof_C = profi1D([1,2],mappa)   
prof_A = profi1D([2,3],mappa)    

# Computing the 1 sigma deviation error
lvl = chi2_min+1. #2.3 for 2 sigma, 3.8 for 3 sigma
diff_B = abs(prof_B-lvl)
diff_C = abs(prof_C-lvl)
diff_A = abs(prof_A-lvl)

B_dx = np.argmin(diff_B[B_chi<Omega_phase]) 
B_sx = np.argmin(diff_B[B_chi>Omega_phase])+len(diff_B[B_chi<Omega_phase]) 
C_dx = np.argmin(diff_C[C_chi<Q_phase])
C_sx = np.argmin(diff_C[C_chi>Q_phase])+len(diff_C[C_chi<Q_phase])
A_dx = np.argmin(diff_A[A_chi<A_phase])
A_sx = np.argmin(diff_A[A_chi>A_phase])+len(diff_A[A_chi<A_phase])


# Computing non symmetric error on Parameters
errA = A_chi[argchi2_min[0]]-A_chi[A_dx]
errAA = A_chi[A_sx]-A_chi[argchi2_min[0]]
errB = B_chi[argchi2_min[1]]-B_chi[B_dx] 
errBB = B_chi[B_sx] -B_chi[argchi2_min[1]]
errC = C_chi[argchi2_min[2]]-C_chi[C_dx]
errCC = C_chi[C_sx]-C_chi[argchi2_min[2]]


# Printing fit results 
print("============== BEST FIT PHASE - C ==================================")
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]] * 1e-3, e=errB * 1e-3, f=errBB* 1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) '.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC))
print(r'chisq = {m:.2f}'.format(m=np.min(mappa)))
print("====================================================================")

# Extracting the best parameters
Bphase, eBphasedx, eBphasesx = B_chi[argchi2_min[1]], errB, errBB
Aphase, eAphasedx, eAphasesx = A_chi[argchi2_min[0]], errA, errAA
Cphase, eCphasedx, eCphasesx = C_chi[argchi2_min[2]], errC, errCC

# Plotting the chi square map
cmap = mpl.colormaps['viridis'].reversed()
level = np.linspace(np.min(chi2D),np.max(chi2D),100)
line_c = 'gray'


# Plot definition
fig, ax = plt.subplots(2, 2, figsize=(8, 8),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
fig.suptitle(r'$\chi^2 \left(\omega_0, Q \right)$ - Phase - C')

# defining smooth plot of chi square values in function of two parameteres
im = ax[0,1].contourf(B_chi,C_chi,chi2D, levels=level, cmap=cmap) 

# plotting the legend's bar of chi square value(and associated color)
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 

# setting the bar label
cbar.set_label(r'$\chi^2$',rotation=360, size = 17)

# Plotting constant chi-square elliptical orbit at Xmin + 1, + 2.3, + 3.8
CS = ax[0, 1].contour(B_chi, C_chi, chi2D, levels=[chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
                      linewidths=2, colors=['red','darkorange', 'saddlebrown'], alpha=1, linestyles='-')

# Add labels directly to the legend
handles, _ = CS.legend_elements()
labels = [r'$\chi^2_{{min}} + 1$', r'$\chi^2_{{min}} + 2.3$', r'$\chi^2_{{min}} + 3.8$']

# lotting chi square value of elliptical orbits
ax[0,1].clabel(CS, inline=True, fontsize=11, fmt='%0.1f')

# plotting deviation bars of parameters
ax[0,1].plot([B0,B1],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B0,B1],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_sx],B_chi[B_sx]],[C0,C1],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_dx],B_chi[B_dx]],[C0,C1],color=line_c, ls='dashed', lw = 1.5)

# plotting best parameters
ax[0,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[C0,C1], ls='dashed', lw = 1.5, color = 'black')
ax[0,1].plot([B0,B1],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black', label = 'Best fit')

#plotting the parabolic curve for Q(also standard dev and best parameter)
ax[0,0].plot(prof_B,C_chi,ls='-')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_sx],C_chi[B_sx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_dx],C_chi[B_dx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black')
ax[0,0].set_xticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])
ax[0,0].text(chi2_min - 10, C_chi[argchi2_min[2]],r'{e:.1f}'.format(e=C_chi[argchi2_min[2]]), color='k',alpha=1, fontsize=10)
ax[0,0].text(chi2_min - 10, C_chi[B_dx],r'{e:.1f}'.format(e=C_chi[B_dx] - C_chi[argchi2_min[2]]), color='b',alpha=1, fontsize=10)
ax[0,0].text(chi2_min - 10, C_chi[B_sx],r'{e:.1f}'.format(e=C_chi[B_sx] - C_chi[argchi2_min[2]]), color='r',alpha=1, fontsize=10)

# plotting the parabolic curve for omega(also standard dev and best parameter)
ax[1,1].plot(B_chi,prof_C)
ax[1,1].plot([B_chi[C_sx],B_chi[C_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[C_dx],B_chi[C_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[int(chi2_min-2),int(chi2_min+10)], ls='dashed', lw = 1.5, color = 'black')
ax[1,1].text(B_chi[argchi2_min[1]]-120, chi2_min - 6.4,r'{e:.4e} $Hz$'.format(e=B_chi[argchi2_min[1]]), color='k',alpha=1, fontsize=10)
ax[1,1].text(B_chi[C_dx]-350, chi2_min +4.3,r'{e:.0e} $Hz$'.format(e=-B_chi[argchi2_min[1]] + B_chi[C_dx]), color='b',alpha=1, fontsize=10)
ax[1,1].text(B_chi[C_sx]+20, chi2_min +4.3,r'{e:.0e} $Hz$'.format(e= -B_chi[argchi2_min[1]] + B_chi[C_sx]), color='r',alpha=1, fontsize=10)
ax[1,1].set_yticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])

# Plotting chi square + 1, + 2.3, + 3.8 level in the lateral graph ax[0, 0]
ax[0, 0].vlines(x=chi2_min + 1, ymin=min(C_chi), ymax=max(C_chi), colors='red', linestyles='-', linewidth=1.2)
ax[0, 0].vlines(x=chi2_min + 2.3, ymin=min(C_chi), ymax=max(C_chi), colors='darkorange', linestyles='-', linewidth=1.2)
ax[0, 0].vlines(x=chi2_min + 3.8, ymin=min(C_chi), ymax=max(C_chi), colors='saddlebrown', linestyles='-', linewidth=1.2)

# Plotting chi square + 1, + 2.3, + 3.8 level in the lateral graph ax[1, 1]
ax[1, 1].hlines(y=chi2_min + 1, xmin=min(B_chi), xmax=max(B_chi), colors='red', linestyles='-', linewidth=1.2)
ax[1, 1].hlines(y=chi2_min + 2.3, xmin=min(B_chi), xmax=max(B_chi), colors='darkorange', linestyles='-', linewidth=1.2)
ax[1, 1].hlines(y=chi2_min + 3.8, xmin=min(B_chi), xmax=max(B_chi), colors='saddlebrown', linestyles='-', linewidth=1.2)

# setting axis labels
ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$Q \, - \, value$', fontsize = 17)
ax[0,0].set_xlabel(r'$\chi^2_{{min}}$', fontsize = 18)
ax[1,1].set_xlabel(r'$\Omega [ \, Hz \, ]$', fontsize = 18, labelpad=15)
ax[1,1].set_ylabel(r'$\chi^2_{{min}}$', fontsize = 18)
ax[0,1].set_xlabel(r'$\Omega [ \, Hz \, ]$', fontsize = 18)
ax[0,1].set_ylabel(r'$Q \, - \, value$', fontsize = 17)
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10))
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

# adding some text
ax[0, 1].text(B_chi[argchi2_min[1]]+30, C_chi[argchi2_min[2]]-0.38,r'$\chi^2_{{best}}$ = {g:.1f}'.format(g=chi2_min), size=14)

# plotting legend
ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.5, 0.95), loc='upper center', fontsize=16)

# Change the size of ticks for 2D plots
ax[0, 0].tick_params(axis='x', labelsize=11)
ax[0, 0].tick_params(axis='y', labelsize=11)

ax[0, 1].tick_params(axis='x', labelsize=11)
ax[0, 1].tick_params(axis='y', labelsize=11)

ax[1, 1].tick_params(axis='x', labelsize=11)
ax[1, 1].tick_params(axis='y', labelsize=11)

plt.savefig('chi_square_map_phase_C'+'.png',
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

# region - Plotting phase with fit and residuals

# Computing the region of 1 sigma deviation for weighted mean fit
y_fit_upper = np.full(len(fit_freq), weighted_mean_phase_residual + weighted_mean_phase_residual_std)
y_fit_down = np.full(len(fit_freq), weighted_mean_phase_residual - weighted_mean_phase_residual_std)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, height_ratios=[2, 1])

ax[0].errorbar(freq,phase,yerr=ephase, fmt='o', label=r'Phase data',ms=3,color='black', zorder = 2, lw = 1.5)
ax[0].plot(fit_freq, fitf_phase(fit_freq, *popt_phase), label=r'Phase fit', color='tab:red', lw = 1.2, zorder = 1)
#ax[0].fill_between(fit_freq, y_fit_down_phase, y_fit_upper_phase, color='skyblue', alpha=0.4, label = r'$\pm \, 1\sigma$ deviation', zorder = 0, lw = 0)

ax[0].text(238, 1.5, r'$\Omega$ = {d:.1f} - {e:.1f} + {f:.1f} kHz'.format(d=B_chi[argchi2_min[1]]/1000, e=errB/1000, f=errBB/1000), size=13)
ax[0].text(238, 1.3,r'$Q$ = {g:.1f} - {h:.1f} + {n:.1f}'.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC), size=13)
ax[0].text(238, 1.1,r'$A$ = {a:.3f} - {b:.3f} + {c:.3f}'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA), size=13)
ax[0].text(238, 0.9,r'$f_R$ = {a:.2f} $\pm$ {b:.2f} KHz'.format(a=f_phase_C/1000,b = f_phase_C_err/1000), size=13)
ax[0].text(238, 0.7,r'$\Delta f_R$ = {a:.1f} $\pm$ {b:.1f} KHz'.format(a=delta_f_phase_C/1000,b = delta_f_phase_C_err/1000), size=13)
ax[0].text(238, 0.5, r'$\chi^2$/DOF = {m:.1f} / {i:.0f}'.format(m=chi2_min_phase, i = df), size=13)

ax[0].set_ylabel(r'$\Delta \Phi \, = \, 2 \, \pi \, f \, \Delta T$', size = 13)
ax[0].set_xlabel(r'Frequency $[\, kHz \, ]$', size = 13)
#ax.set_xlim(189, 265)
#ax.set_ylim(0.055, 0.38)
ax[0].legend(prop={'size': 14}, loc='upper left')

ax[1].errorbar(freq,residual_phase,yerr=phase_residual_err, fmt='o', label=r'Phase residuals',ms=3,color='black', zorder = 2, lw = 1.5)
ax[1].plot(fit_freq, phase_residual_array, label=r'Phase weighted mean', color='tab:red', lw = 1.5, zorder = 1)
ax[1].fill_between(fit_freq, y_fit_down, y_fit_upper, color='gold', alpha=0.5, label = r'Weighted mean $1\sigma$ deviation', zorder = 0, lw = 0)

ax[1].set_ylim(-0.055, 0.04)

# Plotting some text about residuals weighted mean and compatibility with zero
ax[1].text(220, -0.04, r'$\mu_{{\,\text{{residual}}}}$ = {e:.3f} $\pm$ {f:.3f} $mV$'.format(e=weighted_mean_phase_residual, f = weighted_mean_phase_residual_std), size=13)

ax[1].set_ylabel(r'$Residual$', size = 13)
ax[1].set_xlabel(r'Frequency $[\, kHz \, ]$', size = 13)

plt.savefig('phase_C'+'.png',
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

#-------------------------------------------------------------
# RESULTS AND OTHER ANALYSIS
#-------------------------------------------------------------

# region - Defining punctual result for resonance frequency

# Computing theorical values
f_theory = 1/(2*np.pi*np.sqrt(L*C)) # theoretical resonance frequency
ef_theory = np.sqrt((eL*0.5/L)**2/(C*L) + (eC*0.5/C)**2/(C*L))/(2*np.pi) 
delta_f_theory = R/(2*np.pi*L) # theoretical bandwidth
edelta_f_theory = np.sqrt((eR/L)**2 + (R*eL/L**2)**2)/(2*np.pi)

# punctual resonance frequency for phase = 0
f_punc_phase = (236.791 + 236.691)/2
ef_punt_phase = (236.791 - 236.691)/np.sqrt(12) # standard deviation for uniform distribution

# punctual bandwidth for phase = 0
f1_punc_phase = (242.870 + 242.860)/2
ef1_punc_phase = (242.870 - 242.860)/np.sqrt(12) # standard deviation for uniform distribution
f2_punc_phase = (230.490 + 230.550)/2
ef2_punc_phase = (230.550 -230.490)/np.sqrt(12) # standard deviation for uniform distribution

# punctual resonance frequency for phase = 0 from bandwidth
#f_punc_phase_bw = np.sqrt(f1_punc_phase*f2_punc_phase)
#ef_punc_phase_bw = np.sqrt(ef1_punc_phase**2*(f2_punc_phase/f1_punc_phase)/4 + ef2_punc_phase**2*(f1_punc_phase/f2_punc_phase)/4)
f_punc_phase_bw = (f1_punc_phase + f2_punc_phase)/2
ef_punc_phase_bw = np.sqrt((ef1_punc_phase)**2 + (ef2_punc_phase)**2)/2

# punctual resonance frequency for amplification = max
f_punc_amp = (236.795 + 236.697)/2
ef_punc_amp = (236.795 - 236.697)/np.sqrt(12) # standard deviation for uniform distribution

# endregion

# region - Result plot for resonance frequency

# Creating the list with frequency results

# punctual for phase, for bandwidth phase, amplification, fit amp on R, fit phase on R, fit amp on C, fit phase on C
result_freq = [f_punc_phase, f_punc_phase_bw, f_punc_amp, f_T/1000, f_phase/1000]
result_freq_err = [ef_punt_phase, ef_punc_phase_bw, ef_punc_amp, f_T_err/1000, f_phase_err/1000]

# Computing weighted mean
weights = 1 / np.array(result_freq_err)**2  
weighted_mean_1 = np.sum(np.array(result_freq) * weights) / np.sum(weights)  
weighted_mean_err_1 = np.sqrt(1 / np.sum(weights)) 

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), sharex=True, constrained_layout=True)

# Defining the x axis for the plot
IDresult = np.linspace(1, len(result_freq), num = len(result_freq))

# defining points and fit function 
ax.errorbar(IDresult,result_freq,xerr=0, yerr = result_freq_err, fmt='o', label=r'Results', color='firebrick', zorder = 2, lw =1.5, markersize=6)
ax.axhline(y=weighted_mean_1, color='black', linestyle='--', linewidth=1.7, label=r'Weighted mean resonance frequency')
plt.fill_between(np.linspace(0, len(result_freq)+1, num = len(result_freq)), weighted_mean_1 - weighted_mean_err_1,  weighted_mean_1 + weighted_mean_err_1, color='silver', alpha=0.4, label = r'Mean standard deviation')

# setting limit for y axis and the axis labels
ax.set_ylabel(r'$Resonance \, \, frequency \, [KHz]$', size = 18)
ax.set_xlabel(r'$ID_{\text{measure}}$', size = 18)
ax.set_xlim([0 ,len(IDresult) + 1])
ax.set_ylim([236.5, 237])

# Plotting the legend
ax.legend(prop={'size': 17}, loc='upper left', frameon=False).set_zorder(2)

# Add text under each point
text = [r'$f_{\Phi \, - \, P}$', r'$f_{\Phi \, - \, BW}$', r'$f_{A \, - \, P}$', r'$f_{A \, - \, R}$', r'$f_{\Phi \, - \, R}$']

a = 236.55
ax.text(0.8, a, text[0], fontsize=17)
ax.text(1.75, a, text[1], fontsize=17)
ax.text(2.85, a, text[2], fontsize=17)
ax.text(3.85, a, text[3], fontsize=17)
ax.text(4.8, a, text[4], fontsize=17)


plt.savefig('result_f'+'.png',
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - Results plot for Q value 

# Computing Q value using total experimental resistance(no theorical)
Q_Rtot_exp = np.sqrt(L/C)/(Rtot_exp_nog)
eQ_Rtot_exp = np.sqrt((eRtot_exp/Rtot_exp_nog**2)**2*L/C + (eL/(C*Rtot_exp_nog))**2/(4*(L/C)) + (eC*L/(Rtot_exp_nog*C**2))**2/(4*(L/C)))
Q_theory = np.sqrt(L/C)/(R)
eQ_theory = np.sqrt((eR/R**2)**2*L/C + (eL/(C*R))**2/(4*(L/C)) + (eC*L/(R*C**2))**2/(4*(L/C)))
print('Q_Rtot_exp = ', Q_Rtot_exp, ' +/- ', eQ_Rtot_exp)
print('Q_theory = ', Q_theory, ' +/- ', eQ_theory)

# Creating the list with Q results
result_Q = [Q_T_R, Q_phase_R, Q_T_C, Q_phase_C]  
result_Q_err = [eQ_T_R, eQ_phase_R, eQ_T_C, eQ_phase_C]

# Computing weighted mean
weights_Q = 1 / np.array(result_Q_err)**2  
weighted_mean_Q = np.sum(np.array(result_Q) * weights_Q) / np.sum(weights_Q)  
weighted_mean_Q_err = np.sqrt(1 / np.sum(weights_Q)) 

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), sharex=True, constrained_layout=True)

# Defining the x axis for the plot
IDresult_Q = np.linspace(1, len(result_Q), num=len(result_Q))

# Defining points and fit function
ax.errorbar(IDresult_Q, result_Q, xerr=0, yerr=result_Q_err, fmt='o', label=r'Results', color='blue', zorder=2, lw=1.5, markersize=6)
ax.axhline(y=weighted_mean_Q, color='black', linestyle='--', linewidth=1.7, label=r'Weighted mean Q')
plt.fill_between(np.linspace(0, len(result_Q)+1, num=len(result_Q)), weighted_mean_Q - weighted_mean_Q_err, weighted_mean_Q + weighted_mean_Q_err, color='silver', alpha=0.4, label=r'Mean standard deviation')

# Setting limit for y axis and the axis labels
ax.set_ylabel(r'$Q \, \, value$', size=18)
ax.set_xlabel(r'$ID_{\text{measure}}$', size=18)
ax.set_xlim([0, len(IDresult_Q) + 1])
ax.set_ylim([17.4, 20.8])  

# Plotting the legend
ax.legend(prop={'size': 17}, loc='upper left', frameon=False).set_zorder(2)

# Add text under each point
text_Q = [r'$Q_{A\, - \, R}$', r'$Q_{\Phi \, - \, R}$', r'$Q_{A \, - \, C}$', r'$Q_{\Phi \, - \, C}$']

a =  17.6
ax.text(0.8, a, text_Q[0], fontsize=17)
ax.text(1.75, a, text_Q[1], fontsize=17)
ax.text(2.8, a, text_Q[2], fontsize=17)
ax.text(3.85, a, text_Q[3], fontsize=17)

# Save the plot
plt.savefig('result_Q'+'.png',
            pad_inches=1,
            transparent=True,
            facecolor="w",
            edgecolor='w',
            orientation='Portrait',
            dpi=100)

# endregion

# region - Plotting final values for resonance frequency and Q value

print(" ")
print("============== FINAL VALUES ========================================")
print(r'$f_R$ = {a:.2f} $\pm$ {b:.2f} KHz'.format(a=weighted_mean_1, b=weighted_mean_err_1))
print(r'$Q$ = {a:.2f} $\pm$ {b:.2f}'.format(a=weighted_mean_Q, b=weighted_mean_Q_err))
print("====================================================================")

# Computing and plotting weighted mean for Q without the anomaly
# Creating the list with Q results
result_Q = [Q_T_R, Q_phase_R, Q_T_C, Q_phase_C]  
result_Q_err = [eQ_T_R, eQ_phase_R, eQ_T_C, eQ_phase_C]

# Computing weighted mean
weights_Q = 1 / np.array(result_Q_err)**2  
weighted_mean_Q = np.sum(np.array(result_Q) * weights_Q) / np.sum(weights_Q)  
weighted_mean_Q_err = np.sqrt(1 / np.sum(weights_Q)) 

# endregion

#plt.show()