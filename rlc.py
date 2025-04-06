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

# Enable debug mode
DEB = False


# Function definition


def fitf(t, A, B, C, v0, t0):
    """A = amplitude, B = Omega0, C = tau, v0 = vertical offset, t0 = horizontal offset"""
    x = t-t0
    Omega = np.sqrt(B**2-1/C**2)  
    #fitval = A/Omega*np.exp(-x/C)*(1/C**2+Omega**2)*np.sin(Omega*x)+v0
    fitval = (A/Omega)*np.exp(-x/C)*np.sin(Omega*x)+v0
    return fitval

def fitf2(t, A, B, C):
    """A = ampplitude, B = Omega0, C = tau"""
    x = t
    Omega = np.sqrt(B**2-1/C**2)  # 
   # fitval = A/Omega*np.exp(-x/C)*(1/C**2+Omega**2)*np.sin(Omega*x)
    fitval = (A/Omega)*np.exp(-x/C)*np.sin(Omega*x)
    return fitval

def fit_exp(t, A, B, C, v0, t0):
    """A = amplitude, B = Omega0, C = tau, v0 = vertical offset, t0 = horizontal offset"""
    x = t-t0
    Omega = np.sqrt(B**2-1/C**2)  
    #fitval = A/Omega*np.exp(-x/C)*(1/C**2+Omega**2)*np.sin(Omega*x)+v0
    fitval = (A/Omega)*np.exp(-x/C)+v0
    return fitval

def fit_exp2(t, A, B, C):
    """A = ampplitude, B = Omega0, C = tau"""
    x = t
    Omega = np.sqrt(B**2-1/C**2)
    return (A/Omega)*np.exp(-x/C)

def linear(t, m, qy, qx):
    """ y = m(x-qx) - qy"""
    x = t
    return m*(x-qx) + qy

def fitchi2(i,j,k):
    x = tempo
    y = Vout
    y_err = eVout
    AA,BB,CC = A_chi[i],B_chi[j],C_chi[k]
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
    #fitval = (A * (OM**2)) / np.sqrt(omega**4 - 2.0 * omega**2 * (OM**2 - 2.0 * delta**2) + OM**4)
    #residuals = (y - fitval) / y_err
    residuals = (y - fitf2(tempo,AA,BB,CC))
    chi2 = np.sum((residuals / y_err)**2)
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
    """Return quadratic sum of statistical and systematic error - [Resistance]"""
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

## PARAMATERS

# Number of bins for parameter scan
NI = 20
NJ = 20
NK = 20

# region - LOADING DATA, COMPUTING ERRORS AND SETTING INITIAL FIT VALUES 

# Input file name
file = 'data'
inputname = file+'.txt'

# metrix values
C_metrix = 6.55e-9 #nF
R_metrix= 11.63 #ohm
R_metrix_err = computing_R_err(R_metrix)
C_metrix_err = computing_C_err(C_metrix)
L_theory = 500e-6 #muhenry

# Initial parameter values
amplitude = 0.04
Binit = 1/np.sqrt(L_theory*C_metrix)  # Hz --> omega0
Cinit = 2*L_theory/R_metrix # s --> tau
Ainit= amplitude*np.sqrt(Binit**2-1/Cinit**2)
v0init = 0.0
t0init = 0.00000024

# Scale conversion factors
mu = 1000000  # 1 s = 1e6 us

# Assumed scale value
Vscale = 0.010  # V
Tscale = 2.5e-6  # s

# Assumed reading errors
"""letturaV = 0.002*0.41
errscalaV = 0.03*0.41
letturaT = 2.5e-6*0.41"""
#letturaV = Vscale/(25*np.sqrt(24))
letturaV = Vscale*0.41/25
letturaT = Tscale*0.41/25
errscalaV = 1.5/100


# Read data from the input file
data = np.loadtxt(inputname).T
tempo = data[0] #tempo
Vout = data[1] #Vout

# Number of points to fit
# mesure the positive frequency of tempo vector
n = len(tempo[tempo > 0])

# Calculate errors on x and y
eVout = np.sqrt((letturaV)**2 + ((errscalaV * Vout)**2))
etempo = letturaT
# endregion

# region - PLOT DATA WITHOUT FIT

# Plot  Vout vs. tempo 
# defining the plot 
fig, ax = plt.subplots(1, 1, figsize=(6.5,6.5),sharex=True, constrained_layout = True)
ax.errorbar(tempo*mu,Vout,yerr=eVout,xerr=etempo*mu, fmt='o', label=r'$Data$',ms=2)

# defining axis label
ax.set_ylabel(r'$V_{R}\,\, [\, V \,]$', size = 15)
ax.set_xlabel(r'$Time \, [\, \mu s \,]$', size = 15)

# plotting a zero line to see better the vertical offset
ax.plot(tempo*mu, np.zeros(len(tempo)), linestyle = '-', color = 'steelblue', lw = 1, label = r'$V_{{0 - theorical}}$ = 0 V')

ax.legend(prop={'size': 15}, loc='best')

OFFSET = 0.044
# some text --> report the metrix values
ax.text(25, 0.03-OFFSET, r'$C_{{\,\text{{METRIX}}}}$ = {e:.2f} $\pm$ {f:.2f} $nF$'.format(e=C_metrix*10**9, f = C_metrix_err*10**9), size=15)
ax.text(25, 0.027-OFFSET, r'$R_{{\,\text{{METRIX}}}}$ = {e:.2f} $\pm$ {f:.2f} $\Omega$'.format(e=R_metrix, f = R_metrix_err), size=15)
"""ax.text(31, 0.024-OFFSET, r'$\omega_0$ = {e:.0f} $\pm$ {f:.0f} $KHz$'.format(e=Binit/1000, f = np.sqrt((C_metrix_err**2)/(4*L_theory*C_metrix**3))/1000), size=12)
ax.text(31, 0.021-OFFSET, r'$\tau$ = {e:.1f} $\pm$ {f:.1f} $\mu s$'.format(e=Cinit*mu, f = 2*L_theory*R_metrix_err*mu/R_metrix**2), size=12)
ax.text(31, 0.018-OFFSET, r'$L_{{theoretical}}$ = {e:.0f} $\mu H$'.format(e=L_theory*mu), size=12)"""

# saving all
plt.savefig(file+'_no_fit'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# aumatic layout configuration
fig.tight_layout()

#plt.show()

# endregion

# region - PERFORM THE FIT, RESIDUALS AND ERRORS

# Perform the first fit to define t0 and v0

popt, pcov = curve_fit(fitf, tempo, Vout, p0=[Ainit, Binit, Cinit, v0init, t0init], method='lm', sigma=eVout, absolute_sigma=True)

"""
POPT: Fit parameters vector
PCOV: Covariance matrix
bounds sono i limiti inferiori e superiori dei parametri (si richiede positività delle stime in questo caso)
"""

N = len(tempo)
residuA = Vout - fitf(tempo, *popt)

x_fit = np.linspace(min(tempo), max(tempo), 1000)

perr = np.sqrt(np.diag(pcov))

# variables error and chi2
perr = np.sqrt(np.diag(pcov))
chisq = np.sum((residuA/eVout)**2)

# degrees of freedom
df = N - 5

"""
plotting the fit with 1000 points between the min and max frequency
"""

# fitting parameters and errors
A_BF, B_BF, C_BF, v0_BF, t0_BF = popt #fit parameters
eA_BF, eB_BF, eC_BF, ev0_BF, et0_BF = np.sqrt(np.diag(pcov))

# Plot the first fit to define t0
# defining the plot
fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, constrained_layout = True, height_ratios=[2, 1])

#ax[0].plot(x_fit,fitf(x_fit,Ainit,Binit,Cinit,v0init,t0init), label='init guess', linestyle='dashed', color='green')
# plotting data
ax[0].errorbar(tempo*mu,Vout,yerr=eVout,xerr=etempo*mu, fmt='o', label=r'$Data$',ms=1,color='darkorange')

# plotting the fitting function
ax[0].plot(x_fit*mu, fitf(x_fit, *popt), label='Fit', linestyle='--', color='black', lw = 1.5)

# plotting the exp positive function
ax[0].plot(x_fit*mu, fit_exp(x_fit, *popt), label='Exponential fit', linestyle='--', color='darkcyan', lw = 1.5)

# plotting the exp negative function
ax[0].plot(x_fit*mu, -fit_exp(x_fit, *popt)+2*v0_BF, linestyle='--', color='darkcyan', lw = 1.5)

# setting axis labels
ax[0].set_ylabel(r'$V_{R} [\, V \,]$', size = 14)
ax[0].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)

# plotting residual graph 
ax[1].errorbar(tempo*mu,residuA*1000,yerr=eVout*1000, fmt='o', label=r'Residuals$',ms=2,color='darkorange', zorder = 0)
ax[1].set_ylabel(r'Residuals [$\, mV\, $]', size = 14)
ax[0].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)
ax[1].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)
ax[1].plot(tempo*mu,np.full((len(tempo)), statistics.mean(residuA)), lw = 2, color = 'black', zorder = 1)

# plotting a line y = V_0
ax[0].plot(tempo*mu, np.full((len(tempo)), v0_BF), linestyle = '-', lw = 1.5,  color = 'black', label = r'Vertical offset - $V_0$')

ax[0].legend(loc='best', fontsize=13, framealpha=0.8, ncol = 2)

OFFSET = -0.014
# Adding some text
ax[0].text(13, 0.024-OFFSET, r'$\omega_0$ = {e:.1f} $\pm$ {f:.1f} $KHz$'.format(e=B_BF/1000, f = eB_BF/1000), size=13)
ax[0].text(38, 0.024-OFFSET,r'$\tau$ = {e:.2f} $\pm$ {f:.2f} $\mu s$'.format(e=C_BF*mu, f = eC_BF*mu), size=13)
ax[0].text(13, 0.0175-OFFSET, r'$V_0$ = {g:.2} $\pm$ {h:.1} mV'.format(g=v0_BF*1000, h=ev0_BF*1000), size=13)
ax[0].text(38, 0.0175-OFFSET, r'$t_{{\,0}}$ = ({i:.4} $\pm$ {l:.1}) $\mu s$'.format(i=t0_BF*mu, l=et0_BF*mu), size=13)
ax[0].text(13, 0.011-OFFSET, r'$\chi^2$/DOF = {m:.0f} / {i:.0f}'.format(m=chisq, i = df), size=13)
ax[0].text(38, 0.011-OFFSET, r'$r_{{V_0 \, / \, 0}}$ = {p:.0f}'.format(p=compatibility(v0_BF, 0, ev0_BF, 0)), size=13)
ax[1].text(13, 1, r'$\mu_{{\,\text{{residual}}}}$ = {e:.2f} $\pm$ {f:.2f} $mV$'.format(e=statistics.mean(residuA)*1000, f = statistics.stdev(residuA)*1000), size=13)
ax[1].text(43, 1, r'$r_{{\, \mu \, / \, 0}}$ = {e:.2f}'.format(e=compatibility(statistics.mean(residuA), 0, statistics.stdev(residuA), 0)), size=13)

plt.savefig(file+'_scipy_offset'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#fig.tight_layout()

#plt.show()
# endregion

# region - EXTRACT AND PRINT BEST FIT PARAMETERS AND ERRORS

print("============== BEST FIT with SciPy - Vertical/Horizontal offset on ====================")
print(r'A = ({a:.3e} +/- {b:.1e})'.format(a=A_BF,b=eA_BF))
print(r'B = ({c:.5e} +/- {d:.1e}) kHz'.format(c=B_BF * 1e-3, d=eB_BF * 1e-3))
print(r'C = ({e:.3e} +/- {f:.1e}) ms'.format(e=C_BF * 1e3, f=eC_BF * 1e3))
print(r'v0 = ({g:.4e} +/- {h:.1e}) mV'.format(g=v0_BF * 1e3, h=ev0_BF * 1e3))
print(r't0 = ({i:.4e} +/- {l:.1e}) ms'.format(i=t0_BF * 1e3, l=et0_BF * 1e3))
print(r'chisq/DOF = {m:.2f} / {i:.0f}'.format(m=chisq, i = df))
print("=======================================================================================")
# endregion

# region - PERFORM THE FIT WITHOUT t0 or V0[HORIZONTAL OFFSET]

# Perform the fit to masked from t0

shift= 0.0
Vout = Vout[tempo>t0_BF+shift]-v0_BF#+0.0035 #Vout
eVout = eVout[tempo>t0_BF+shift]
tempo = tempo[tempo>t0_BF+shift]-t0_BF
N = len(tempo)


popt, pcov = curve_fit(fitf2, tempo, Vout, p0=[Ainit, Binit, Cinit], method='lm', sigma=eVout, absolute_sigma=True)

"""
POPT: Vettore con la stima dei parametri dal fit
PCOV: Matrice delle covarianze
bounds sono i limiti inferiori e superiori dei parametri (si richiede positività delle stime in questo caso)
"""

perr = np.sqrt(np.diag(pcov))
#print( ' ampiezza = {a:.3f} +/- {b:.3f} \n delta = {c:.1f} +/- {d:.1f} kHz \n Omega = {e:.1f} +/- {f:.1f} kHz '.format(a=popt[0], b=perr[0],c=popt[1]/1000,d=perr[1]/1000,e=popt[2]/1000,f=perr[2]/1000))

residuA = Vout - fitf2(tempo, *popt)
# variables error and chi2
perr = np.sqrt(np.diag(pcov))
chisq = np.sum((residuA/eVout)**2)
df = N - 3


x_fit = np.linspace(min(tempo), max(tempo), 1000)

"""
fit tracciato con mille punti fra la freq min e max
"""
# uncomment if you want to plot the fit done with scipy but without t0 and v0
"""# defining the plot
fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, constrained_layout = True, height_ratios=[2, 1])

#ax[0].plot(x_fit,fitf(x_fit,Ainit,Binit,Cinit,v0init,t0init), label='init guess', linestyle='dashed', color='green')
# plotting data
ax[0].errorbar(tempo*mu,Vout,yerr=eVout,xerr=etempo*mu, fmt='o', label=r'$Data$',ms=1,color='darkorange')

# plotting the fitting function
ax[0].plot(x_fit*mu, fitf2(x_fit, *popt), label='Fit', linestyle='--', color='black', lw = 1.5)

# plotting the exp positive function
ax[0].plot(x_fit*mu, fit_exp2(x_fit, *popt), label='Exponential fit', linestyle='--', color='darkcyan', lw = 1.5)

# plotting the exp negative function
ax[0].plot(x_fit*mu, -fit_exp2(x_fit, *popt), linestyle='--', color='darkcyan', lw = 1.5)

# setting axis labels
ax[0].set_ylabel(r'$V_{R} [\, V \,]$', size = 14)
ax[0].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)

# plotting residual graph 
ax[1].errorbar(tempo*mu,residuA*1000,yerr=eVout*1000, fmt='o', label=r'Residuals$',ms=2,color='darkorange')
ax[1].set_ylabel(r'Residuals [$\, mV\, $]', size = 14)
ax[0].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)
ax[1].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)
ax[1].plot(tempo*mu,np.full((len(tempo)), statistics.mean(residuA)), lw = 2, color = 'black')

# plotting a zero line to see better the vertical offset
ax[0].plot(tempo*mu, np.zeros(len(tempo)), linestyle = '-', color = 'black', lw = 1.5, label = r'$V_{{0 - theorical}}$ = 0 V')

ax[0].legend(loc='best', fontsize=13, framealpha=0.8, ncol = 2)

OFFSET = -0.014
# Adding some text
ax[0].text(13, 0.024-OFFSET, r'$\omega_0$ = {e:.1f} $\pm$ {f:.1f} $KHz$'.format(e=B_BF/1000, f = eB_BF/1000), size=13)
ax[0].text(38, 0.024-OFFSET,r'$\tau$ = {e:.2f} $\pm$ {f:.2f} $\mu s$'.format(e=C_BF*mu, f = eC_BF*mu), size=13)
ax[0].text(13, 0.011-OFFSET, r'$\chi^2$/DOF = {m:.0f} / {i:.0f}'.format(m=chisq, i = df), size=13)
ax[1].text(13, 1, r'$\mu_{{\,\text{{residual}}}}$ = {e:.2f} $\pm$ {f:.2f} $mV$'.format(e=statistics.mean(residuA)*1000, f = statistics.stdev(residuA)*1000), size=13)
ax[1].text(43, 1, r'$r_{{\, \mu \, / \, 0}}$ = {e:.3f}'.format(e=compatibility(statistics.mean(residuA), 0, statistics.stdev(residuA), 0)), size=13)

plt.savefig(file+'_scipy_no_offset'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#fig.tight_layout()

plt.show()"""

# endregion

# region - EXTRACT AND PRINT BEST FIT PARAMETERS AND ERRORS
A_BF, B_BF, C_BF = popt #parametri del best fit
eA_BF, eB_BF, eC_BF = np.sqrt(np.diag(pcov))

print("============== BEST FIT with SciPy - Vertical/Horizontal offset off ===================")
print(r'A = ({a:.3e} +/- {b:.1e})'.format(a=A_BF,b=eA_BF))
print(r'B = ({c:.5e} +/- {d:.1e}) kHz'.format(c=B_BF * 1e-3, d=eB_BF * 1e-3))
print(r'C = ({e:.3e} +/- {f:.1e}) ms'.format(e=C_BF * 1e3, f=eC_BF * 1e3))
print(r'chisq/DOF = {m:.2f} / {i:.0f}'.format(m=chisq, i = df))
print("=======================================================================================")

# Define the interval for parameter limits
NSI = 3 #2?
A0, A1 = A_BF - NSI * eA_BF, A_BF + NSI * eA_BF
B0, B1 = B_BF - NSI * eB_BF, B_BF + NSI * eB_BF
C0, C1 = C_BF - NSI * eC_BF, C_BF + NSI * eC_BF

print("INTERVAL OF VALUES IN ONE SIGMA DEVIATION")
print(f"(A0 = {A0}, A1 = {A1})")
print(f"(B0 = {B0}, B1 = {B1}) Hz")
print(f"(C0 = {C0}, C1 = {C1}) s")

# endregion

# this will return also a non symmetric error on parameters
# region - FIRST FITTING USING THE CHI SQUARE MINIMIZATION [WITHOUT USING PYTHON LIBRARIES]

step = 100

A_chi = np.linspace(A0,A1,step)
B_chi = np.linspace(B0,B1,step)
C_chi = np.linspace(C0,C1,step)

mappa = np.zeros((step,step,step))
item = [(i,j,k) for i in range(step) for j in range(step) for k in range(step)]

# assign the global pool
pool = multiprocessing.pool.ThreadPool(100)
# issue top level tasks to pool and wait
pool.starmap(fitchi2, item, chunksize=10)
# close the pool
pool.close()

mappa = np.asarray(mappa)            
#print(mappa.shape)
#print(np.argmin(mappa))

chi2_min = np.min(mappa)
#print(mappa.shape,np.unravel_index(np.argmin(mappa),mappa.shape),chi2_min)

"""
fit tracciato min chi2
"""
argchi2_min = np.unravel_index(np.argmin(mappa),mappa.shape)

residui_chi2 = Vout - fitf2(tempo,A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]])
chisq_res = np.sum((residui_chi2/eVout)**2)

#print(chi2_min,argchi2_min, chisq_res)

#endregion

# region -  PRINTING FIT PARAMETERS AND COMPUTING CHI SQUARE 3D MAP
#Calcolo Profilazione 2D
chi2D = profi2D(1,mappa) # --> Mappa tau - omega0
Achi2D = profi2D(2,mappa) # --> Mappa A -tau
Bchi2D = profi2D(3, mappa) # --> Mappa A - omega0

#Calcolo Profilazioni 1D
prof_B = profi1D([1,3],mappa) # --> chi square for tau 
prof_C = profi1D([1,2],mappa)  # --> chi square for omega0 
prof_A = profi1D([2,3],mappa)  # --> chi square for A

# trovo l'errore sui parametri

lvl = chi2_min+1. # 2.3 # 3.8
diff_B = abs(prof_B-lvl)
diff_C = abs(prof_C-lvl)
diff_A = abs(prof_A-lvl)

#print(diff_B, diff_B[B_chi<B_BF])

B_dx = np.argmin(diff_B[B_chi<B_BF])
B_sx = np.argmin(diff_B[B_chi>B_BF])+len(diff_B[B_chi<B_BF])
C_dx = np.argmin(diff_C[C_chi<C_BF])
C_sx = np.argmin(diff_C[C_chi>C_BF])+len(diff_C[C_chi<C_BF])
A_dx = np.argmin(diff_A[A_chi<A_BF])
A_sx = np.argmin(diff_A[A_chi>A_BF])+len(diff_A[A_chi<A_BF])

#print(B_dx,B_sx,C_dx,C_sx,A_dx,A_sx)
errA = A_chi[argchi2_min[0]]-A_chi[A_dx]
errAA = A_chi[A_sx]-A_chi[argchi2_min[0]]
errB = B_chi[argchi2_min[1]]-B_chi[B_dx] 
errBB = B_chi[B_sx] -B_chi[argchi2_min[1]]
errC = C_chi[argchi2_min[2]]-C_chi[C_dx]
errCC = C_chi[C_sx]-C_chi[argchi2_min[2]]

#print(A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]])
#print(A_chi)

print("============== BEST FIT with chi2 minimization - NO PYTHON LIBRARIES ==================")
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]] * 1e-3, e=errB * 1e-3, f=errBB* 1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) ms'.format(g=C_chi[argchi2_min[2]] * 1e3, h=errC * 1e3,  n=errCC * 1e3))
print(r'chisq/DOF = {m:.2f} / {i:.0f}'.format(m=np.min(mappa), i = len(tempo) -3))
print("=======================================================================================")

# endregion 

# region - PLOT THE FIT WITH CHI SQUARE MINIMIZATION [WITHOUT USING PYTHON LIBRARIES]

# defining the plot
fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5),sharex=True, constrained_layout = True, height_ratios=[2, 1])

#ax[0].plot(x_fit,fitf(x_fit,Ainit,Binit,Cinit,v0init,t0init), label='init guess', linestyle='dashed', color='green')
# plotting data
ax[0].errorbar(tempo*mu,Vout,yerr=eVout,xerr=etempo*mu, fmt='o', label=r'$Data$',ms=1,color='darkorange')

# plotting the fitting function
ax[0].plot(x_fit*mu, fitf2(x_fit, *popt), label='Fit', linestyle='--', color='black', lw = 1.5)

# plotting the exp positive function
ax[0].plot(x_fit*mu, fit_exp2(x_fit, *popt), label='Exponential fit', linestyle='--', color='darkcyan', lw = 1.5)

# plotting the exp negative function
ax[0].plot(x_fit*mu, -fit_exp2(x_fit, *popt), linestyle='--', color='darkcyan', lw = 1.5)

# setting axis labels
ax[0].set_ylabel(r'$V_{R} [\, V \,]$', size = 14)
ax[0].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)

# plotting residual graph 
ax[1].errorbar(tempo*mu,residuA*1000,yerr=eVout*1000, fmt='o', label=r'Residuals$',ms=2,color='darkorange', zorder = 0)
ax[1].set_ylabel(r'Residuals [$\, mV\, $]', size = 14)
ax[0].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)
ax[1].set_xlabel(r'Time [$\, \mu s \,$]', size = 14)
ax[1].plot(tempo*mu,np.full((len(tempo)), statistics.mean(residuA)), lw = 2, color = 'black', zorder = 1)

# plotting a zero line to see better the vertical offset
ax[0].plot(tempo*mu, np.zeros(len(tempo)), linestyle = '-', color = 'black', lw = 1.5, label = r'$V_{{0 - theorical}}$ = 0 V')

ax[0].legend(loc='best', fontsize=13, framealpha=0.8, ncol = 2)

OFFSET = -0.01
# Adding some text
ax[0].text(27, 0.024-OFFSET, r'$\omega_0$ = {d:.2f} - {e:.2f} + {f:.2f} kHz'.format(d=B_chi[argchi2_min[1]]/1000, e=errB/1000, f=errBB/1000), size=13)
ax[0].text(27, 0.017-OFFSET,r'$\tau$ = {g:.2f} - {h:.2f} + {n:.2f} $\mu s$'.format(g=C_chi[argchi2_min[2]]*mu, h=errC*mu,  n=errCC*mu), size=13)
ax[0].text(27, 0.010-OFFSET, r'$\chi^2$/DOF = {m:.0f} / {i:.0f}'.format(m=chisq, i = df), size=13)
ax[1].text(13, 1, r'$\mu_{{\,\text{{residual}}}}$ = {e:.2f} $\pm$ {f:.2f} $mV$'.format(e=statistics.mean(residuA)*1000, f = statistics.stdev(residuA)*1000), size=13)
ax[1].text(43, 1, r'$r_{{\, \mu \, / \, 0}}$ = {e:.2f}'.format(e=compatibility(statistics.mean(residuA), 0, statistics.stdev(residuA), 0)), size=13)

plt.savefig(file+'_N0_scipy_no_offset'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#fig.tight_layout()

#plt.show()

# endregion

# region - PLOTTING CHI SQUARE 3D MAP

cmap = mpl.colormaps['viridis'].reversed()
#print(chi2D.shape)
level = np.linspace(np.min(chi2D),np.max(chi2D),100)
line_c = 'gray'

# plot definition
fig, ax = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True, 
                       height_ratios=[3, 1], width_ratios=[1, 3], sharex='col', sharey='row')

# some correction to measure units 
B_chi = B_chi/1000
C_chi = C_chi*mu

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
ax[0,1].clabel(CS, inline=True, fontsize=11, fmt='%10.0f')

# plotting deviation bars of parameters
ax[0,1].plot([B0/1000,B1/1000],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B0/1000,B1/1000],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_sx],B_chi[B_sx]],[C0*mu,C1*mu],color=line_c, ls='dashed', lw = 1.5)
ax[0,1].plot([B_chi[B_dx],B_chi[B_dx]],[C0*mu,C1*mu],color=line_c, ls='dashed', lw = 1.5)

# plotting best parameters
ax[0,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[C0*mu,C1*mu], ls='dashed', lw = 1.5, color = 'black')
ax[0,1].plot([B0/1000,B1/1000],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black', label = 'Best fit')

#plotting the parabolic curve for tau(also standard dev and best parameter)
ax[0,0].plot(prof_B,C_chi,ls='-') 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_sx],C_chi[B_sx]], color=line_c, ls='dashed') 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[B_dx],C_chi[B_dx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black')
ax[0,0].set_xticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])
ax[0,0].text(chi2_min - 14,C_chi[argchi2_min[2]],r'{e:.2f} $\mu s$'.format(e=C_chi[argchi2_min[2]]), color='k',alpha=1, fontsize=10)
ax[0,0].text(chi2_min - 14,C_chi[B_dx],r'{e:.2f} $\mu s$'.format(e=C_chi[B_dx]), color='b',alpha=1, fontsize=10)
ax[0,0].text(chi2_min -14,C_chi[B_sx],r'{e:.2f} $\mu s$'.format(e=C_chi[B_sx]), color='r',alpha=1, fontsize=10)

# plotting the parabolic curve for omega(also standard dev and best parameter)
ax[1,1].plot(B_chi,prof_C) 
ax[1,1].plot([B_chi[C_sx],B_chi[C_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed') 
ax[1,1].plot([B_chi[C_dx],B_chi[C_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed') 
ax[1,1].plot([B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]],[int(chi2_min-2),int(chi2_min+10)], ls='dashed', lw = 1.5, color = 'black') 
ax[1,1].text(B_chi[argchi2_min[1]]-0.032, chi2_min - 7,r'{e:.2f} $KHz$'.format(e=B_chi[argchi2_min[1]]), color='k',alpha=1, fontsize=10)
ax[1,1].text(B_chi[C_dx]-0.06, chi2_min - 7,r'{e:.2f} $KHz$'.format(e=B_chi[C_dx]), color='b',alpha=1, fontsize=10)
ax[1,1].text(B_chi[C_sx], chi2_min -7,r'{e:.2f} $KHz$'.format(e=B_chi[C_sx]), color='r',alpha=1, fontsize=10)
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
ax[0,0].set_ylabel(r'$\tau\left(\mu s\right)$', fontsize = 17) 
ax[0,0].set_xlabel(r'$\chi^2_{{min}}$', fontsize = 17) 
ax[1,1].set_xlabel(r'$\omega_0\left(KHz\right)$', fontsize = 17) 
ax[1,1].set_ylabel(r'$\chi^2_{{min}}$', fontsize = 17) 
ax[0,1].set_xlabel(r'$\omega_0\left(KHz\right)$', fontsize = 17)
ax[0,1].set_ylabel(r'$\tau\left(\mu s\right)$', fontsize = 17) 
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10)) 
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

# adding some text
ax[0, 1].text(B_chi[argchi2_min[1]]+0.032, C_chi[argchi2_min[2]]-0.04,r'$\chi^2_{{best}}$ = {g:.0f}'.format(g=chi2_min), size=14)
"""#ax[0, 1].text(B_chi[argchi2_min[2]]+0.005, int(chi2_min+2), r'$\omega_0$ = {d:.2f} - {e:.2f} + {f:.2f} kHz'.format(d=B_chi[argchi2_min[1]], e=errB/1000, f=errBB/1000), size=13)
ax[0, 0].text(int(chi2_min+2), C_chi[argchi2_min[2]]+0.001,r'$\tau$ = {g:.3f} $\mu s$'.format(g=C_chi[argchi2_min[2]]), size=11)
ax[0, 0].text(int(chi2_min+2), C_chi[C_sx]+0.001,r' + {g:.3f} $\mu s$'.format(g=errC*mu), size=11)
ax[0, 0].text(int(chi2_min+2), C_chi[C_dx]+0.001,r'- {g:.3f} $\mu s$'.format(g=errCC*mu), size=11)"""

# plotting legend
ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.5, 0.95), loc='upper center', fontsize=16)

# Change the size of ticks for 2D plots
ax[0, 0].tick_params(axis='x', labelsize=11)
ax[0, 0].tick_params(axis='y', labelsize=11)

ax[0, 1].tick_params(axis='x', labelsize=11)
ax[0, 1].tick_params(axis='y', labelsize=11)

ax[1, 1].tick_params(axis='x', labelsize=11)
ax[1, 1].tick_params(axis='y', labelsize=11)

plt.savefig(file+'_chi_square_map'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

# region - CHI SQUARE MAP BEETWEEN A AND TAU

# converting A units in mV
A_chi = A_chi/1000

fig, ax = plt.subplots(2, 2, figsize=(8, 8),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
im = ax[0,1].contourf(A_chi,C_chi,Achi2D, levels=level, cmap=cmap)
#cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 
#cbar.set_label(r'$\chi^2$',rotation=360)

# setting the bar label
cbar.set_label(r'$\chi^2$',rotation=360, size = 17)

# Plotting constant chi-square elliptical orbit at Xmin + 1, + 2.3, + 3.8
CS = ax[0, 1].contour(A_chi, C_chi, Achi2D, levels=[chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
                      linewidths=2, colors=['red','darkorange', 'saddlebrown'], alpha=1, linestyles='-')

# Add labels directly to the legend
handles, _ = CS.legend_elements()
labels = [r'$\chi^2_{{min}} + 1$', r'$\chi^2_{{min}} + 2.3$', r'$\chi^2_{{min}} + 3.8$']

# lotting chi square value of elliptical orbits
ax[0,1].clabel(CS, inline=True, fontsize=11, fmt='%10.0f')

#ax[0,1].text(A_chi[np.argmin(prof_A)],C_chi[np.argmin(prof_C)]+50.,str(np.around(chi2_min,1)),color='k',alpha=0.5, fontsize=9)
ax[0,1].plot([A0/1000,A1/1000],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed')
ax[0,1].plot([A0/1000,A1/1000],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed')
ax[0,1].plot([A_chi[A_sx],A_chi[A_sx]],[C0*mu,C1*mu],color=line_c, ls='dashed')
ax[0,1].plot([A_chi[A_dx],A_chi[A_dx]],[C0*mu,C1*mu],color=line_c, ls='dashed')

#ax[0,0].plot(prof_A,C_chi) 
ax[0,0].plot(prof_B,C_chi) 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[A_sx],C_chi[A_sx]], color=line_c, ls='dashed') 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[A_dx],C_chi[A_dx]], color=line_c, ls='dashed') 

#ax[1,1].plot(A_chi,prof_C) 
ax[1,1].plot(A_chi,prof_A) 
ax[1,1].plot([A_chi[C_sx],A_chi[C_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed') 
ax[1,1].plot([A_chi[C_dx],A_chi[C_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
#ax[1,1].text(A_chi[np.argmin(prof_A)]-0.001,50,str(np.around(A_chi[np.argmin(prof_A)],3)), color='k',alpha=0.5, fontsize=9)
#ax[1,1].text(A_chi[C_sx]+0.001,54.5,str(np.around(A_chi[C_sx]-A_chi[np.argmin(prof_A)],3)), color='b',alpha=0.5, fontsize=9)
#ax[1,1].text(A_chi[C_dx]-0.006,54.5,str(np.around(A_chi[C_dx]-A_chi[np.argmin(prof_A)],3)), color='r',alpha=0.5, fontsize=9)

# plotting best parameters
ax[0,1].plot([A_chi[argchi2_min[0]],A_chi[argchi2_min[0]]],[C0*mu,C1*mu], ls='dashed', lw = 1.5, color = 'black')
ax[0,1].plot([A0/1000,A1/1000],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black', label = 'Best fit')
ax[1,1].plot([A_chi[argchi2_min[0]],A_chi[argchi2_min[0]]],[int(chi2_min-2),int(chi2_min+10)], ls='dashed', lw = 1.5, color = 'black')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[argchi2_min[2]],C_chi[argchi2_min[2]]], ls='dashed', lw = 1.5, color = 'black')

# setting axis labels 
ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$\tau\left(\mu s\right)$', fontsize = 18) 
ax[0,0].set_xlabel(r'$\chi^2_{{min}}$', fontsize = 18) 
ax[1,1].set_xlabel(r'$A \, [\, \frac{mV}{\tau s}\,]$', fontsize = 18) 
ax[1,1].set_ylabel(r'$\chi^2_{{min}}$', fontsize = 18) 
ax[0,1].set_xlabel(r'$A \, [\, \frac{mV}{\tau s}\,]$', fontsize = 18)
ax[0,1].set_ylabel(r'$\tau\left(\mu s\right)$', fontsize = 18) 
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10)) 
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

# plotting legend
ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.5, 0.95), loc='upper center', fontsize=16)

# Change the size of ticks for 2D plots
ax[0, 0].tick_params(axis='x', labelsize=11)
ax[0, 0].tick_params(axis='y', labelsize=11)

ax[0, 1].tick_params(axis='x', labelsize=11)
ax[0, 1].tick_params(axis='y', labelsize=11)

ax[1, 1].tick_params(axis='x', labelsize=11)
ax[1, 1].tick_params(axis='y', labelsize=11)

#plt.show()

plt.savefig(file+'_A_tau_map'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - CHI SQUARE MAP BEETWEEN A AND OMEGA

fig, ax = plt.subplots(2, 2, figsize=(8, 8),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
im = ax[0,1].contourf(A_chi,B_chi,Bchi2D, levels=level, cmap=cmap)
#cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 
#cbar.set_label(r'$\chi^2$',rotation=360)

# setting the bar label
cbar.set_label(r'$\chi^2$',rotation=360, size = 17)

# Plotting constant chi-square elliptical orbit at Xmin + 1, + 2.3, + 3.8
CS = ax[0, 1].contour(A_chi, B_chi, Bchi2D, levels=[chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
                      linewidths=2, colors=['red','darkorange', 'saddlebrown'], alpha=1, linestyles='-')

# Add labels directly to the legend
handles, _ = CS.legend_elements()
labels = [r'$\chi^2_{{min}} + 1$', r'$\chi^2_{{min}} + 2.3$', r'$\chi^2_{{min}} + 3.8$']

# lotting chi square value of elliptical orbits
ax[0,1].clabel(CS, inline=True, fontsize=11, fmt='%10.0f')

#ax[0,1].text(A_chi[np.argmin(prof_A)],C_chi[np.argmin(prof_C)]+50.,str(np.around(chi2_min,1)),color='k',alpha=0.5, fontsize=9)
ax[0,1].plot([A0/1000,A1/1000],[B_chi[C_sx],B_chi[C_sx]],color=line_c, ls='dashed')
ax[0,1].plot([A0/1000,A1/1000],[B_chi[C_dx],B_chi[C_dx]],color=line_c, ls='dashed')
ax[0,1].plot([A_chi[A_sx],A_chi[A_sx]],[B0/1000,B1/1000],color=line_c, ls='dashed')
ax[0,1].plot([A_chi[A_dx],A_chi[A_dx]],[B0/1000,B1/1000],color=line_c, ls='dashed')

#ax[0,0].plot(prof_A,B_chi) 
ax[0,0].plot(prof_C,B_chi) 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[B_chi[A_sx],B_chi[A_sx]], color=line_c, ls='dashed') 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[B_chi[A_dx],B_chi[A_dx]], color=line_c, ls='dashed') 

#ax[1,1].plot(A_chi,prof_B)
ax[1,1].plot(A_chi,prof_A)  
ax[1,1].plot([A_chi[C_sx],A_chi[C_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed') 
ax[1,1].plot([A_chi[C_dx],A_chi[C_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')
#ax[1,1].text(A_chi[np.argmin(prof_A)]-0.001,50,str(np.around(A_chi[np.argmin(prof_A)],3)), color='k',alpha=0.5, fontsize=9)
#ax[1,1].text(A_chi[C_sx]+0.001,54.5,str(np.around(A_chi[C_sx]-A_chi[np.argmin(prof_A)],3)), color='b',alpha=0.5, fontsize=9)
#ax[1,1].text(A_chi[C_dx]-0.006,54.5,str(np.around(A_chi[C_dx]-A_chi[np.argmin(prof_A)],3)), color='r',alpha=0.5, fontsize=9)

# plotting best parameters
ax[0,1].plot([A_chi[argchi2_min[0]],A_chi[argchi2_min[0]]],[B0/1000,B1/1000], ls='dashed', lw = 1.5, color = 'black')
ax[0,1].plot([A0/1000,A1/1000],[B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]], ls='dashed', lw = 1.5, color = 'black', label = 'Best fit')
ax[1,1].plot([A_chi[argchi2_min[0]],A_chi[argchi2_min[0]]],[int(chi2_min-2),int(chi2_min+10)], ls='dashed', lw = 1.5, color = 'black')
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[B_chi[argchi2_min[1]],B_chi[argchi2_min[1]]], ls='dashed', lw = 1.5, color = 'black')

# setting axis labels 
ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$\omega_0\left(KHz\right)$', fontsize = 18) 
ax[0,0].set_xlabel(r'$\chi^2_{{min}}$', fontsize = 18) 
ax[1,1].set_xlabel(r'$A \, [\, \frac{mV}{\tau s}\,]$', fontsize = 18) 
ax[1,1].set_ylabel(r'$\chi^2_{{min}}$', fontsize = 18) 
ax[0,1].set_xlabel(r'$A \, [\, \frac{mV}{\tau s}\,]$', fontsize = 18)
ax[0,1].set_ylabel(r'$\omega_0\left(KHz\right)$', fontsize = 18) 
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10)) 
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

# plotting legend
ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.5, 0.95), loc='upper center', fontsize=16)

# Change the size of ticks for 2D plots
ax[0, 0].tick_params(axis='x', labelsize=11)
ax[0, 0].tick_params(axis='y', labelsize=11)

ax[0, 1].tick_params(axis='x', labelsize=11)
ax[0, 1].tick_params(axis='y', labelsize=11)

ax[1, 1].tick_params(axis='x', labelsize=11)
ax[1, 1].tick_params(axis='y', labelsize=11)

#plt.show()

plt.savefig(file+'_A_omega_map'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - PLOTTING CHI SQUARE 3D FUNCTION(NO CONTOURF)

# Define the plot
fig = plt.figure(figsize=(6.5, 6.5), tight_layout=True)

# Select plotting style as 3D
ax = plt.axes(projection='3d')

# Plotting B and C as X and Y axis of the plot - Z as the chi square values
X, Y = np.meshgrid(B_chi, C_chi)
Z = chi2D

# Plotting the chi square surface near the best value
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, cmap = 'plasma')

# Add a color bar and shrink it
cbar = fig.colorbar(surf, shrink=0.8)

# Setting colorbar' name
cbar.set_label(r'$\chi^2$',rotation=360, size = 13, labelpad = 20)

# delating automatic rotation
ax.zaxis.set_rotate_label(False)  

# Set axis labels
ax.set_xlabel(r'$\omega_0\left(KHz\right)$', fontsize=13, labelpad=15)
ax.set_ylabel(r'$\tau\left(\mu s\right)$', fontsize=13, labelpad=15)
ax.set_zlabel(r'$\chi^2$', fontsize=13, labelpad=10, rotation = 90)

plt.savefig(file+'_chi_square_3D'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

# endregion

# region - COVARIANCE ANALYSIS

# find autovalues and autovector of the covariance matrix
autoval, autovector = np.linalg.eig(pcov[1:, 1:]) 

# find the autovector connected with the major samiaxis 
if(np.abs(autoval[0]) < np.abs(autoval[1])):
    major_semi_axis = autovector[0]
    minor_semi_axis = autovector[1]
    major_autoval = autoval[0]
    minor_autoval = autoval[1]
else: 
    major_semi_axis = autovector[1]
    minor_semi_axis = autovector[0]
    major_autoval = autoval[1]
    minor_autoval = autoval[0]

# normalized with KHz and micros
major_semi_axis[0] = major_semi_axis[0]*mu
major_semi_axis[1] = major_semi_axis[1]/1000
minor_semi_axis[0] = minor_semi_axis[0]*mu
minor_semi_axis[1] = minor_semi_axis[1]/1000

# computing the m coefficient of the line // to the major semiaxis
m_linear_chi = major_semi_axis[1]/major_semi_axis[0]

# endregion

# region - PLOTTING CHI SQUARE 3D MAP WITH COVARIANCE LINEAR CORRELATION

cmap = mpl.colormaps['viridis'].reversed()
#print(chi2D.shape)
level = np.linspace(np.min(chi2D),np.max(chi2D),100)
line_c = 'gray'

# plot definition
fig, ax = plt.subplots(2, 2, figsize=(6.5, 6.5), constrained_layout=True, 
                       height_ratios=[3, 1], width_ratios=[1, 3], sharex='col', sharey='row')

# defining smooth plot of chi square values in function of two parameteres
im = ax[0,1].contourf(B_chi,C_chi,chi2D, levels=level, cmap=cmap) 

# plotting the legend's bar of chi square value(and associated color)
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 

# setting the bar label
cbar.set_label(r'$\chi^2$',rotation=360, size = 13)

# Plotting constant chi-square elliptical orbit at Xmin + 1, + 2.3, + 3.8
CS = ax[0, 1].contour(B_chi, C_chi, chi2D, levels=[chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
                      linewidths=2, colors=['red','darkorange', 'saddlebrown'], alpha=1, linestyles='-')

# Add labels directly to the legend
handles, _ = CS.legend_elements()
labels = [r'$\chi^2_{{min}} + 1$', r'$\chi^2_{{min}} + 2.3$', r'$\chi^2_{{min}} + 3.8$']

# lotting chi square value of elliptical orbits
ax[0,1].clabel(CS, inline=True, fontsize=11, fmt='%10.0f')

# plotting correlation function
ax[0, 1].plot(B_chi, linear(B_chi, m_linear_chi, C_chi[argchi2_min[2]], B_chi[argchi2_min[1]]), lw = 2, color = 'whitesmoke', ls = '--', alpha=1, label = "Major eigenvector inclination")

#plotting the parabolic curve for tau(also standard dev and best parameter)
ax[0,0].plot(prof_B,C_chi,ls='-') 
ax[0,0].set_xticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])
#ax[0,0].text(49,C_chi[np.argmin(prof_C)],str(np.around(C_chi[np.argmin(prof_C)],0)), color='k',alpha=0.5, fontsize=9)
#ax[0,0].text(52,C_chi[B_sx]+50.,str(np.around(C_chi[B_sx]-C_chi[np.argmin(prof_C)],0)), color='b',alpha=0.5, fontsize=9)
#ax[0,0].text(52,C_chi[B_dx]-120.,str(np.around(C_chi[B_dx]-C_chi[np.argmin(prof_C)],0)), color='r',alpha=0.5, fontsize=9)

#plotting the parabolic curve for omega(also standard dev and best parameter)
ax[1,1].plot(B_chi,prof_C) 

#ax[1,1].text(B_chi[np.argmin(prof_B)]-200,50,str(np.around(B_chi[np.argmin(prof_B)],0)), color='k',alpha=0.5, fontsize=9)
#ax[1,1].text(B_chi[C_sx]+50.,54.5,str(np.around(B_chi[C_sx]-B_chi[np.argmin(prof_B)],0)), color='b',alpha=0.5, fontsize=9)
#ax[1,1].text(B_chi[C_dx]-270.,54.5,str(np.around(B_chi[C_dx]-B_chi[np.argmin(prof_B)],0)), color='r',alpha=0.5, fontsize=9)
ax[1,1].set_yticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])

# setting axis labels and limits
ax[1,0].set_axis_off()
ax[0, 1].set_ylim(min(C_chi), max(C_chi))
ax[0,0].set_ylabel(r'$\tau\left(\mu s\right)$', fontsize = 13) 
ax[0,0].set_xlabel(r'$\chi^2_{{min}}$', fontsize = 13) 
ax[1,1].set_xlabel(r'$\omega_0\left(KHz\right)$', fontsize = 13) 
ax[1,1].set_ylabel(r'$\chi^2_{{min}}$', fontsize = 13) 
ax[0,1].set_xlabel(r'$\omega_0\left(KHz\right)$', fontsize = 13)
ax[0,1].set_ylabel(r'$\tau\left(\mu s\right)$', fontsize = 13) 
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10)) 
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

# adding some text
#ax[0, 1].text(B_chi[argchi2_min[1]]+0.025, C_chi[argchi2_min[2]]+0.040,r'$\rho_{{pearson}}$ = {g:.1f}'.format(g=pers_tau_omega), size=12)
"""#ax[0, 1].text(B_chi[argchi2_min[2]]+0.005, int(chi2_min+2), r'$\omega_0$ = {d:.2f} - {e:.2f} + {f:.2f} kHz'.format(d=B_chi[argchi2_min[1]], e=errB/1000, f=errBB/1000), size=13)
ax[0, 0].text(int(chi2_min+2), C_chi[argchi2_min[2]]+0.001,r'$\tau$ = {g:.3f} $\mu s$'.format(g=C_chi[argchi2_min[2]]), size=11)
ax[0, 0].text(int(chi2_min+2), C_chi[C_sx]+0.001,r' + {g:.3f} $\mu s$'.format(g=errC*mu), size=11)
ax[0, 0].text(int(chi2_min+2), C_chi[C_dx]+0.001,r'- {g:.3f} $\mu s$'.format(g=errCC*mu), size=11)"""

# plotting legends
ax[1, 0].legend(handles, labels, bbox_to_anchor=(0.5, 0.95), loc='upper center', fontsize=13)
ax[0, 1].legend(loc='lower left', fontsize=12, framealpha=0, ncol = 2)

plt.savefig(file+'_chi_square_cov'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

#plt.show()

# endregion

# region - COMPUTING L AND C FROM DATA

# computing the total resistance(data taken using metrix)
Rtot = 11.63 + 48.7839 + 12.00 # resistance of inductance, generator, R

# computing the error not considering the error on Rgenerator --> TO VERIFY
Rtot_err = np.sqrt(3*pow(0.01*8, 2)/3 + pow(Rtot*0.1/100, 2)/3)

L = C_chi[argchi2_min[2]]*Rtot/2 #microH
C = 2/(C_chi[argchi2_min[2]]*Rtot*pow(B_chi[argchi2_min[1]], 2)) #F

L_err = np.sqrt(pow(Rtot*errC*mu/2, 2) + pow(C_chi[argchi2_min[2]]*Rtot_err/2, 2)) #microH
C_err = (2/(Rtot*C_chi[argchi2_min[2]]*B_chi[argchi2_min[1]]**2))*np.sqrt(pow(errC*mu/C_chi[argchi2_min[2]], 2) + pow(Rtot_err/Rtot, 2) + pow(2*errB/(1000*B_chi[argchi2_min[1]]), 2)) #F

print("=======================================================================================")
print("R = ", Rtot, " +- ", Rtot_err) 
print("L = ", L, " +- ", L_err, " microH")
print("C = ", C*1000000000, " +- ", C_err*1000000000, " nF")
print("Compatibility C_fit and C_metrix = ", compatibility(C, C_metrix, C_err, C_metrix_err))
print("=======================================================================================")
# endregion
 