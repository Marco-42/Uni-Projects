import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool


# settaggio globale grafici    
#print(plt.style.available)
#plt.style.use('classic')
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
         'figure.constrained_layout.use' : True
          }
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# Enable debug mode
DEB = False


# Function definition


def fitf_C(x, A, B, C):
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
    fitval = A / np.sqrt((1-omega**2/B**2)**2+1/C**2*omega**2/B**2)
    return fitval

def fitf_C_phase(x, A, B, C):
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
    fitval = np.arctan((1/(B/omega - omega/B))/C)
    return fitval

def fitf_R(x, A, B, C):
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
    fitval = A / np.sqrt(1+C**2*(omega**2/B**2-B**2/omega**2)**2)
    return fitval

def fitchi2(i,j,k):
    x = fr
    y = TR
    y_err = eTR
    AA,BB,CC = A_chi[i],B_chi[j],C_chi[k]
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
#    residuals = (y -  fitf_R(x,AA,BB,CC))  # Seleziono fit su R
    residuals = (y -  fitf_C(x,AA,BB,CC))  # Seleziono fit su C
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



## PARAMATERS

# Number of bins for parameter scan
NI = 20
NJ = 20
NK = 20

# Input file name
file = 'dataC'   # seleziono per fit su C
# file = 'RLC_Rres'   # seleziono per fit su R
inputname = file+'.txt'

# Frequency limits for the fit function (in kHz)
frfit0 = 10.0
frfit1 = 100.0

R = 12.07 #Ohm
C = 1.05e-9 #Farad
L = 445.4771548344286e-6 #Henry

# Initial parameter values
Cinit = np.sqrt(L/C)/R
Binit = 1/np.sqrt(L*C)
Ainit = 1

# Assumed reading errors
letturaV = 0.1*0.41
errscalaV = 0.03*0.41


#### LOAD DATA
    

# Read data from the input file
data = np.loadtxt(inputname).T
fr = data[0] #frequenze
Vin = data[1] #Vin
Vo = data[2] #Vout
phase = 2*np.pi*data[0]*data[3] #fase


# Number of points to fit
# va a contare il numero di frequenze nel vettore fr che siano maggiori di zero
N = len(fr[fr > 0])

# Calculate errors on x and y
eVo = np.sqrt((letturaV * 0.2)**2 + (errscalaV * Vo)**2)
eVin = np.sqrt((letturaV * 0.2)**2 + (errscalaV * Vin)**2)

# Calculate the transfer function
TR = Vo / Vin
eTR = TR * np.sqrt((eVo / Vo)**2 + (eVin / Vin)**2 + 2*(errscalaV**2))

# Plot Vin and Vout vs. f e the transfer function vs. f

fig, ax = plt.subplots(1, 2, figsize=(5, 4),sharex=True, constrained_layout = True, width_ratios=[1, 1])
ax[0].errorbar(fr,Vin,yerr=eVin, fmt='o', label=r'$V_{in}$',ms=2)
ax[0].errorbar(fr,Vo,yerr=eVo, fmt='o', label=r'$V_{out}$',ms=2)
ax[0].legend(prop={'size': 10}, loc='best')
ax[0].set_ylabel(r'Voltaggio (V)')

ax[1].errorbar(fr,TR,yerr=eTR, fmt='o', label=r'$T=\frac{V_{out}}{V{in}}$',ms=2,color='red')
ax[1].legend(prop={'size': 10}, loc='best')
ax[1].set_ylabel(r'Funzione di trasferimento $T_C$')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].yaxis.set_ticks_position('right')
ax[1].yaxis.set_label_position('right')

plt.savefig(file+'_1'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()


# Perform the fit

popt, pcov = curve_fit(fitf_C, fr, TR, p0=[Ainit, Binit, Cinit], method='lm', sigma=eTR, absolute_sigma=True)

"""
POPT: Vettore con la stima dei parametri dal fit
PCOV: Matrice delle covarianze
bounds sono i limiti inferiori e superiori dei parametri (si richiede positività delle stime in questo caso)
"""

perr = np.sqrt(np.diag(pcov))
print( ' ampiezza = {a:.3f} +/- {b:.3f} \n omega0 = {c:.1f} +/- {d:.1f} kHz \n Q-valore = {e:.1f} +/- {f:.1f}'.format(a=popt[0], b=perr[0],c=popt[1]/1000,d=perr[1]/1000,e=popt[2],f=perr[2]))

residuA = TR - fitf_C(fr, *popt)
chisq = np.sum((residuA/eTR)**2)
df = N - 3
chisq_rid = chisq/df

x_fit = np.linspace(min(fr), max(fr), 1000)

"""
fit tracciato con mille punti fra la freq min e max
"""

# Plot the fit
fig, ax = plt.subplots(2, 1, figsize=(5, 4),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fitf_C(x_fit, *popt), label='Fit', linestyle='--', color='black')
ax[0].plot(x_fit,fitf_C(x_fit,Ainit,Binit,Cinit), label='init guess', linestyle='dashed', color='green')
ax[0].errorbar(fr,TR,yerr=eTR, fmt='o', label=r'$T=\frac{V_{out}}{V{in}}$',ms=2,color='red')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Funzione di trasferimento $T_C$')
#ax[0].set_xticks([20,30,40,50])

ax[1].errorbar(fr,residuA,yerr=eTR, fmt='o', label=r'Residui$',ms=2,color='red')
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].plot(fr,np.zeros(len(fr)),color='black')

plt.savefig(file+'_2'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()


# Extract and print best fit (BF) parameters and errors
A_BF, B_BF, C_BF = popt #parametri del best fit
eA_BF, eB_BF, eC_BF = np.sqrt(np.diag(pcov)) # errori del BF

print("============== BEST FIT with SciPy ====================")
print(r'A = ({a:.3e} +/- {b:.1e})'.format(a=A_BF,b=eA_BF))
print(r'B = ({c:.5e} +/- {d:.1e}) kHz'.format(c=B_BF * 1e-3, d=eB_BF * 1e-3))
print(r'C = ({e:.3e} +/- {f:.1e})'.format(e=C_BF, f=eC_BF))
print(r'chisq = {m:.2f}'.format(m=chisq))
print("=======================================================")

"""
Ora che abbiamo effettuato la regressione ai minimi quadrati utilizzando una libreria di Python (Scipy), 
proviamo ad ottenere lo stesso risultato calcolando a mano dalla curva del chi2 i parametri piu' probabili
ed il loro errore.
"""

"""
Per prima cosa definiremo una matrice di punti A,B e C su cui calcolare il chi2. 
Per farlo la centriamo sul valore BF che abbiamo trovato con scipy e ci allarghiamo di un 2sigma
su ogni parametro (sempre considerando l'errore di scipy)
"""

# Define the interval for parameter limits
NSI = 2 # numero di sigma rispetto all'errore di scipy
A0, A1 = A_BF - NSI * eA_BF, A_BF + NSI * eA_BF # estremi di scansione del parametro A
B0, B1 = B_BF - NSI * eB_BF, B_BF + NSI * eB_BF # estremi di scansione del parametro B
C0, C1 = C_BF - NSI * eC_BF, C_BF + NSI * eC_BF # estremi di scansione del parametro C

"""
print(f"(A0 = {A0}, A1 = {A1}) V")
print(f"(B0 = {B0}, B1 = {B1}) Hz")
print(f"(C0 = {C0}, C1 = {C1}) s")
"""


# Calcolo mappa Chi2 3D

step = 30 # discretizzazione all'interno dell'intervallo di scansione

# array dei diversi parametri
A_chi = np.linspace(A0,A1,step)
B_chi = np.linspace(B0,B1,step)
C_chi = np.linspace(C0,C1,step)

# inizializzo la matrice 3D del chi2
mappa = np.zeros((step,step,step))
# creo una lista degli indici da mappare con il pool su piu processori
item = [(i,j,k) for i in range(step) for j in range(step) for k in range(step)]

# assign the global pool
# inizializzo il pool per mandare su piu processori (fino a 100 processi in parallelo)
pool = multiprocessing.pool.ThreadPool(100)
# issue top level tasks to pool and wait
# mappo la lista 'item' di indici nella funzione fitchi2 che si rifa' agli array dei parametri definiti
#e in uscita salva il valore del chi2 nella posizione corretta della matrice 3D. 
pool.starmap(fitchi2, item, chunksize=10)
# close the pool
pool.close()

# alla fine si ottiene la mappa 3D del chi2
mappa = np.asarray(mappa)            

# trovo il minimo del chi2 e la sua posizione nella matrice 3D
chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa),mappa.shape)

# calcolo i residui della regressione utilizzando i valori dei parametri del minimo del chi2
# ricontrollo che il minimo del chi2 sia coerente.
residui_chi2 = TR - fitf_C(fr,A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]])


chisq_res = np.sum((residui_chi2/eTR)**2)

print(chi2_min,argchi2_min, chisq_res)

# Grafico nuovamente la regressione e i residui, questa volta ottenuti calcolando a mano il minimo del chi2
fig, ax = plt.subplots(2, 1, figsize=(3, 5),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fitf_C(x_fit, A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]]), label='Fit', linestyle='--', color='blue')
#ax[0].plot(x_fit,fitf2(x_fit,Ainit,Binit,Cinit), label='init guess', linestyle='dashed', color='green')
ax[0].errorbar(fr,TR,yerr=eTR, fmt='o', label=r'$V_{out}$',ms=2,color='red')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Funzione di trasferimento $T_C$')
#ax[0].set_xticks([20,30,40,50])

ax[1].errorbar(fr,residuA,yerr=eTR, fmt='o', label=r'Residui$',ms=2,color='red')
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].plot(fr,np.zeros(N))

plt.savefig(file+'_3'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()

"""
A questo punto devo calcolare l'errore sui singoli parametri con il chi2+1, 
e profilare il chi2 sui diversi parametri. Utilizzo le funzioni sopra definite.
"""

#Calcolo Profilazione 2D
chi2D = profi2D(1,mappa)
#Achi2D = profi2D(2,mappa) 

#Calcolo Profilazioni 1D
prof_B = profi1D([1,3],mappa)
prof_C = profi1D([1,2],mappa)   
prof_A = profi1D([2,3],mappa)    

"""
Per trovare l'errore sui parametri devo trovare i valori a chi2+1
Per farlo sottraiamo alle profilazioni chi2+1 e ne facciamo il valore assoluto
In questo modo i dati saranno tutti positivi ed avranno un minimo a chi2+1
"""

lvl = chi2_min+1. # 2.3 (2parametri) # 3.8 (3parametri)
diff_B = abs(prof_B-lvl)
diff_C = abs(prof_C-lvl)
diff_A = abs(prof_A-lvl)

B_dx = np.argmin(diff_B[B_chi<B_BF]) # minimo di B per valori inferiori al BF
B_sx = np.argmin(diff_B[B_chi>B_BF])+len(diff_B[B_chi<B_BF]) # minimo di B per valori superiori al BF
C_dx = np.argmin(diff_C[C_chi<C_BF])
C_sx = np.argmin(diff_C[C_chi>C_BF])+len(diff_C[C_chi<C_BF])
A_dx = np.argmin(diff_A[A_chi<A_BF])
A_sx = np.argmin(diff_A[A_chi>A_BF])+len(diff_A[A_chi<A_BF])
#print(B_dx,B_sx,C_dx,C_sx,A_dx,A_sx)

# Facendo la differenza rispetto al BF ottengo gli errori a dx e a sx del BF
errA = A_chi[argchi2_min[0]]-A_chi[A_dx]
errAA = A_chi[A_sx]-A_chi[argchi2_min[0]]
errB = B_chi[argchi2_min[1]]-B_chi[B_dx] 
errBB = B_chi[B_sx] -B_chi[argchi2_min[1]]
errC = C_chi[argchi2_min[2]]-C_chi[C_dx]
errCC = C_chi[C_sx]-C_chi[argchi2_min[2]]


print("============== BEST FIT with chi2 ====================")
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]] * 1e-3, e=errB * 1e-3, f=errBB* 1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) '.format(g=C_chi[argchi2_min[2]], h=errC,  n=errCC))
print(r'chisq = {m:.2f}'.format(m=np.min(mappa)))
print("=======================================================")


"""
Adesso faccio il grafico delle profilazioni 2D e 1D dei diversi parametri
"""

# definisco la mappa colore
cmap = mpl.colormaps['plasma'].reversed()
level = np.linspace(np.min(chi2D),np.max(chi2D),100)
line_c = 'gray'


# Profilazione di Omega e Q
fig, ax = plt.subplots(2, 2, figsize=(5.5, 5),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
fig.suptitle(r'$\chi^2 \left(\omega_0, Q \right)$')
im = ax[0,1].contourf(B_chi,C_chi,chi2D, levels=level, cmap=cmap) 
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 
cbar.set_label(r'$\chi^2$',rotation=360)

CS = ax[0,1].contour(B_chi,C_chi,chi2D, levels=[chi2_min+0.0001,chi2_min+1,chi2_min+2.3,chi2_min+3.8],linewidths=1, colors='k',alpha=0.5,linestyles='dotted')
ax[0,1].clabel(CS, inline=True, fontsize=9, fmt='%.1f')
ax[0,1].text(B_chi[np.argmin(prof_B)],C_chi[np.argmin(prof_C)],r'{g:.0f}'.format(g=chi2_min), color='k',alpha=0.5, fontsize=9)
ax[0,1].plot([B0,B1],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed')
ax[0,1].plot([B0,B1],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed')
ax[0,1].plot([B_chi[B_sx],B_chi[B_sx]],[C0,C1],color=line_c, ls='dashed')
ax[0,1].plot([B_chi[B_dx],B_chi[B_dx]],[C0,C1],color=line_c, ls='dashed')

ax[0,0].plot(prof_B,C_chi,ls='-') 
ax[0,0].plot([int(chi2_min-1),int(chi2_min+4)],[C_chi[C_sx],C_chi[C_sx]], color=line_c, ls='dashed') 
ax[0,0].plot([int(chi2_min-1),int(chi2_min+4)],[C_chi[C_dx],C_chi[C_dx]], color=line_c, ls='dashed')

ax[0,0].set_xticks([int(chi2_min),int(chi2_min+1),int(chi2_min+4),int(chi2_min+6)])
ax[0,0].text(int(chi2_min+1),C_chi[np.argmin(prof_C)],r'{g:.2f}'.format(g=C_chi[np.argmin(prof_C)]), color='k',alpha=0.5, fontsize=9)
ax[0,0].text(int(chi2_min+2),C_chi[C_sx],r'{g:.2f}'.format(g=errCC), color='b',alpha=0.5, fontsize=9)
ax[0,0].text(int(chi2_min+2),C_chi[C_dx],r'{g:.2f}'.format(g=-1*errC), color='r',alpha=0.5, fontsize=9)

ax[1,1].plot(B_chi,prof_C) 
ax[1,1].plot([B_chi[B_sx],B_chi[B_sx]],[int(chi2_min-1),int(chi2_min+4)], color=line_c, ls='dashed') 
ax[1,1].plot([B_chi[B_dx],B_chi[B_dx]],[int(chi2_min-1),int(chi2_min+4)], color=line_c, ls='dashed')

ax[1,1].text(B_chi[np.argmin(prof_B)],int(chi2_min+1),r'{g:.3e}'.format(g=B_chi[np.argmin(prof_B)]), color='k',alpha=0.5, fontsize=9)
ax[1,1].text(B_chi[B_sx],int(chi2_min+2),r'{g:.0e}'.format(g=errBB), color='b',alpha=0.5, fontsize=9)
ax[1,1].text(B_chi[B_dx],int(chi2_min+2),r'{g:.0e}'.format(g=-1*errB), color='r',alpha=0.5, fontsize=9)
ax[1,1].set_yticks([int(chi2_min),int(chi2_min+4),int(chi2_min+6)])


ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$Q-valore$') 
ax[1,1].set_xlabel(r'$\omega_0\left(Hz\right)$',loc='center') 
ax[0,0].set_xlim(int(chi2_min-1),int(chi2_min+4)) 
ax[1,1].set_ylim(int(chi2_min-1),int(chi2_min+4))

plt.savefig(file+'_4'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()



