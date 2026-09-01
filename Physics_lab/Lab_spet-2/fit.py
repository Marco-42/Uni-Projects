import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler

# Setting plot style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '10',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',
          'legend.facecolor':     'w',
          'legend.edgecolor':     'w',
          'figure.figsize': (6, 4),
         'axes.labelsize': '10',
         'figure.titlesize' : '14',
         'axes.titlesize':'12',
         'xtick.labelsize':'10',
         'ytick.labelsize':'10',
         'lines.linewidth': '1',
         'text.usetex': False,
         'axes.formatter.min_exponent': '2',
         'figure.subplot.left':'0.125',
         'figure.subplot.bottom':'0.125',
         'figure.subplot.right':'0.925',
         'figure.subplot.top':'0.925',
         'figure.subplot.wspace':'0.1',
         'figure.subplot.hspace':'0.1',
          }
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# Definisci la funzione di fit: a / (x + c)^2 + b
def fit_function(x, a, c, b):
    return a / (x + c) ** 2 + b

# Leggi i dati dal file
data = np.loadtxt('Distanza_data.txt')
x_data = data[:, 0]
y_data = data[:, 1]   # assumiamo siano i rate (non log)
time_data = data[:, 2]

# Calcola l'incertezza assoluta sui rate (da conteggi statistici)
# Se y_data è già rate R, allora N = R * t, sigma_R = sqrt(N) / t = sqrt(R * t) / t
sigma_abs = np.sqrt(y_data * time_data) / time_data

# Stima iniziale e vincoli per evitare singolarità (x + c = 0)
c_lower = -np.min(x_data) + 1e-6
p0 = [ (np.max(y_data)-np.min(y_data)) * np.max(x_data) ** 2, 1.0, np.min(y_data) ]  # guess ragionevole
bounds = ([-np.inf, c_lower, -np.inf], [np.inf, np.inf, np.inf])

# Esegui il fit con pesi (sigma assolute)
popt, pcov = curve_fit(fit_function, x_data, y_data, sigma=sigma_abs, absolute_sigma=True, p0=p0, bounds=bounds)

a_fit, c_fit, b_fit = popt
a_err, c_err, b_err = np.sqrt(np.diag(pcov))

# Residui e chi^2 ridotto
residuals = y_data - fit_function(x_data, *popt)
chi2 = np.sum((residuals / sigma_abs) ** 2)
ndof = len(x_data) - len(popt)
chi2_red = chi2 / ndof if ndof > 0 else np.nan

# Stampa i risultati
print("=== Risultati del Fit ===")
print(f"a = {a_fit:.6g} ± {a_err:.6g}")
print(f"c = {c_fit:.6g} ± {c_err:.6g}")
print(f"b = {b_fit:.6g} ± {b_err:.6g}")
print(f"Chi^2 = {chi2:.4f}, dof = {ndof}, Chi^2 ridotto = {chi2_red:.4f}")

# Stampa tutti i dati con valori fittati e residui
print("\n=== Dati con Valori Fittati ===")
print(f"{'x (um)':<12} {'y_obs':<12} {'y_fit':<12} {'residui':<12} {'sigma':<12}")
print("-" * 60)
for i in range(len(x_data)):
    y_fit_val = fit_function(x_data[i], *popt)
    print(f"{x_data[i]:<12.6f} {y_data[i]:<12.6f} {y_fit_val:<12.6f} {residuals[i]:<12.6f} {sigma_abs[i]:<12.6f}")

print("\n=== Parametri Fittati ===")
print(f"Formula: a/(x+c)² + b")
print(f"a = {a_fit:.6g} ± {a_err:.6g}")
print(f"c = {c_fit:.6g} ± {c_err:.6g}")
print(f"b = {b_fit:.6g} ± {b_err:.6g}")

# Plot dei dati e fit
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(x_data, y_data, yerr=sigma_abs, fmt='o', label='Data', markersize=8, capsize=5, capthick=2, color='darkblue')

x_fit = np.linspace(x_data.min() - 0.5, x_data.max() + 0.5, 500)
y_fit = fit_function(x_fit, *popt)
ax.plot(x_fit, y_fit, '-', linewidth=2, color='darkorange', label='Fit: a/(x+c)² + b')

# Testo con i parametri sul grafico
param_text = (
    f"a = {a_fit:.4g} ± {a_err:.4g}\n"
    f"c = {c_fit:.4g} ± {c_err:.4g}\n"
    f"b = {b_fit:.4g} ± {b_err:.4g}\n"
    f"χ²/ndof = {chi2_red:.3f}"
)
ax.text(0.98, 0.02, param_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Thickness (um)', fontsize=12)
ax.set_ylabel('Rate', fontsize=12)
ax.set_title('Fit a/(x + c)² + b', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Cu_fit_thickness_a_over_xc_plus_b.png', dpi=300)

# Plot residui
plt.figure(figsize=(9, 5))
plt.errorbar(x_data, residuals, yerr=sigma_abs, fmt='o', color='purple', markersize=8, capsize=5, capthick=2, label='Residuals')
residuals_std = np.std(residuals)
plt.axhline(np.mean(residuals), color='gray', linestyle='--', label=f'Mean Residual: {np.mean(residuals):.2f} ± {residuals_std:.2f}')
plt.fill_between(np.linspace(x_data.min() - 0.5, x_data.max() + 0.5, 300), np.mean(residuals) - residuals_std, np.mean(residuals) + residuals_std, color='gray', alpha=0.2, label='±1σ Residuals')
plt.legend(fontsize=10)
plt.xlabel('Thickness (um)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals of the Fit', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Cu_residuals_fit_thickness_a_over_xc_plus_b.png', dpi=300)

plt.show()

# Ritorna i parametri come output utile (opzionale se integrato in altra funzione)
fit_results = {'a': (a_fit, a_err), 'c': (c_fit, c_err), 'b': (b_fit, b_err), 'chi2_red': chi2_red}


