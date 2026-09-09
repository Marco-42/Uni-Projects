import os
import re
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import mplhep as hep

# Setting the plot style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '12',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (6, 4),
         'axes.labelsize': '14',
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

INPUT_CANDIDATES = [
	"./cluster_multiplicity/cluster_multipliticy_first10bins_1.txt",  # common typo in file name
	"./cluster_multiplicity/cluster_multiplicity_first10bins_1.txt",
]
OUTPUT_PLOT = "./cluster_multiplicity/cluster_multiplicity_first10bins_2d.png"
OUTPUT_PLOT_ALL = "./cluster_multiplicity/cluster_multiplicity_first10bins_all_vcasn.png"

# Function to resolve the input path from candidates
def resolve_input_path(candidates):
	for path in candidates:
		if os.path.exists(path):
			return path
	raise FileNotFoundError(
		"Nessun file trovato tra i candidati: " + ", ".join(candidates)
	)

# Function to parse the multiplicity text file
def parse_multiplicity_txt(file_path):
	"""
	Parse blocks like:
	VCASN: 10
	multiplicity count
	1   123
	...
	10  456
	"""
	data_by_vcasn = {}
	current_vcasn = None

	with open(file_path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line:
				continue

			match_vcasn = re.match(r"^VCASN:\s*(\d+)\s*$", line)
			if match_vcasn:
				current_vcasn = int(match_vcasn.group(1))
				data_by_vcasn[current_vcasn] = {}
				continue

			if current_vcasn is None:
				continue

			if line.lower().startswith("multiplicity"):
				continue

			parts = line.split()
			if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
				multiplicity = int(parts[0])
				count = int(parts[1])
				if 1 <= multiplicity <= 10:
					data_by_vcasn[current_vcasn][multiplicity] = count

	if not data_by_vcasn:
		raise ValueError("Nessun blocco VCASN valido trovato nel file di input.")

	return data_by_vcasn

# Function to compute mean counts excluding a specific VCASN
def mean_counts_excluding_vcasn(data_by_vcasn, excluded_vcasn=3):
	multiplicities = list(range(1, 11))
	selected_vcasn = [v for v in sorted(data_by_vcasn.keys()) if v != excluded_vcasn]

	if not selected_vcasn:
		raise ValueError(f"Nessun VCASN disponibile diverso da {excluded_vcasn}.")

	counts_matrix = np.array(
		[[data_by_vcasn[v].get(m, 0) for m in multiplicities] for v in selected_vcasn],
		dtype=float,
	)

	mean_counts = counts_matrix.mean(axis=0)
	return multiplicities, mean_counts, selected_vcasn

# Function to plot mean counts excluding a specific VCASN
def plot_mean_counts(data_by_vcasn, output_path):
	multiplicities, mean_counts, used_vcasn = mean_counts_excluding_vcasn(data_by_vcasn, excluded_vcasn=3)

	plt.figure(figsize=(10, 6))
	# Errore sulla media stimato con incertezza di Poisson dei conteggi medi.
	mean_err = np.sqrt(mean_counts)
	plt.errorbar(
		multiplicities,
		mean_counts,
		yerr=mean_err,
		marker="o",
		linewidth=2,
		markersize=6,
		color="tab:blue",
		label=f"Media VCASN != 3 (N={len(used_vcasn)})",
	)

	plt.xlabel("Molteplicita cluster")
	plt.ylabel("Conteggi medi")
	plt.title("Media conteggi cluster (bin 1-10) su VCASN != 3")
	plt.xticks(multiplicities)
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.show()

# Function to plot counts for all VCASN
def plot_all_vcasn_counts(data_by_vcasn, output_path):
	multiplicities = list(range(1, 11))
	sorted_vcasn = sorted(data_by_vcasn.keys())

	plt.figure(figsize=(10, 6))
	cmap = plt.get_cmap("tab10" if len(sorted_vcasn) <= 10 else "tab20")

	for idx, vcasn in enumerate(sorted_vcasn):
		counts = [data_by_vcasn[vcasn].get(m, 0) for m in multiplicities]
		plt.errorbar(
			multiplicities,
			counts,
			yerr=np.sqrt(counts),
			marker="o",
			linewidth=1.8,
			markersize=4.5,
			color=cmap(idx),
			label=f"VCASN {vcasn}",
		)

	plt.xlabel("Molteplicita cluster")
	plt.ylabel("Conteggi")
	plt.title("Conteggi cluster (bin 1-10) per tutti i VCASN")
	plt.xticks(multiplicities)
	plt.grid(alpha=0.3)
	plt.legend(title="Soglia", ncol=2)
	plt.tight_layout()

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.show()

# Function to take compton probabilitis plot from a two collum text file
def plot_xy_from_txt(file_path):
	data = np.loadtxt(file_path)
	x = data[:, 0]
	y = data[:, 1]

	return x, y

# Function to fit exponential decay to the data
def fit_exponential_decay(x, y):

	# Fit the data to an exponential decay function
	def exp_func(x, a, b, c):
		return a * np.exp(-b * x + c)

	from scipy.optimize import curve_fit

	popt, pcov = curve_fit(exp_func, x, y, p0=(1, 0.1, 0))
	return popt, pcov

if __name__ == "__main__":

	# Take a look at the cluster multiplicity data
	input_path = resolve_input_path(INPUT_CANDIDATES)
	data = parse_multiplicity_txt(input_path)

	# Plot the mean counts excluding VCASN 3 and all VCASN counts
	# plot_mean_counts(data, OUTPUT_PLOT)
	# plot_all_vcasn_counts(data, OUTPUT_PLOT_ALL)

	# Take a look at the compton probabilities
	x, y = plot_xy_from_txt("compton_prob.txt")

	# Fit the data to an exponential decay function
	popt, pcov = fit_exponential_decay(x, y)

	# Plotting the compton probabilities and the fitted curve
	plt.figure(figsize=(10, 6))
	plt.scatter(x, y, label=r"Density * Ratio $\leq$ 0.6 - Multiplicity $\leq$ 3", color="tab:blue", linestyle='None', marker = 'o')
	plt.plot(np.linspace(x.min(), x.max(), 100), popt[0] * np.exp(-popt[1] * np.linspace(x.min(), x.max(), 100) + popt[2]), label="Exponential Fit", color="darkorange", lw = 1.5)
	plt.xlabel("THR")
	plt.ylabel("Count (%)")
	plt.title("High multiplicity Compton events")
	plt.legend()
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig("./plots/compton_probabilities.png", dpi=300)

	print(f"Fitted parameters: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}")

	plt.show()
