import re
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from cycler import cycler
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# ==========================================================
# Setting the plot style
# ==========================================================

plt.style.use(hep.style.ROOT)

params = {
    'legend.fontsize': '12',
    'legend.loc': 'upper right',
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.facecolor': 'w',
    'legend.edgecolor': 'w',
    'figure.figsize': (6, 4),
    'axes.labelsize': '14',
    'figure.titlesize': '14',
    'axes.titlesize': '12',
    'xtick.labelsize': '10',
    'ytick.labelsize': '10',
    'lines.linewidth': '2',
    'text.usetex': False,
    'axes.formatter.min_exponent': '2',
    'figure.subplot.left': '0.125',
    'figure.subplot.bottom': '0.125',
    'figure.subplot.right': '0.925',
    'figure.subplot.top': '0.925',
    'figure.subplot.wspace': '0.1',
    'figure.subplot.hspace': '0.1',
}

plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# ==========================================================
# Lettura istogramma
# ==========================================================

# Funzione per leggere un istogramma da un file di testo
def read_histogram(filename):
    """
    Legge un istogramma nel formato: colonna 1 energy(MeV), colonna 2 counts.
    """
    energy = []
    counts = []
    with open(filename, 'r') as f:
        for line in f:
            if re.match(r'^\s*#', line):
                continue  # Skip comment lines
            parts = line.split()
            if len(parts) >= 2:
                energy.append(float(parts[0]))
                counts.append(float(parts[1]))

    return np.array(energy), np.array(counts)

# Funzione per il rebinning dei grafici
def rebin_histogram(energy, counts, new_bin_count):
    """
    Rebin an histogram while conserving the total counts.
    """
    if len(energy) != len(counts):
        raise ValueError("Energy and counts arrays must have the same length.")
    if len(energy) == 0 or new_bin_count < 1:
        raise ValueError("The histogram must be non-empty and have at least one new bin.")

    # Infer the original bin edges from the bin centers.
    old_edges = np.empty(len(energy) + 1)
    old_edges[1:-1] = (energy[:-1] + energy[1:]) / 2
    old_edges[0] = energy[0] - (old_edges[1] - energy[0])
    old_edges[-1] = energy[-1] + (energy[-1] - old_edges[-2])

    # Calculate the new bin edges over the full original histogram range.
    new_bin_edges = np.linspace(old_edges[0], old_edges[-1], new_bin_count + 1)
    rebinned_counts = np.zeros(new_bin_count)

    # Split each original count according to its overlap with new bins.
    for old_left, old_right, count in zip(old_edges[:-1], old_edges[1:], counts):
        old_width = old_right - old_left
        first_new_bin = max(np.searchsorted(new_bin_edges, old_left, side="right") - 1, 0)
        last_new_bin = min(np.searchsorted(new_bin_edges, old_right, side="left"), new_bin_count - 1)
        for new_index in range(first_new_bin, last_new_bin + 1):
            overlap = max(
                0.0,
                min(old_right, new_bin_edges[new_index + 1])
                - max(old_left, new_bin_edges[new_index]),
            )
            rebinned_counts[new_index] += count * overlap / old_width

    # Return the centers of the new bins.
    rebinned_energy = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2

    return rebinned_energy, rebinned_counts

# NEUTRON SPECTRUM PLOTTING
# ==========================================================
# File da plottare
# ==========================================================
"""
files = [
    ("Emerging_Neutrons.txt", "Source primary - Counts 1374723", "mediumblue"),
    ("Secondary_Neutrons_Emerging.txt", "Source secondary - Counts 189691", "dodgerblue"),
    ("Primary_Neutrons_Reaching.txt", "Reaching sensor primary - Counts 86123", "darkorange"),
    ("Secondary_Neutrons_Reaching.txt", "Reaching sensor secondary - Counts 2264", "gold"),
    ("Primary_Neutrons_interacting.txt", "Interaction with sensor primary - Counts 273", "firebrick"),
    ("Secondary_Neutrons_interacting.txt", "Interaction with sensor secondary - Counts 15", "darkmagenta"),
]
"""
files = [
    ("Emerging_Neutrons.txt", "Source primary", "mediumblue"),
    ("Secondary_Neutrons_Emerging.txt", "Source secondary", "dodgerblue"),
    ("Primary_Neutrons_Reaching.txt", "Reaching sensor primary - Counts", "darkorange"),
    ("Secondary_Neutrons_Reaching.txt", "Reaching sensor secondary - Counts", "gold"),
    ("Primary_Neutrons_interacting.txt", "Interaction primary - Counts", "firebrick"),
    ("Secondary_Neutrons_interacting.txt", "Interaction secondary - Counts", "darkmagenta"),
]
# ==========================================================
# Somma del secondo e terzo spettro (opzionale)
# ==========================================================

first_e, first_c = read_histogram(files[1][0])
second_e, second_c = read_histogram(files[2][0])

step = min(np.diff(first_e).min(), np.diff(second_e).min())

emin = min(first_e.min(), second_e.min())
emax = max(first_e.max(), second_e.max())

common_e = np.arange(emin, emax + step, step)

d1 = first_c / np.mean(np.diff(first_e))
d2 = second_c / np.mean(np.diff(second_e))

d1_common = np.interp(common_e, first_e, d1, left=0, right=0)
d2_common = np.interp(common_e, second_e, d2, left=0, right=0)

common_counts = (d1_common + d2_common) * step


# ==========================================================
# Plot
# ==========================================================

fig, ax = plt.subplots(figsize=(9,6))

raw_spectra = []
for filename, label, color in files:
    energy, counts = read_histogram(filename)

    # Rebinning the histogram to a new number of bins (e.g., 100)
    new_bin_count = 110
    energy, counts = rebin_histogram(energy, counts, new_bin_count)

    raw_spectra.append((energy, counts, label, color))

source_total = raw_spectra[0][1].sum() + raw_spectra[1][1].sum()

percentual = [(counts.sum() * 100 / source_total) for energy, counts, label, color in raw_spectra]

# Printing the total count of each dataset
print("Neutron Spectra Total Counts:")
for energy, counts, label, color in raw_spectra:
    print(f"{label}: {counts.sum()}")

spectra = [
    (energy, counts / source_total * 100.0, label, color)
    for energy, counts, label, color in raw_spectra
]

# Asse Y ibrido: parte bassa log, parte alta lineare.
y_floor = 1e-7
transition_candidates = [np.max(s[1]) for s in spectra[2:5]]
y_transition = max(0.1, max(transition_candidates) * 1.2)

def y_forward(y):
    y = np.asarray(y, dtype=float)
    y_safe = np.maximum(y, y_floor)
    return np.where(
        y_safe <= y_transition,
        np.log10(y_safe),
        np.log10(y_transition) + (y_safe - y_transition) / y_transition,
    )

def y_inverse(yy):
    yy = np.asarray(yy, dtype=float)
    return np.where(
        yy <= np.log10(y_transition),
        10 ** yy,
        y_transition * (1.0 + yy - np.log10(y_transition)),
    )

ax.set_yscale("function", functions=(y_forward, y_inverse))

legend_handles = []

for i, (energy, counts, label, color) in enumerate(spectra):

    legend_label = f"{label}"
    if i > 1 and i < 4:
        handle, = ax.step(
            energy,
            counts,
            where="mid",
            linewidth=2,
            color=color,
            # Combining the label with the percentual value for the legend
            label = f"{legend_label} {percentual[i]:.2f} %"
        )

        legend_handles.append(handle)
    elif i <= 1:
        handle, = ax.step(
            energy,
            counts,
            where="mid",
            linewidth=2,
            color=color,
            label=legend_label,
        )

        legend_handles.append(handle)

    elif i >= 4:
        handle, = ax.step(
        energy,
        counts,
        where="mid",
        linewidth=2,
        color=color,
        # Combining the label with the percentual value for the legend
        label = f"{legend_label} {percentual[i]:.4f} %"
    )

        legend_handles.append(handle)

    else:
        # Solo voce in legenda
        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                linewidth=2,
                label=legend_label
            )
        )

# Tick espliciti per scala ibrida: decadi nella zona log, pochi tick nella zona lineare.
positive_values = np.concatenate([c[c > 0] for _, c, _, _ in spectra])
if positive_values.size > 0:
    y_min_pos = max(positive_values.min(), y_floor)
    y_max = positive_values.max()

    log_ticks = []
    if y_min_pos < y_transition:
        dmin = int(np.floor(np.log10(y_min_pos)))
        dmax = int(np.ceil(np.log10(y_transition)))
        log_ticks = [10 ** d for d in range(dmin, dmax + 1) if 10 ** d < y_transition]

    lin_ticks = []
    if y_max > y_transition:
        nice_locator = mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        lin_ticks = [tick for tick in nice_locator.tick_values(y_transition, y_max) if tick > y_transition]

    custom_ticks = sorted(set(log_ticks + lin_ticks))
    if custom_ticks:
        ax.yaxis.set_major_locator(mticker.FixedLocator(custom_ticks))
        minor_ticks = []

        # Tick intermedi nella parte logaritmica: 2, 3, 5 e 7 per decade.
        if y_min_pos < y_transition:
            dmin = int(np.floor(np.log10(y_min_pos)))
            dmax = int(np.ceil(np.log10(y_transition)))
            for decade in range(dmin, dmax + 1):
                base = 10 ** decade
                for factor in (2, 3, 5, 7):
                    tick = factor * base
                    if y_min_pos <= tick < y_transition:
                        minor_ticks.append(tick)

        # Tick intermedi nella parte lineare: passo uniforme nel tratto 0.1-2.
        if y_max > y_transition:
            dense_start = max(y_transition, 0.1)
            dense_stop = min(2.0, y_max)
            minor_ticks.extend(
                np.arange(dense_start + 0.1, dense_stop, 0.1)
            )

            if y_max > 2.0:
                minor_ticks.extend(
                    np.arange(2.2, y_max, 0.2)
                )

        minor_ticks = sorted(set(minor_ticks) - set(custom_ticks))
        ax.yaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1g"))
        ax.set_ylim(bottom=y_floor, top=y_max * 1.05)

# ==========================================================
# Grafica
# ==========================================================

ax.set_xlabel("Energy (MeV)")
ax.set_ylabel("Normalized counts (%)")

ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.tick_params(axis="y", which="minor", direction="in", length=4)
ax.tick_params(axis="y", which="major", direction="in", length=8)
ax.legend(handles=legend_handles)
ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_xlim(left=0, right=13.6)
ax.set_ylim(y_floor, 2.4)
plt.title("Neutron Spectrum from GEANT4 Simulation")
plt.tight_layout()


# PHOTON SPECTRUM PLOTTING
# ==========================================================
# File da plottare
# ==========================================================
"""
files = [
    ("Secondary_Gamma_Emerging.txt", "Emerging - Counts: 1129064", "firebrick"),
    ("Secondary_Gamma_Reaching.txt", "Reaching sensor secondary - Counts: 62518", "darkorange"),
    ("Secondary_Gamma_interacting.txt", "Interaction with sensor secondary - Counts: 231", "darkorchid"),
]
"""
files = [
    ("Secondary_Gamma_Emerging.txt", "Emerging", "firebrick"),
    ("Secondary_Gamma_Reaching.txt", "Reaching sensor - Counts", "darkorange"),
    ("Secondary_Gamma_interacting.txt", "Interaction with sensor - Counts", "darkorchid"),
]

# ==========================================================
# Plot
# ==========================================================

fig, ax = plt.subplots(figsize=(9,6))

legend_handles = []

raw_spectra = []
for filename, label, color in files:
    energy, counts = read_histogram(filename)

    raw_spectra.append((energy, counts, label, color))

percentual_photon = [(counts.sum() * 100 / source_total) for energy, counts, label, color in raw_spectra]

# Printing the total count of each dataset
print("Photon Spectra Total Counts:")
for energy, counts, label, color in raw_spectra:
    print(f"{label}: {counts.sum()}")

source_total = raw_spectra[0][1].sum() + raw_spectra[1][1].sum()
spectra = [
    (energy, counts / source_total * 100.0, label, color)
    for energy, counts, label, color in raw_spectra
]

for i, (energy, counts, label, color) in enumerate(spectra):

    legend_label = f"{label}"


    if i == 0:
            # Disegna solo i primi tre istogrammi
            handle, = ax.step(
                energy,
                counts,
                where="mid",
                linewidth=2,
                color=color,
                label=legend_label,
            )

            legend_handles.append(handle)
    elif i == 1:
        # Disegna solo i primi tre istogrammi
        handle, = ax.step(
            energy,
            counts,
            where="mid",
            linewidth=2,
            color=color,
            # Combining the label with the percentual value for the legend
            label = f"{legend_label} {percentual_photon[i]:.2f} %",
        )
        legend_handles.append(handle)
    elif i == 2:
        # Disegna solo i primi tre istogrammi
        handle, = ax.step(
            energy,
            counts,
            where="mid",
            linewidth=2,
            color=color,
            # Combining the label with the percentual value for the legend
            label = f"{legend_label} {percentual_photon[i]:.4f} %",
        )
        legend_handles.append(handle)
    else:
        # Solo voce in legenda
        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                linewidth=2,
                label=legend_label
            )
        )

# ==========================================================
# Grafica
# ==========================================================

ax.set_xlabel("Energy (MeV)")
ax.set_ylabel("Normalized counts (%)")
ax.set_yscale("log")
ax.tick_params(direction="in", top=True, right=True)
ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend(handles=legend_handles)
plt.title("Photon Spectrum from GEANT4 Simulation")
plt.tight_layout()
plt.show()
