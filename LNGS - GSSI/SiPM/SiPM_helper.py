# SiMP_helper.py - Helper functions for SiPM (Silicon photon multipliers) data analysis and visualization

# Import necessary libraries
import ast
import numpy as np
import statistics
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from cycler import cycler
import mplhep as hep
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.stats import landau
try:
	from . import langau
except ImportError:
	import langau

# Graph style
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

# ==============================================
#          SiPM CLASS DEFINITION
# ==============================================

# Class to represent a SiPM (Silicon photon multiplier) with attributes extracted from a code string
class SiPM:
    def __init__(self, CODE_str):
        self.board = CODE_str.split("_")[0]
        self.side = CODE_str.split("_")[1]
        self.layer = CODE_str.split("_")[2]
        self.channel = CODE_str.split("_")[3]
        self.SiPM_number = CODE_str.split("_")[4]

# Class to represent the points of a finger plot
class FingerPlotPoint:
    def __init__(self, peaks, peaks_error, bg_idx, bg_values, bg_error, voltage):
        self.peaks = peaks
        self.peaks_error = peaks_error
        self.bg_idx = bg_idx
        self.bg_values = bg_values
        self.bg_error = bg_error
        self.voltage = voltage

# ==============================================
#           DATA EXTRACTION FUNCTIONS
# ==============================================

# Function to extract a column from a CSV file and return row numbers and values as NumPy arrays
def extract_SiPM_data(file_path, column_number, one_based_column=True):
    """
    Extract one column from a CSV file and return two NumPy arrays:
    1) row numbers of extracted values (1-based)
    2) values found in the selected column

    Args:
        file_path (str): Path to the CSV file.
        column_number (int): Column index to extract.
            By default this is interpreted as 1-based.
        one_based_column (bool): If True, `column_number` starts from 1.
            If False, `column_number` starts from 0.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - row_numbers: array of row numbers (1-based)
            - column_values: array of extracted values (float)
    """
    if not isinstance(column_number, int):
        raise TypeError("column_number must be an integer")

    selected_col = column_number - 1 if one_based_column else column_number
    if selected_col < 0:
        raise ValueError("column_number must be >= 1 (or >= 0 if one_based_column=False)")

    df = pd.read_csv(file_path, header=None)

    if selected_col >= df.shape[1]:
        raise IndexError(
            f"Column {column_number} is out of range. CSV has {df.shape[1]} columns."
        )

    column_series = pd.to_numeric(df.iloc[:, selected_col], errors="coerce")
    valid_mask = ~column_series.isna()

    row_numbers = np.arange(1, len(column_series) + 1, dtype=int)[valid_mask.to_numpy()]
    column_values = column_series[valid_mask].to_numpy(dtype=float)

    return row_numbers, column_values

# Function to calculate the column number in the CSV file based on DAQ, gain type, CITIROC, and channel
def get_column_number(DAQ, gain_type, CITIROC, channel):

    """Calculate the column number in the CSV file based on DAQ, gain type, CITIROC, and channel.
    DAQ: 1 or 2
    gain_type: "LG" or "HG"
    CITIROC: "A", "B", "C", or "D"
    channel: 0 to 31
    """
    
    column_number = 0

    if DAQ == 1:
      column_number += 0
    elif DAQ == 2:
      column_number += 256
    
    if gain_type == "LG":
        column_number += 0
    elif gain_type == "HG":
        column_number += 128
    
    if CITIROC == "A":
        column_number += 0
    elif CITIROC == "B":
        column_number += 32
    elif CITIROC == "C":
        column_number += 64
    elif CITIROC == "D":
        column_number += 96

    return column_number + channel

# ==============================================
#          PEAKS HELPER FUNCTIONS
# ==============================================

# Function to rebin the values
def rebin(values, bin_size):
    """Rebin the values by summing over non-overlapping bins of a specified size."""
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")
    n_bins = len(values) // bin_size
    rebinned = np.array([np.sum(values[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
    return rebinned

# Function to rebin paired x and y arrays using non-overlapping bins of fixed size.
def rebin_xy(x, y, bin_size, x_mode="mean", y_mode="sum"):
    """
    Rebin paired x and y arrays using non-overlapping bins of fixed size.

    Args:
        x (array-like): x values.
        y (array-like): y values.
        bin_size (int): number of consecutive samples per bin.
        x_mode (str): how to collapse x inside each bin. Supported values:
            "mean" (default), "first", "last".
        y_mode (str): how to collapse y inside each bin. Supported values:
            "sum" (default), "mean", "first", "last".

    Returns:
        tuple[np.ndarray, np.ndarray]: rebinned x and y arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")

    n_bins = len(x) // bin_size
    if n_bins == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    x_rebinned = []
    y_rebinned = []

    for i in range(n_bins):
        x_bin = x[i * bin_size:(i + 1) * bin_size]
        y_bin = y[i * bin_size:(i + 1) * bin_size]

        if x_mode == "mean":
            x_value = np.mean(x_bin)
        elif x_mode == "first":
            x_value = x_bin[0]
        elif x_mode == "last":
            x_value = x_bin[-1]
        else:
            raise ValueError("x_mode must be 'mean', 'first', or 'last'")

        if y_mode == "sum":
            y_value = np.sum(y_bin)
        elif y_mode == "mean":
            y_value = np.mean(y_bin)
        elif y_mode == "first":
            y_value = y_bin[0]
        elif y_mode == "last":
            y_value = y_bin[-1]
        else:
            raise ValueError("y_mode must be 'sum', 'mean', 'first', or 'last'")

        x_rebinned.append(x_value)
        y_rebinned.append(y_value)

    return np.asarray(x_rebinned, dtype=float), np.asarray(y_rebinned, dtype=float)

# Function to find peaks in a 1D array using SciPy's `find_peaks` if available, or a fallback method
def find_peaks(y, use_scipy=True, prominence=None, distance=None):
    """
    Find peaks in a 1D array. Uses `scipy.signal.find_peaks` when available;
    otherwise falls back to a NumPy-based local-maxima + greedy distance filter.

    Args:
        y (array-like): 1D data array.
        use_scipy (bool): try SciPy's `find_peaks` if True.
        prominence (float|None): prominence parameter for SciPy or absolute
            amplitude threshold for the fallback.
        distance (int|None): minimal horizontal distance between peaks (samples).

    Returns:
        tuple[np.ndarray, np.ndarray]: (peak_indices, peak_values)
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("Input array `y` must be 1-dimensional")

    if use_scipy:
        try:
            from scipy.signal import find_peaks

            kwargs = {}
            if distance is not None:
                kwargs['distance'] = int(distance)
            if prominence is not None:
                kwargs['prominence'] = float(prominence)

            peaks, props = find_peaks(y, **kwargs)
            return peaks, y[peaks]
        except Exception:
            # SciPy not available or failed -> fallback
            pass

    # Fallback: local maxima (ignore first and last samples)
    if y.size < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    if cand.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Determine prominence threshold
    if prominence is None:
        prom_thr = (np.nanmax(y) - np.nanmin(y)) * 0.05
    else:
        prom_thr = float(prominence)

    cand = np.array([i for i in cand if y[i] - np.nanmin(y) >= prom_thr])
    if cand.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Minimal distance between peaks
    if distance is None:
        dist = max(1, int(len(y) / 50))
    else:
        dist = int(distance)

    # Greedy selection by amplitude
    selected = []
    candidates = list(cand)
    while candidates:
        idx_max = max(candidates, key=lambda i: y[i])
        selected.append(idx_max)
        candidates = [c for c in candidates if abs(c - idx_max) > dist]

    selected = np.array(sorted(selected), dtype=int)
    return selected, y[selected]

# Function to define a Gaussian function for curve fitting
def gauss_function(x, A, mu, sigma):
    """Gaussian function for curve fitting."""
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# Function to fit a Gaussian function to the data (x, y, sy) and return the optimal parameters
def gauss_fit(x, y, sy, p0=None):
    """Fit a Gaussian function to the data (x, y, sy) and return the optimal parameters."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = np.asarray(sy, dtype=float)

    if x.size < 3 or y.size < 3:
        raise ValueError("Need at least 3 points to fit a Gaussian")

    # Avoid zero or invalid uncertainties in the fit weights.
    sy = np.where(np.isfinite(sy) & (sy > 0), sy, 1.0)

    if p0 is None:
        A_guess = np.max(y)
        mu_guess = x[np.argmax(y)]
        sigma_guess = np.std(x)
        if not np.isfinite(sigma_guess) or sigma_guess <= 0:
            span = np.max(x) - np.min(x)
            sigma_guess = max(span / 6.0, 1.0)
        p0 = [A_guess, mu_guess, sigma_guess]

    bounds = ([0.0, np.min(x), 1e-12], [np.inf, np.max(x), np.inf])

    try:
        popt, pcov = curve_fit(
            gauss_function,
            x,
            y,
            p0=p0,
            sigma=sy,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=5000,
        )
    except RuntimeError:
        # Retry once with a broader sigma guess and a higher iteration budget.
        retry_p0 = [p0[0], p0[1], max(p0[2] * 2.0, 1.0)]
        popt, pcov = curve_fit(
            gauss_function,
            x,
            y,
            p0=retry_p0,
            sigma=sy,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=20000,
        )
    return popt, np.sqrt(np.diag(pcov))

# Integrating over a region
def integration_without_bg(x, y, bg_values, x_min, x_max):
    """Integrate the function defined by (x, y) over the region [x_min, x_max]."""
    mask = (x >= x_min) & (x <= x_max)
    return np.trapz(y[mask] - bg_values[mask], x[mask])

# ==============================================
#             FINGER PLOT ANALYSIS
# ==============================================

# Linear function for fitting distance between peaks vs voltage
def linear_func(x, a, b):
    return a * x + b

# Linear fit to distance between peaks vs voltage
def linear_fit(x, y, sy):
    """
    Fit a linear function to the data (x, y, sy) and return the optimal parameters.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = np.asarray(sy, dtype=float)

    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least 2 points to fit a line")

    # Avoid zero or invalid uncertainties in the fit weights.
    sy = np.where(np.isfinite(sy) & (sy > 0), sy, 1.0)

    p0 = [0.0, np.mean(y)]

    try:
        popt, pcov = curve_fit(
            linear_func,
            x,
            y,
            p0=p0,
            sigma=sy,
            absolute_sigma=True,
            maxfev=5000,
        )
    except RuntimeError:
        # Retry once with a higher iteration budget.
        popt, pcov = curve_fit(
            linear_func,
            x,
            y,
            p0=p0,
            sigma=sy,
            absolute_sigma=True,
            maxfev=20000,
        )
    return popt, np.sqrt(np.diag(pcov))

# ==============================================
#         BACKGROUND POINTS EXTRACTION
# ==============================================

# Function to extract the background points
def extract_bg(x, y, x_min, x_max, use_scipy=True, prominence=1, distance=None):
    """
    Extract the background points from the data (x, y) by selecting the points
    that are not peaks (i.e., those with indices not in `peaks_idx`).
    """

    # Transorming the lowest point of the fingerplot(bg points) to new peaks
    # Getting only x and y values with x between x_min and x_max
    mask = (x >= x_min) & (x <= x_max)
    x_mask = x[mask]
    y_mask = y[mask]
    
    # Searching for bg peaks
    peaks_idx, peaks_values = find_peaks(-y_mask, use_scipy=use_scipy, prominence = prominence, distance=distance)

    # Return normalized values
    peaks_values = -peaks_values
    peaks_idx = peaks_idx + np.where(mask)[0][0]

    return peaks_idx, peaks_values

# Model function for the background points: Landau distribution convolved with a Gaussian component
def background_model(xdata, landau_mean, landau_width, gauss_width, langau_amplitude):
    
    return langau_amplitude * langau.pdf(xdata, landau_x_mpv=landau_mean, landau_xi=landau_width, gauss_sigma=gauss_width)
    
# Fitting function for the background points
def bg_fitting(x, y, y_err, landau_mean, landau_width, gauss_width, langau_amplitude):
    """Fit the background points with a Landau plus Gaussian distribution."""

    popt, pcov = curve_fit(
        background_model,
        x,
        y,
        p0=(landau_mean, landau_width, gauss_width, langau_amplitude),
        sigma=y_err,
        maxfev=10000,
    )

    return popt, np.sqrt(np.diag(pcov))

# ==============================================
#          POISSON FITTING OF PEAKS
# =============================================

# Making a poisson fit with a discrate offset on x values
def poisson_fit(integrated_values):
    x_no_offset = np.arange(len(integrated_values))


    def poisson_model(k, lam, b):
        x = x_no_offset - b
        return lam**k * np.exp(-lam) / np.math.factorial(k)
    
    popt, pcov = curve_fit(
        poisson_model,
        x_no_offset,
        integrated_values,
        p0=(np.mean(integrated_values), 0),
        maxfev=10000,
    )
    return popt[0], popt[1], np.sqrt(np.diag(pcov))