# MAIN CLASS USED TO ANALYZE THE SRIM SIMULATIONS - Author: Marco Segato

# ========================== IMPORT DEPENDENCIES ==========================
# PYSRIM IMPORT
from srim.output import Collision
from srim.output import Results

# OTHER IMPORTS
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

# ========================== FUNCTIONS ==========================

# DATE EXTRACTION FUNCTIONS

# Function to extract data from RANGE_3D.txt file
def get_range_xyz(path, filename="RANGE_3D.txt"):
    """
    Extract data from RANGE_3D.txt file in 4 numpy arrays.
    \nInput: path (str) - path to the folder containing the RANGE_3D.txt file
    filename is RANGE_3D.txt by default, but can be changed if needed.
    \nReturn: ion_number(int), depth(float), y(float), z(float)
    """

    # Read the RANGE_3D.txt file
    file_path = os.path.join(path, filename)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Found the line where the data starts (after the line with 'Ion       Depth  X   Lateral Y   Lateral Z')
    start_idx = None
    for i, line in enumerate(lines):
        if 'Ion' in line and 'Depth' in line and 'Lateral' in line:
            start_idx = i + 2  # skip the header line and the line of dashes
            break
    if start_idx is None:
        raise ValueError("RANGE_3D.txt file format is not as expected. Could not find data header.")

    # Extract the data into numpy arrays
    ion_number = []

    # Range is in Angstroms, depth is the range along x
    depth = []
    y = []
    z = []

    for line in lines[start_idx:]:
        # Not all lines are data lines, some may be empty or contain non-numeric characters, so we need to check
        if len(line.strip()) == 0 or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue

        # Append the data to the respective lists
        ion_number.append(int(parts[0]))
        depth.append(float(parts[1]))
        y.append(float(parts[2]))
        z.append(float(parts[3]))

    return np.array(ion_number), np.array(depth), np.array(y), np.array(z)

# Function to extract data from IONIZATION.txt file
def get_ionization(path):
    """Extract ionization data from IONIZATION.txt file in 3 numpy arrays.
    \nInput: path (str) - path to the folder containing the IONIZATION.txt file
    \nReturn: depth(float), ions(float), recoils(float)"""
    results = Results(path)
    ioniz = results.ioniz
    return ioniz.depth, ioniz.ions, ioniz.recoils

# Class to extract data from RESULTS_TABLE
class Textract:

    def __init__(self, tabella_path, elemento):
        
        self.tabella_path = tabella_path
        self.elemento = elemento

        """
        Extract data for a single element from the RESULTS_TABLE.txt file\n.
        Input:
            tabella_path (str): path to the RESULTS_TABLE.txt file
            elemento (str): name of the element (e.g., "Boron", "Oxygen", "Silicon")\n
        Output:
            list of dictionaries, one for each data row of the element
        """

        self.dati = []
        with open(self.tabella_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Find the column headers (first occurrence of 'Ion angle')
        header_idx = None
        for i, line in enumerate(lines):
            if 'Ion angle' in line:
                header_idx = i
                break
        if header_idx is None:
            raise ValueError("Column headers not found.")
        headers = [h.strip() for h in lines[header_idx].split('\t') if h.strip()]

        # Find the section for the element of interest (first occurrence of 'ION TYPE: {elemento}')
        start_idx = None
        for i, line in enumerate(lines):
            if f"ION TYPE: {self.elemento}" in line:
                start_idx = i + 1
                break
        if start_idx is None:
            raise ValueError(f"Element {self.elemento} not found in the table.")

        # Find the end of the section (next empty line or with ---- or new ION TYPE)
        end_idx = None
        for i in range(start_idx+1, len(lines)):
            if lines[i].strip() == '' or 'ION TYPE:' in lines[i] or '---' in lines[i]:
                end_idx = i
                break
        if end_idx is None:
            end_idx = len(lines)

        # Extract data row by row (after the header, but only in the section of the element)
        for i in range(start_idx, end_idx):
            row = lines[i].strip()
            if not row or row.startswith('---') or 'Ion angle' in row or "ION TYPE:" in row:
                continue
            valori = [v.strip() for v in row.split('\t') if v.strip()]
            if len(valori) != len(headers):
                continue
            valori_float = []

            # Convert values to float, handling '***' as None
            for v in valori:
                if v == '***':
                    valori_float.append(None)
                else:
                    try:
                        valori_float.append(float(v))
                    except ValueError:
                        valori_float.append(None)

            self.dati.append(valori_float)
        
        # Now we have a list of lists (self.dati) where each inner list corresponds to a row of data.
        self.angle = [d[0] for d in self.dati]
        self.rangex = [d[1] for d in self.dati]
        self.rangey = [d[2] for d in self.dati]
        self.rangez = [d[3] for d in self.dati]
        self.rangex_dev = [d[4] for d in self.dati]
        self.rangey_dev = [d[5] for d in self.dati]
        self.rangez_dev = [d[6] for d in self.dati]
        self.LET = [d[7] for d in self.dati]
        self.LET_dev = [d[8] for d in self.dati]
        self.E_Si = [d[9] for d in self.dati]
        self.E_SiO2 = [d[10] for d in self.dati]
        self.E_tot = [d[11] for d in self.dati]
        
    # Define getter methods for each attribute
    def angle(self):
        """Getting angle"""
        return np.array(self.angle)
    
    # Define getter methods for each attribute
    def rangex(self):
        """Getting rangex"""
        return np.array(self.rangex)

    # Define getter methods for each attribute
    def rangey(self):
        """Getting rangey"""
        return np.array(self.rangey)
    
    # Define getter methods for each attribute
    def rangez(self):
        """Getting rangez"""
        return np.array(self.rangez)
    
    # Define getter methods for each attribute
    def rangex_dev(self):
        """Getting rangex_dev"""
        return np.array(self.rangex_dev)
    
    # Define getter methods for each attribute
    def rangey_dev(self):
        """Getting rangey_dev"""
        return np.array(self.rangey_dev)
    
    # Define getter methods for each attribute
    def rangez_dev(self):
        """Getting rangez_dev"""
        return np.array(self.rangez_dev)
    
    # Define getter methods for each attribute
    def LET(self):
        """Getting LET"""
        return np.array(self.LET)

    # Define getter methods for each attribute
    def LET_dev(self):
        """Getting LET_dev"""
        return np.array(self.LET_dev)
    
    # Define getter methods for each attribute
    def E_Si(self):
        """Getting E_Si"""
        return np.array(self.E_Si)
    
    # Define getter methods for each attribute
    def E_SiO2(self):
        """Getting E_SiO2"""
        return np.array(self.E_SiO2)

    # Define getter methods for each attribute
    def E_tot(self):
        """Getting E_tot"""
        return np.array(self.E_tot)
    
# NUMERICAL FUNCTIONS DEFINITION

# Gaussian distribution
def gauss(x, mu, sigma):
    """
    Gaussian distribution function.
    \nInput: x (array-like) - variable, mu (float) - mean, sigma (float) - standard deviation
    \nReturn: Gaussian distribution values for the input x
    """

    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))

# BRAGG PEAK FIT FUNCTION 1

def bragg_1(x, a, b, c, d, f):
    """
    Bragg peak fit function.
    \nInput: x (array-like), a, b, c, d (floats) - fit function parameters
    \nReturn: a * exp(-((b-x)^2)/(2*d^2)) * (b-x)^c
    """
    #return (a * x**2 + b * x + c) / x**2
    return a*np.exp(-(b-x)**2/(2*d**2))*(np.abs(b-x))**c + f

# FITTING FUNCTIONS

# Bragg peak fit function 1
def bragg_1_fit(x, y, p0=None):
    """
    Fit the Bragg peak using the bragg_1 function
    \nInput: x (array-like), y (array-like), p0 (array-like, optional) - initial parameters guess
    \nReturn: popt (fitted parameters), pcov (covariance matrix), x_fit (array), y_fit (array)
    \n FUNCTION: a * exp(-((b-x)^2)/(2*d^2)) * (b-x)^c
    """

    # Convert in to array
    x = np.array(x)
    y = np.array(y)

    # Remove x==0 to avoid division by zero
    mask = x != 0
    x_fitdata = x[mask]
    y_fitdata = y[mask]

    # Set a default initial guess if not provided
    if p0 is None:
        p0 = [1.0, 1.0, 1.0, 1.0, 0.0]  # [a, b, c, d, f]

    # Fit the data using curve_fit
    popt, pcov = curve_fit(bragg_1, x_fitdata, y_fitdata, p0=p0)

    # Evaluate fit for plotting (avoid x==0)
    x_fit = np.linspace(np.min(x_fitdata), np.max(x_fitdata), 1000)
    y_fit = bragg_1(x_fit, *popt)

    return popt, pcov, x_fit, y_fit

# Gaussian fit from mean and std
def gauss_simplified_fit(x, bin_length):

    """
    Fit a Gaussian distribution to the data using the mean and standard deviation of the data.
    \nInput: x (array-like) - data to fit, bin_length (float) - length of each bin
    \nReturn: x (array-like) - variable, Gaussian fit values for the input x, mean, standard deviation
    """

    mu = statistics.mean(x)
    sigma = statistics.stdev(x)
    x_fit = np.linspace(min(x), max(x), 1000)
    return x_fit, len(x)*bin_length*gauss(x_fit, mu, sigma), mu, sigma

# CONVERSION FUNCTIONS

def LET_conversion(LET, density):
    """
    Convert LET from eV/Angstrom to MeV*cm^2/mg.
    \nInput: LET (float) - LET in eV/Angstrom, density (float) - density of the material in g/cm^3
    \nReturn: LET in MeV*cm^2/mg
    """
    return LET*100/density
