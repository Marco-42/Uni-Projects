# ARCADIA_HELPER.py - Helper functions for ARCADIA SENSOR DATA ANALYSIS

# Import necessary libraries
import ast
import numpy as np
import statistics
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from cycler import cycler
import mplhep as hep

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

# ==================== VARIABLE DEFINITIONS =================

ARCADIA_PIXEL_SIZE = 25 # um
ARCADIA_SIZE = (512, 480) # pixels
ARCADIA_THICKNESS = 200 # um

# ==================== CLASS DEFINITIONS ====================

# Class to represent a pixel in the detector
class Pixel:
    """Class to represent a pixel in the detector.
    \nAttributes:
    \n - row : row coordinate of the pixel
    \n - col : column coordinate of the pixel
    \n - ts_ext : timestamp of the pixel
    """
    def __init__(self, coords, ts_ext, swapCol = False):
        self.coords = coords
        self.ts_ext = ts_ext
        # swap col if detector is flipped: col 0 becomes col 511, col 511 becomes col 0
        # row stay the same
        if swapCol is True:
            self.row = coords[0] #abs(511 - coords[0])
            self.col =  511 - coords[1]
        else:
            self.row = coords[0]
            self.col = coords[1]        

# Class to represent a cluster of pixels in the detector
class Cluster:
    """
    Class to represent a cluster of pixels in the detector.
    \nAttributes:
    \n - pixels : list of pixels in the cluster
    \n - pixel_timestamps : list of timestamps of the pixels in the cluster
    \n - timestamp : timestamp of the cluster
    \n - delta_col : maximum column difference between pixels in the cluster
    \n - delta_row : maximum row difference between pixels in the cluster
    \n - size : number of pixels in the cluster
    \n - row_center : row coordinate of the cluster center
    \n - col_center : column coordinate of the cluster center
    \n - area : area of the bounding box containing the cluster
    \n - density : density of the cluster
    \n - detector : detector to which the cluster belongs
    \n - timestamp_spread : time spread of the pixels in the cluster
    """
    
    def __init__(self):
        self.pixels =  []
        self.pixel_timestamps = []   
        self.timestamp = None
        self.delta_col = 0
        self.delta_row = 0
        self.size = 0
        self.row_center = 0
        self.col_center = 0
        self.area = 0
        self.density = 0
        self.detector = None
        self.timestamp_spread = 0
        self.status = None

    # Method to add a pixel to the cluster
    def add(self, pix : Pixel):
        """
        Add a pixel to the list of pixels in the cluster.
        """
        self.pixels.append(pix)
        self.pixel_timestamps.append(pix.ts_ext)
        # cluster timestamp is the timestamp of the first hit in time related to the cluster
        if self.timestamp is None or pix.ts_ext < self.timestamp:
            self.timestamp = pix.ts_ext

    # Method to calculate the center of the cluster
    def calculate_center(self):
        mean_row = statistics.mean([x.row for x in self.pixels])
        mean_col = statistics.mean([x.col for x in self.pixels])
        # Add half pixel to "center" cluster row and column
        self.row_center = mean_row + 0.5
        self.col_center = mean_col + 0.5

    # Method to calculate the maximum column difference between pixels in the cluster
    def calculate_delta_col(self):
        self.delta_col=0
        for i,pix1 in enumerate(self.pixels):
            for pix2 in self.pixels[i+1:]:
                h = abs(pix1.col - pix2.col)
                if h > self.delta_col:
                    self.delta_col = h
        self.delta_col += 1 # -> needed because if our clz has 3 pix in a col, then h = 2 (because is the difference between the col coords) but the actual height is 3
    
    # Method to calculate the maximum row difference between pixels in the cluster
    def calculate_delta_row(self):
        self.delta_row=0
        for i,pix1 in enumerate(self.pixels):
            for pix2 in self.pixels[i+1:]:
                w = abs(pix1.row - pix2.row)
                if w > self.delta_row:
                    self.delta_row = w # -> needed because if our clz has 3 pix in a row, then h = 2 (because is the difference between the row coords) but the actual height is 3
        self.delta_row += 1    

    # Method to calculate the density of the cluster
    def calculate_density(self, precision=3):
        """
        Calculate the density of the cluster.

        The density is defined as the number of unique pixels in the cluster
        divided by the area of the smallest bounding box containing the cluster.

        Density is always <= 1. If a value > 1 is obtained, an exception is raised
        because it indicates an inconsistency in the cluster data.
        """

        if len(self.pixels) == 0:
            self.area = 0
            self.density = 0.0
            return self.density

        # Extract coordinates
        rows = [p.row for p in self.pixels]
        cols = [p.col for p in self.pixels]

        # Bounding box limits
        row_min = min(rows)
        row_max = max(rows)
        col_min = min(cols)
        col_max = max(cols)

        # Bounding box dimensions
        height = row_max - row_min + 1
        width = col_max - col_min + 1

        # Bounding box area
        self.area = height * width

        # Count unique pixels only
        unique_pixels = {(p.row, p.col) for p in self.pixels}
        temp_n_pix = len(unique_pixels)

        self.density = np.round(temp_n_pix / self.area, decimals=precision)

        return self.density
    
    # Method to add the detector number to the cluster
    def add_detNo(self, detNo):
        self.detNo = detNo

    # Method to calculate the time spread of the pixels in the cluster
    def calculate_time_spread(self):
        max_ts = max(self.pixel_timestamps)
        min_ts = min(self.pixel_timestamps)
        timestamp_spread = max_ts - min_ts
        return timestamp_spread

    # Function to calculate the cluster dimentions x, y
    def calculate_dimensions(self):
        self.calculate_delta_col()
        self.calculate_delta_row()
        return self.delta_row, self.delta_col
    
# ==================== FUNCTION DEFINITIONS ====================

# Function to check the format of the 'pixels' column and convert it to a list of tuples if necessary
def pixels_check(pixels):
    """
    Function to check the format of the 'pixels' column and convert it to a list of tuples if necessary.
    \n Input:
    \n - pixels: can be a string representation of a list of tuples, or a single tuple, or a list of tuples.
    \n Output:
    \n - A list of tuples representing the pixel coordinates, or (np.nan, np.nan) if the input format is invalid.
    """    
    
    # The input 'pixels' can be a string representation of a list of tuples,
    if isinstance(pixels, str):
        try:
            pixels = ast.literal_eval(pixels)
        except (ValueError, SyntaxError):
            return np.nan, np.nan
    
    # Or it can be a single tuple, in which case we convert it to a list of one tuple.
    if isinstance(pixels, tuple) and len(pixels) == 2:
        pixels = [pixels]
    
    # If it's not a list of tuples, we return NaN for both x and y.
    if not isinstance(pixels, list) or len(pixels) == 0:
        return np.nan, np.nan
    
    return pixels