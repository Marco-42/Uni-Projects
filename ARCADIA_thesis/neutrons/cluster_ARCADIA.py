# Class to clusterize the dataframe of hits and calculate cluster properties

# Import necessary libraries
import pandas as pd
import numpy as np
import statistics
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt
from cycler import cycler
import mplhep as hep

# Import helper functions
from ARCADIA_HELPER import *


# Graph style
# Setting plot style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '14',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (8, 6),
         'axes.labelsize': '14',
         'figure.titlesize' : '14',
         'axes.titlesize':'14',
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

# Function to clusterize the dataframe of hits and calculate cluster properties

def _clusterize_df(df: pd.DataFrame, clz_limit: int, time_thr: int, clz_size: int) -> pd.DataFrame:
    """
    Function to clusterize the dataframe of hits and calculate cluster properties.
    \nInput:
    \n - df: dataframe of hits with columns 'row', 'col', 'ts_ext'
    \n - clz_limit: distance cluster treshold 
    \n - time_thr: time cluster threshold
    \n - clz_size: minimum size of a cluster (if 0, no size limit is set)
    \nOutput: dataframe of clusters with columns 'ts_ext', 'pixels', 'multiplicity'
    """
    
    print('[AH_Cluster] Start clusterization')

    # # Remove accidental index column from CSV/Excel exports.
    # if 'Unnamed: 0' in df.columns:
    #     df = df.drop(columns=['Unnamed: 0'])

    # Force core columns to numeric so each hit can be processed safely.
    numeric_cols = ['ts_ext', 'row', 'col']
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with invalid numeric payload.
    df = df.dropna(subset=numeric_cols)

    # Warn and print any hits outside the expected detector/time bounds.
    out_of_bounds_mask = (
        (df['row'] > 513) |
        (df['col'] > 513) |
        (df['ts_ext'] > 10000000000)
    )
    if out_of_bounds_mask.any():
        print('[AH_Cluster] Warning: found out-of-bounds values (row/col > 513 or ts_ext > 10000000000)')
        print(df.loc[out_of_bounds_mask, ['ts_ext', 'row', 'col']].to_string(index=False))
    else: 
        print('[AH_Cluster] All hits within expected bounds.')

    # # Spatial coordinates and timestamps are expected to be integers.
    # df[['ts_ext', 'row', 'col']] = df[['ts_ext', 'row', 'col']].astype(np.int64)

    # Sort hits by timestamp and reset index if not already sorted
    df = df.sort_values(by='ts_ext').reset_index(drop=True)
    time_correlated = []

    # Group pixels by time threshold
    for i in tqdm(range(len(df)), desc=f'Group pixels by time threshold = {time_thr} ts', unit='pkt'):
        
        # Select the pixel at index i
        pixel = df.iloc[i]
        # If it's the first pixel, start a new group
        # If the time difference between the current pixel and the last pixel 
        # in the last group is less than the time threshold, add it to the current group
        if i == 0:
            time_correlated.append([pixel])
        elif pixel['ts_ext'] - time_correlated[-1][0]['ts_ext'] < time_thr:
            time_correlated[-1].append(pixel)
        else:
            time_correlated.append([pixel])

    # Now we have a list of groups of pixels that are time-correlated. 
    # We can now clusterize them spatially.
    cluster_id = 0
    cluster_assignments = [-1] * len(df)

    # For each group of time-correlated pixels, we will clusterize them spatially
    #  using a simple distance-based clustering algorithm.
    for group in tqdm(time_correlated, desc = f'Find spatial correlation < {clz_limit} pixel', unit = 'tw'):
        list_box = []
        for pixel in group:
            idx = pixel.name
            pix = Pixel((pixel['row'], pixel['col']), pixel['ts_ext'])
            list_box.append((pix, idx))

        while list_box:
            new_clz = Cluster()
            pix, idx = list_box.pop(0)
            new_clz.add(pix)
            pixels_to_check = new_clz.pixels[:]
            assigned_indices = [idx]

            while pixels_to_check:
                d = pixels_to_check.pop(0)
                for p, i in list_box[:]:
                    delta_y = abs(p.row - d.row)
                    delta_x = abs(p.col - d.col)
                    if max(delta_x, delta_y) < clz_limit:
                        new_clz.add(p)
                        pixels_to_check.append(p)
                        assigned_indices.append(i)
                        list_box.remove((p, i))

            if len(new_clz.pixels) > clz_size and clz_size != 0:
                continue

            # Assign cluster ID
            for i in assigned_indices:
                cluster_assignments[i] = cluster_id
            cluster_id += 1

    # Add cluster assignments to the dataframe
    df['cluster_id'] = cluster_assignments

    # Create a new column with the pixel coordinates as tuples
    df['pixels'] = list(zip(df.row, df.col))

    max_clz_id = df['cluster_id'].max()

    df_to_check = df[df['cluster_id'] != -1]
    df_not_assigned = df[df['cluster_id'] == -1]

    # Group by cluster_id
    for clz_id, group in tqdm(df_to_check.groupby('cluster_id'), desc = "Check for fake clusters", unit = 'clz'):
        if group['pixels'].nunique() == 1:
            # all pixels identical → need to split
            # keep the first row as is, reassign the rest
            idx_to_change = group.index[1:]
            n = len(idx_to_change)
            # assign new cluster_ids incrementally
            df.loc[idx_to_change, 'cluster_id'] = range(max_clz_id+1, max_clz_id+n+1)
            max_clz_id += n

    print('[AH_Cluster] Check for fake clusters done!')
    del df_to_check
    
    merged_df = df[df['cluster_id'] != -1].groupby('cluster_id').agg({
        'ts_ext': 'min',
        'pixels': list
    }).reset_index()

    print("[AH_Cluster] Merge clusters done!")
    
    # not skip not assigned clusters 
    # unlikely to have those but just in case
    if not df_not_assigned.empty:
        df_not_assigned['pixels'] = df_not_assigned['pixels'].apply(lambda x: [x])
        merged_df = pd.concat([merged_df, df_not_assigned], ignore_index=True)

    del df_not_assigned
    
    # Calculate multiplicity and drop cluster_id column
    merged_df['multiplicity'] = merged_df['pixels'].apply(len) # Count of pixels in cluster
    merged_df.drop(columns=['cluster_id'], inplace=True)

    print('[AH_Cluster] Clusterization done!')

    # Calculate multiplicity [cluster_id, multiplicity]
    cluster_multiplicity = merged_df['pixels'].apply(len)

    print('[AH_Cluster] Get cluster multiplicity done!')

    # Return the merged dataframe with cluster properties
    # New dataframe structure: 'ts_ext', 'pixels', 'multiplicity'
    return  merged_df, cluster_multiplicity

# Function to calculate the center of a cluster given its pixels
def _cluster_center(pixels):
    """
    Function to calculate the center of a cluster given its pixels.
    \nInput: list of pixel coordinates (row, col) in the cluster
    \nOutput: (x_center, y_center) coordinates of the cluster center
    """
    
    # Check if pixels is a string representation of a list of tuples and convert it to a list of tuples if necessary
    pixels = pixels_check(pixels)

    # Now we can calculate the center of the cluster using the Cluster class from ARCADIA_HELPER.
    cluster = Cluster()
    for pixel in pixels:
        if isinstance(pixel, tuple) and len(pixel) == 2:
            cluster.add(Pixel(pixel, 0))

    if len(cluster.pixels) == 0:
        return np.nan, np.nan
    
    # Calculate the center for the single cluster
    cluster.calculate_center()

    return cluster.col_center, cluster.row_center

# Function to normalize cluster pixels (row, col) to ensure they are within the detector bounds
def _normalize_cluster_pixels(pixels):
    """
    Return a clean list of (row, col) tuples from serialized cluster pixels.
    \nInput: pixels can be a string representation of a list of tuples, or a single tuple, or a list of tuples.
    \nOutput: A list of (row, col) tuples representing the pixel coordinates, or an empty list if the input format is invalid.
    """

    # Check if pixels is a string representation of a list of tuples and convert it to a list of tuples if necessary
    pixels = pixels_check(pixels)

    clean_pixels = []
    for pix in pixels:
        if isinstance(pix, tuple) and len(pix) == 2:
            row, col = pix
            try:
                clean_pixels.append((int(row), int(col)))
            except (TypeError, ValueError):
                continue

    return clean_pixels

# Function to extract the dimentions(horizontal and vertical of each cluster)
def _cluster_dimensions(pixels):
    """
    Function to extract the dimensions (horizontal and vertical) of each cluster.
    \nInput: list of pixel coordinates (row, col) in the cluster
    \nOutput: (x, y) of the cluster
    """
    
    # Check if pixels is a string representation of a list of tuples and convert it to a list of tuples if necessary
    pixels = pixels_check(pixels)

    if len(pixels) == 0:
        return np.nan, np.nan

    # Calculate the dimensions of the cluster using the Cluster class from ARCADIA_HELPER.
    cluster = Cluster()
    for pixel in pixels:
        if isinstance(pixel, tuple) and len(pixel) == 2:
            cluster.add(Pixel(pixel, 0))

    # Calculate the dimensions for the single cluster
    x, y = cluster.calculate_dimensions()

    return x, y

# Function to extract the cluster density using the function in ARCADIA_HELPER
def _cluster_density(pixels):
    """
    Function to extract the cluster density using the function in ARCADIA_HELPER.
    \nInput: dataframe of clusters with columns 'ts_ext', 'pixels', 'multiplicity'
    \nOutput: cluster density
    """
        
    # Check if pixels is a string representation of a list of tuples and convert it to a list of tuples if necessary
    pixels = pixels_check(pixels)

    if len(pixels) == 0:
        return np.nan, np.nan

    # Calculate the dimensions of the cluster using the Cluster class from ARCADIA_HELPER.
    cluster = Cluster()
    for pixel in pixels:
        if isinstance(pixel, tuple) and len(pixel) == 2:
            cluster.add(Pixel(pixel, 0))

    # Calculate the cluster density using the function from ARCADIA_HELPER
    cluster_density = cluster.calculate_density()

    return cluster_density

# Function to get the total number of clusters in the dataframe
def get_total_clusters(df: pd.DataFrame) -> int:
    """
    Function to get the total number of clusters in the dataframe.
    \nInput: dataframe of clusters with columns 'ts_ext', 'pixels', 'multiplicity'
    \nOutput: total number of clusters
    """
    
    # The total number of clusters is simply the number of rows in the dataframe
    total_clusters = len(df)
    
    return total_clusters