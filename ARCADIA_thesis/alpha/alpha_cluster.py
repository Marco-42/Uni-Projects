# DATA ANALYSIS FOR ARCADIA SENSOR - NEUTRONS AND GAMMA RAYS - AmBe SOURCE

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import statistics
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cycler import cycler
import mplhep as hep

# Import helper functions
from ARCADIA_HELPER import *

# Import the clusterization function
from cluster_ARCADIA import _clusterize_df, _cluster_center, _normalize_cluster_pixels, get_total_clusters, _cluster_dimensions, _cluster_density



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

# Variable definition and execution settings
FILE_PATH = "./data/"
CVS_NAME = "DATA_alfa_VCASN1.csv"

exist = False # Flag to indicate if the clusterized dataframe was loaded from file
do_it_anyway = False # Flag to force the clusterization process even if the clusterized dataframe already exists

show_all_clusters = False # Flag to show all the clusters
show_all_multiplicity_clusters = False # Flag to show only clusters with multiplicity = COMPLETE_ANALYSIS

# vcasn = [1, 3, 5, 7, 10, 15, 20]
angle = [30] # Working with high threshold

multiplicity_txt_path = "./cluster_multiplicity_all_bins.txt"
ts_unit_ns = 200
graph_increment = 0

COMPLETE_ANALYSIS = 8 # Cluster size to full analysis for clusters
SEARCH_CLUSTER_TIME = 2
 # seconds to search for a cluster with COMPLETE_ANALYSIS multiplicity

# ===========================================================
#                      DATA EXTRACTION
# ===========================================================

with open(multiplicity_txt_path, "w", encoding="utf-8") as txt_file:
    txt_file.write("Cluster multiplicity counts for ANGLE\n")
    txt_file.write("\n")

for a in angle:

    FILE_PATH = f"./data/"
    SAVE_PATH = f"./plots/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"PROCESSING ANGLE {a}...")
    print(" ")
    # Opening the dataframe of hits
    # Search if a clusterized dataframe already exists, if not, create it
    try:
        if not do_it_anyway:
            df_clusters = pd.read_csv(FILE_PATH + "df_cluster.csv")
            cluster_multiplicity = df_clusters['multiplicity']
            print("Clusterized dataframe already exists. Loaded from file. ANGLE: ", a)
            exist = True # Flag to indicate that the clusterized dataframe was loaded from file
        else: 
            raise FileNotFoundError
    except FileNotFoundError:
        print("Clusterized dataframe not found. Starting clusterization process.")
        try: 
            df_hits = pd.read_csv(FILE_PATH + CVS_NAME)
        except FileNotFoundError:
            print(f"Data file {CVS_NAME} not found in {FILE_PATH}. Please check the file path and name.")
            continue

        # Decoding - Clusterization
        df_clusters, cluster_multiplicity = _clusterize_df(df_hits, clz_limit = 2, time_thr = 40, clz_size = 0)

    print("Clusterization complete. Number of clusters found: ", len(df_clusters))

    # Search for cluster multiplicity populations
    class_counts = cluster_multiplicity.value_counts().sort_index()

    # Save multiplicity counts to txt with one block per ANGLE.
    counts_of_clusters = class_counts.reindex(range(1, class_counts.index[-1] + 1), fill_value=0).astype(int)
    with open(multiplicity_txt_path, "a", encoding="utf-8") as txt_file:
        txt_file.write(f"ANGLE: {a}\n")
        txt_file.write("multiplicity\tcount\n")
        for mult, count in counts_of_clusters.items():
            txt_file.write(f"{mult}\t{count}\n")
        txt_file.write("\n")

# ===========================================================
#                 DATA ANALYSIS AND PLOTTING
# ===========================================================

    # PLOTTING HISTOGRAM OF CLUSTER MULTIPLICITY
    plt.figure(figsize=(8,6))
    max_mult = int(cluster_multiplicity.max())
    bins = np.arange(0.5, max_mult + 1.5, 1)
    plt.hist(cluster_multiplicity, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Cluster Multiplicity')
    plt.ylabel('Count')
    plt.title(f'Cluster Multiplicity Distribution - ANGLE {a}')
    if max_mult <= 5:
        xticks = np.arange(1, max_mult + 1, 1)
    else:
        xticks = np.concatenate((np.arange(1, 6, 1), np.arange(10, max_mult + 1, 5)))
    plt.xticks(xticks)
    plt.grid(axis='y', alpha=0.75)
    #plt.yscale('log')  # Use logarithmic scale for better visibility of lower counts

    plt.savefig(SAVE_PATH + f"cluster_multiplicity_hist_alpha.png", dpi=300)
    plt.xlim(0, 25)
    print(f"Graph {graph_increment} - Done")
    graph_increment += 1

    # Extract the vertical and horizontal dimensions of every cluster for each multiplicity
    cluster_dimensions = df_clusters['pixels'].apply(_cluster_dimensions)
    df_clusters['vertical_dimension'] = cluster_dimensions.apply(lambda d: d[0])
    df_clusters['horizontal_dimension'] = cluster_dimensions.apply(lambda d: d[1])

    df_clusters['ratio_x_y'] = df_clusters.apply(
        lambda row: min(row['horizontal_dimension'], row['vertical_dimension']) / max(row['horizontal_dimension'], row['vertical_dimension'])
        if max(row['horizontal_dimension'], row['vertical_dimension']) != 0 else np.nan,
        axis=1
    )
    df_clusters['ratio_x_y_bin'] = df_clusters['ratio_x_y'].round(2)

    # Plotting a colourplot (x axis = x/y ratio, y axis = multiplicity, color = number of clusters) for the cluster dimensions
    plt.figure(figsize=(9, 6))

    multiplicity_min = int(df_clusters["multiplicity"].min())
    multiplicity_max = int(df_clusters["multiplicity"].max())

    y_edges = np.arange(
        multiplicity_min - 0.5,
        multiplicity_max + 1.5,
        1
    )

    ratio_xy_min = df_clusters["ratio_x_y"].min()
    ratio_xy_max = df_clusters["ratio_x_y"].max()

    ratio_offset = (ratio_xy_max - ratio_xy_min)/7

    x_edges = np.arange(ratio_xy_min - ratio_offset/2, ratio_xy_max + ratio_offset/2, ratio_offset)

    h = plt.hist2d(
        df_clusters["ratio_x_y"],
        df_clusters["multiplicity"],
        bins=[x_edges, y_edges],
        cmap="viridis",
        norm=LogNorm(),
        cmin=1
    )


    # Colorbar
    cb = plt.colorbar(h[3])
    cb.set_label("Number of clusters", fontsize=14)

    plt.grid(alpha=0.25)

    plt.xlabel('Ratio x/y')
    plt.ylabel('Multiplicity')

    plt.tight_layout()

    print(f"Graph {graph_increment} - Done")
    graph_increment += 1

    # Getting the cluster density
    cluster_density = df_clusters['pixels'].apply(_cluster_density)
    df_clusters['density'] = cluster_density

    # Plotting only the clusters with ratio x/y between 0.5 and 1.5
    df_filtered = df_clusters[
        (df_clusters['ratio_x_y'] >= 0.5) &
        (df_clusters['ratio_x_y'] <= 1.5)
    ]

    # Plot
    # plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    hb = plt.hexbin(
        df_filtered["density"],
        df_filtered["multiplicity"],
        gridsize=40,          
        cmap="viridis",
        bins="log",       
        mincnt=1           
    )

    # Colorbar
    cb = plt.colorbar(hb)
    cb.set_label("Number of clusters", fontsize=12)

    # Labels
    plt.xlabel("Cluster Density", fontsize=13)
    plt.ylabel("Multiplicity", fontsize=13)
    plt.title(f"Cluster Density - ANGLE {a}", fontsize=16)

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        SAVE_PATH + f"cluster_density_hexbin_alpha.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Graph {graph_increment} - Done")
    graph_increment += 1


    # Number of cluster with: 
    # 0.9 <= density <= 1.0
    # multiplicity > 7

    selected_clusters = df_filtered[
        (df_filtered["density"] >= 0.6) &
        (df_filtered["density"] <= 1.0) &
        (df_filtered["multiplicity"] > 4)
    ]

    n_selected = len(selected_clusters)

    # if exist == False:
    # Save the dataframe of clusters to a CSV file
    df_clusters.to_csv(FILE_PATH + "df_cluster.csv", index=False)

    # ALL CLUSTER MAPS - plotting only if show_all is True
    if show_all_clusters:
        # Calculate the center of each cluster and add it as new columns in the dataframe
        xy_centers = df_clusters['pixels'].apply(_cluster_center)
        df_clusters['x'] = xy_centers.apply(lambda p: p[0]) # Getting x of the cluster center
        df_clusters['y'] = xy_centers.apply(lambda p: p[1]) # Getting y of the cluster center

        # Build a pixel occupancy map on the fixed detector grid.
        valid_centers = df_clusters[['x', 'y']].dropna()

        if valid_centers.empty:
            print(f"No cluster found for ANGLE {a}.")
            print(" ")
            continue

        # Convert center coordinates to pixel indices and keep only physical detector range.
        x_pix = np.floor(valid_centers['x']).astype(int)
        y_pix = np.floor(valid_centers['y']).astype(int)

        in_sensor = (
            (x_pix >= 0)
            & (x_pix < ARCADIA_SIZE[1])
            & (y_pix >= 0)
            & (y_pix < ARCADIA_SIZE[0])
        )
        x_pix = x_pix[in_sensor]
        y_pix = y_pix[in_sensor]

        n_cols = ARCADIA_SIZE[1]
        n_rows = ARCADIA_SIZE[0]

        if len(x_pix) == 0:
            print(f"No valid cluster centers in sensor range for ANGLE {a}.")
            print(" ")
            continue
        
        # PLOTTING CLUSTER CENTERS MAP
        pixel_cluster_counts = np.zeros((n_rows, n_cols), dtype=int)
        np.add.at(pixel_cluster_counts, (y_pix, x_pix), 1)

        plt.figure(figsize=(9, 7))
        img = plt.imshow(
            pixel_cluster_counts,
            origin='lower',
            cmap='inferno',
            interpolation='nearest',
            extent=[0, n_cols, 0, n_rows],
            aspect='equal'
        )
        cbar = plt.colorbar(img)
        cbar.set_label('Number of Cluster Centers for Pixel')

        plt.xlabel('X [pixel]')
        plt.ylabel('Y [pixel]')
        plt.title(f'Cluster Centers Map - ANGLE {a}')
        plt.grid(alpha=0.25)
        plt.savefig(SAVE_PATH + f"cluster_centers_map_alpha.png", dpi=300)
        print(f"Graph {graph_increment} - Done")
        graph_increment += 1

    # Full-cluster map limited to the first 1 second from the first cluster timestamp.
    time_window_ts = int((SEARCH_CLUSTER_TIME * 1000000000) / ts_unit_ns)  # 1 s -> 50000000 ts units

    if 'ts_ext' in df_clusters.columns:
        ts_start = df_clusters['ts_ext'].min()
        df_first_setted_s = df_clusters[df_clusters['ts_ext'] <= (ts_start + time_window_ts)].copy()
    else:
        df_first_setted_s = pd.DataFrame(columns=df_clusters.columns)

    full_cluster_map = np.zeros(ARCADIA_SIZE, dtype=int)
    for pixels in df_first_setted_s.get('pixels', []):
        for row, col in _normalize_cluster_pixels(pixels):
            if 0 <= row < ARCADIA_SIZE[0] and 0 <= col < ARCADIA_SIZE[1]:
                full_cluster_map[row, col] += 1

    # PLOTTING FULL-CLUSTER MAP FOR THE FIRST SEARCH_CLUSTER_TIME seconds
    plt.figure(figsize=(9, 7))
    img_full = plt.imshow(
        full_cluster_map,
        origin='lower',
        cmap='viridis',
        interpolation='nearest',
        extent=[0, ARCADIA_SIZE[1], 0, ARCADIA_SIZE[0]],
        aspect='equal'
    )
    cbar_full = plt.colorbar(img_full)
    cbar_full.set_label('Number of times the pixel appears in clusters within the time window')

    plt.xlabel('X [pixel]')
    plt.ylabel('Y [pixel]')
    plt.title(f'Cluster example - ANGLE {a}')
    plt.grid(alpha=0.25)
    plt.savefig(SAVE_PATH + f"full_cluster_map_alpha_first_{SEARCH_CLUSTER_TIME}s.png", dpi=300)
    print(f"Graph {graph_increment} - Done")
    graph_increment += 1

    # Full-cluster map for multiplicity-COMPLETE_ANALYSIS clusters only, in the same time window.
    if 'multiplicity' in df_first_setted_s.columns:
        df_first_setted_s_mult = df_first_setted_s[df_first_setted_s['multiplicity'] == COMPLETE_ANALYSIS].copy()
    else:
        df_first_setted_s_mult = pd.DataFrame(columns=df_first_setted_s.columns)

    full_cluster_map_mult = np.zeros(ARCADIA_SIZE, dtype=int)
    for pixels in df_first_setted_s_mult.get('pixels', []):
        for row, col in _normalize_cluster_pixels(pixels):
            if 0 <= row < ARCADIA_SIZE[0] and 0 <= col < ARCADIA_SIZE[1]:
                full_cluster_map_mult[row, col] += 1

    # PLOTTING FULL-CLUSTER MAP FOR MULTIPLICITY-COMPLETE_ANALYSIS CLUSTERS IN THE FIRST SEARCH_CLUSTER_TIME seconds
    import matplotlib.colors as mcolors

    track = (full_cluster_map_mult > 0).astype(int)

    cmap = mcolors.ListedColormap([
        (0.2, 0.2, 0.2, 0.2),
        "darkorange"
    ])
    plt.figure(figsize=(9, 7))
    plt.imshow(
        track,
        origin='lower',
        cmap=cmap,
        interpolation='nearest',
        extent=[0, ARCADIA_SIZE[1], 0, ARCADIA_SIZE[0]],
        aspect='equal',
        vmin=0,
        vmax=1
    )

    plt.xlabel('x [pixel]')
    plt.ylabel('y [pixel]')
    plt.title(f'Cluster example - Multiplicity {COMPLETE_ANALYSIS} - ANGLE {a}')
    plt.grid(alpha=0.25)
    plt.savefig(
        SAVE_PATH + f"Cluster_map_mult_{COMPLETE_ANALYSIS}_alpha_first_{SEARCH_CLUSTER_TIME}s.png",
        dpi=300
    )
    print(f"Graph {graph_increment} - Done")
    graph_increment += 1

    if show_all_multiplicity_clusters:
        # Plotting all the cluster with multiplicty COMPLETE_ANALYSIS in all the acquisition time
        if 'ts_ext' in df_clusters.columns:
            ts_end = df_clusters['ts_ext'].max()
            df_first_setted_s = df_clusters[df_clusters['ts_ext'] <= ts_end].copy()
        else:
            df_first_setted_s = pd.DataFrame(columns=df_clusters.columns)

        full_cluster_map = np.zeros(ARCADIA_SIZE, dtype=int)
        for pixels in df_first_setted_s.get('pixels', []):
            for row, col in _normalize_cluster_pixels(pixels):
                if 0 <= row < ARCADIA_SIZE[0] and 0 <= col < ARCADIA_SIZE[1]:
                    full_cluster_map[row, col] += 1

        plt.figure(figsize=(9, 7))
        img_full = plt.imshow(
            full_cluster_map,
            origin='lower',
            cmap='viridis',
            interpolation='nearest',
            extent=[0, ARCADIA_SIZE[1], 0, ARCADIA_SIZE[0]],
            aspect='equal'
        )

        cbar_full = plt.colorbar(img_full)
        cbar_full.set_label('Number of times the pixel appears in clusters')
        plt.xlabel('X [pixel]')
        plt.ylabel('Y [pixel]')
        plt.title(f'{COMPLETE_ANALYSIS} Cluster in all the acquisition time - ANGLE {a}')
        plt.grid(alpha=0.25)
        plt.savefig(SAVE_PATH + f"full_cluster_map_mult_{COMPLETE_ANALYSIS}_alpha_all_time.png", dpi=300)
        print(f"All {COMPLETE_ANALYSIS} Clusters - Done")

    # Getting the histogramm of cluster_multiplicity
    cluster_multiplicity_hist = class_counts

    # Plotting total cluster multiplicity
    plt.figure(figsize=(8,6))
    plt.bar(cluster_multiplicity_hist.index, cluster_multiplicity_hist.values, label='Total Clusters', alpha=0.7, color='navy')
    #plt.bar(simulation_multiplicity_data.keys(), simulation_multiplicity_data.values(), label='Simulation', alpha=0.7)
    plt.xlabel('Cluster Multiplicity')
    plt.ylabel('Counts')
    plt.title(f'Cluster Multiplicity - ANGLE {a}')
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(SAVE_PATH + f"total_cluster_multiplicity_alpha.png", dpi=300)

    # Computing the rate of events in cluster per seconds
    total_time_s = (df_clusters['ts_ext'].max() - df_clusters['ts_ext'].min()) * ts_unit_ns / 1e9
    total_clusters = len(df_clusters)
    rate_clusters_per_second = total_clusters / total_time_s if total_time_s > 0 else 0
    print(f"Rate of clusters per second: {rate_clusters_per_second:.2f}")
    print(f"Total clusters: {total_clusters}, Total time (s): {total_time_s:.2f}")
plt.show()