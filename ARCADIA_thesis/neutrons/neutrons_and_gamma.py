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
FILE_PATH = "./data/vcasn20/intermediate_results/"
CVS_NAME = "RAW_new.csv"

exist = False # Flag to indicate if the clusterized dataframe was loaded from file
do_it_anyway = False # Flag to force the clusterization process even if the clusterized dataframe already exists

show_all_clusters = False # Flag to show all the clusters
show_all_multiplicity_clusters = False # Flag to show only clusters with multiplicity = COMPLETE_ANALYSIS

# vcasn = [1, 3, 5, 7, 10, 15, 20]
vcasn = [20] # Working with high threshold

multiplicity_txt_path = "./clust" \
"er_multiplicity//cluster_multiplicity_all_bins.txt"
ts_unit_ns = 200
graph_increment = 0

COMPLETE_ANALYSIS = 50 # Cluster size to full analysis for clusters
SEARCH_CLUSTER_TIME = 2000 # seconds to search for a cluster with COMPLETE_ANALYSIS multiplicity

# Simulation confrontation parameters
#nuetron_index = 0.19 / (0.19 + 0.13)
#photon_index = 0.13 / (0.19 + 0.13)
#nuetron_index = 0.3258 / (0.3258 + 0.3695)
#photon_index = 0.3695 / (0.3258 + 0.3695)
nuetron_index = 288 / (288 + 231)
photon_index = 231 / (288 + 231)


tot_exp_neutron = []
tot_exp_photon = []
rem_neutron = []
rem_neutron_err = []
rem_photon = []
rem_photon_err = []
simulated_neutron_multiplicity = []
simulated_photon_multiplicity = []

# ===========================================================
#                      DATA EXTRACTION
# ===========================================================

with open(multiplicity_txt_path, "w", encoding="utf-8") as txt_file:
    txt_file.write("Cluster multiplicity counts for VCASN\n")
    txt_file.write("\n")

for i in vcasn:

    # Setting the treshold for the clusterization process based on the VCASN value
    THR = 63 - i

    FILE_PATH = f"./data/vcasn{i}/intermediate_results/"
    SAVE_PATH = f"./plots/vcasn{i}/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"PROCESSING VCASN {i}...")
    print(" ")
    # Opening the dataframe of hits
    # Search if a clusterized dataframe already exists, if not, create it
    try:
        if not do_it_anyway:
            df_clusters = pd.read_csv(FILE_PATH + "df_cluster.csv")
            cluster_multiplicity = df_clusters['multiplicity']
            print("Clusterized dataframe already exists. Loaded from file. VCASN: ", i)
            exist = True # Flag to indicate that the clusterized dataframe was loaded from file
        else: 
            raise FileNotFoundError
    except FileNotFoundError:
        print("Clusterized dataframe not found. Starting clusterization process.")
        df_hits = pd.read_csv(FILE_PATH + CVS_NAME)

        # Decoding - Clusterization
        df_clusters, cluster_multiplicity = _clusterize_df(df_hits, clz_limit = 2, time_thr = 40, clz_size = 0)

    print("Clusterization complete. Number of clusters found: ", len(df_clusters))

    # Search for cluster multiplicity populations
    class_counts = cluster_multiplicity.value_counts().sort_index()

    # Save multiplicity counts to txt with one block per VCASN.
    counts_of_clusters = class_counts.reindex(range(1, class_counts.index[-1] + 1), fill_value=0).astype(int)
    with open(multiplicity_txt_path, "a", encoding="utf-8") as txt_file:
        txt_file.write(f"VCASN: {i}\n")
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
    plt.title(f'Cluster Multiplicity Distribution - THR  {THR}')
    if max_mult <= 5:
        xticks = np.arange(1, max_mult + 1, 1)
    else:
        xticks = np.concatenate((np.arange(1, 6, 1), np.arange(10, max_mult + 1, 5)))
    plt.xticks(xticks)
    plt.grid(axis='y', alpha=0.75)
    #plt.yscale('log')  # Use logarithmic scale for better visibility of lower counts

    plt.savefig(f"./cluster_multiplicity/no_log/cluster_multiplicity_no_log_vcasn{i}.png", dpi=300)
    plt.savefig(SAVE_PATH + f"cluster_multiplicity_hist_vcasn{i}.png", dpi=300)
    plt.xlim(0, 25)
    print(f"Graph {graph_increment} - Done")
    graph_increment += 1

    # Extract the vertical and horizontal dimensions of every cluster for each multiplicity
    cluster_dimensions = df_clusters['pixels'].apply(_cluster_dimensions)
    df_clusters['vertical_dimension'] = cluster_dimensions.apply(lambda d: d[0])
    df_clusters['horizontal_dimension'] = cluster_dimensions.apply(lambda d: d[1])

    # Compute the ration x/y for each cluster and add it as a new column in the dataframe
    # df_clusters['ratio_x_y'] = df_clusters.apply(lambda row: row['horizontal_dimension'] / row['vertical_dimension'] if row['vertical_dimension'] != 0 else np.nan, axis=1)
    
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

    ratio_offset = (ratio_xy_max - ratio_xy_min)/30

    x_edges = np.arange(ratio_xy_min - ratio_offset/1.9, ratio_xy_max + ratio_offset/1.9, ratio_offset)

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
    plt.title(f'Cluster Dimensions - THR {THR}')

    plt.tight_layout()

    plt.savefig(
        SAVE_PATH + f"cluster_dimensions_heatmap_vcasn{i}.png",
        dpi=300
    )

    print(f"Graph {graph_increment} - Done")
    graph_increment += 1

    # plt.figure(figsize=(9, 6))

    # hb = plt.hexbin(
    #     df_clusters["ratio_x_y_bin"],
    #     df_clusters["multiplicity"],
    #     gridsize=40,          
    #     cmap="viridis",
    #     bins="log",       
    #     mincnt=1           
    # )

    # # Colorbar
    # cb = plt.colorbar(hb)
    # cb.set_label("Number of clusters", fontsize=14)

    # plt.grid(alpha=0.25)

    # plt.xlabel('Ratio x/y')
    # plt.ylabel('Multiplicity')
    # plt.title(f'Cluster Dimensions - THR  {THR}')
    # plt.tight_layout()
    # plt.savefig(SAVE_PATH + f"cluster_dimensions_heatmap_vcasn{i}.png", dpi=300)
    # print(f"Graph {graph_increment} - Done")
    # graph_increment += 1

    # Getting the cluster density
    cluster_density = df_clusters['pixels'].apply(_cluster_density)
    df_clusters['density'] = cluster_density

    # Plotting only the clusters with ratio x/y between 0.5 and 1.5
    df_filtered = df_clusters[
        (df_clusters['ratio_x_y'] > 0.6)
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
    cb.set_label("Number of clusters", fontsize=14)

    # Labels
    plt.xlabel("Cluster Density", fontsize=14)
    plt.ylabel("Multiplicity", fontsize=14)
    plt.title(f"Cluster Density - THR  {THR}", fontsize=18)

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        SAVE_PATH + f"cluster_density_hexbin_vcasn{i}.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Graph {graph_increment} - Done")
    graph_increment += 1

    # Plotting on x density * ratio, on y the multiplicity, color = number of clusters
    df_filtered_all_density = df_clusters

    # Computing the ratio as the minimum dimension divided by the maximum dimension for each cluster
    df_filtered_all_density['min_max_ratio'] = df_filtered_all_density.apply(
        lambda row: min(row['horizontal_dimension'], row['vertical_dimension']) / max(row['horizontal_dimension'], row['vertical_dimension'])
        if max(row['horizontal_dimension'], row['vertical_dimension']) != 0 else np.nan,
        axis=1
    )

    density_ratio = df_filtered_all_density['density'] * df_filtered_all_density['min_max_ratio']

    # Plotting the hexbin for density * ratio vs multiplicity
    plt.figure(figsize=(10, 7))
    hb = plt.hexbin(
        density_ratio,
        df_filtered_all_density["multiplicity"],
        gridsize=40,
        cmap="plasma",
        bins="log",
        mincnt=1
    )

    # Colorbar
    cb = plt.colorbar(hb)
    cb.set_label("Number of clusters", fontsize=14)

    # Labels
    plt.xlabel("Cluster Density * ratio", fontsize=14)
    plt.ylabel("Multiplicity", fontsize=14)
    plt.title(f"Cluster Density * Ratio - THR  {THR}", fontsize=18)

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        SAVE_PATH + f"cluster_density_per_ratio_hexbin_vcasn{i}.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Graph {graph_increment} - Done")
    graph_increment += 1

    # Taking data only with this mask
     # Number of cluster with: 
    # 0 <= density * ratio <= 0.6
    # multiplicity >= 3

    selected_clusters = df_filtered_all_density[
        (df_filtered_all_density["density"] * df_filtered_all_density["min_max_ratio"] <= 0.6) &
        (df_filtered_all_density["multiplicity"] >= 3)
    ]

    n_selected = len(selected_clusters)

    if exist == False:
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
            print(f"No cluster found for VCASN {i}.")
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
            print(f"No valid cluster centers in sensor range for VCASN {i}.")
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
        plt.title(f'Cluster Centers Map - THR  {THR}')
        plt.grid(alpha=0.25)
        plt.savefig(SAVE_PATH + f"cluster_centers_map_vcasn{i}.png", dpi=300)
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

    plt.xlabel('X [pixel]', size = 14)
    plt.ylabel('Y [pixel]', size = 14)
    plt.title(f'Cluster example - THR  {THR}', size = 14)
    plt.grid(alpha=0.25)
    plt.savefig(SAVE_PATH + f"full_cluster_map_vcasn{i}_first_{SEARCH_CLUSTER_TIME}s.png", dpi=300)
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

    plt.xlabel('x [pixel]', size = 14)
    plt.ylabel('y [pixel]', size = 14)
    plt.title(f'Cluster example - Multiplicity {COMPLETE_ANALYSIS} - THR  {THR}', size = 14)
    plt.grid(alpha=0.25)
    plt.savefig(
        SAVE_PATH + f"Cluster_map_mult_{COMPLETE_ANALYSIS}_vcasn{i}_first_{SEARCH_CLUSTER_TIME}s.png",
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
        plt.xlabel('X [pixel]', size = 14)
        plt.ylabel('Y [pixel]', size = 14)
        plt.title(f'{COMPLETE_ANALYSIS} Cluster in all the acquisition time - THR  {THR}', size = 14)
        plt.grid(alpha=0.25)
        plt.savefig(SAVE_PATH + f"full_cluster_map_mult_{COMPLETE_ANALYSIS}_vcasn{i}_all_time.png", dpi=300)
        print(f"All {COMPLETE_ANALYSIS} Clusters - Done")

# ===========================================================
#                  SIMULATION CONFRONTATION
# ===========================================================

    # Getting the total number of clusters
    total_clusters = get_total_clusters(df_clusters)

    print(" ")
    print(f"Total number of clusters: {total_clusters}")
    print("Expected neutron percentage: ", nuetron_index * 100, " %")
    print("Expected photon percentage: ", photon_index * 100, " %")

    expected_neutron_clusters = int(total_clusters * nuetron_index)
    expected_photon_clusters = int(total_clusters * photon_index)

    # Confronting with the simulation
    # Getting the histogramm of cluster_multiplicity
    cluster_multiplicity_hist = class_counts

    # FIRST ANALYSIS - REMOVING NEUTRON CLUSTERS UNIFORMALLY FROM THE HISTOGRAMM
    print("FIRST ANALYSIS - Removing neutron clusters uniformally from the histogramm")

    # Nutrons active only cluster with 1 - 2 - 3 - 4 multiplicity
    # Removing uniformally the expected neutron clusters from the histogramm
    photon_multiplicity_hist = cluster_multiplicity_hist.copy()

    neutron_counts = 0
    neutron_counts_err = 0
    for mult in range(1, 5):
        if mult in cluster_multiplicity_hist.index:
            count = cluster_multiplicity_hist[mult]
            if count > 0:
                #to_remove = min(expected_neutron_clusters // 4, count)
                to_remove = min(int(cluster_multiplicity_hist[mult] * nuetron_index), count)
                neutron_counts += to_remove
                neutron_counts_err = np.sqrt(neutron_counts_err**2 + to_remove)  # Assuming Poisson statistics for error estimation
                photon_multiplicity_hist[mult] = cluster_multiplicity_hist[mult] - to_remove
    print(" ")
    print(f"Total neutron clusters removed: {neutron_counts}")
    print(f"Total neutron expected: {expected_neutron_clusters}")
    print(f"Difference expected - evaluated: {expected_neutron_clusters - neutron_counts} clusters = {(expected_neutron_clusters - neutron_counts) / expected_neutron_clusters * 100:.2f} %")
    print(" ")
    print(f"Total photon clusters expected: {expected_photon_clusters}")
    print(f"Total photon clusters evaluated: {photon_multiplicity_hist.sum()}")
    print(f"Difference expected - evaluated: {expected_photon_clusters - photon_multiplicity_hist.sum()} clusters = {(expected_photon_clusters - photon_multiplicity_hist.sum()) / expected_photon_clusters * 100:.2f} %")

    print(" ")    
    print(f"Clusters with 0 ≤ density * min_max_ratio ≤ 0.6 and multiplicity >= 3: {n_selected}")
    print(f"Percentual: {(n_selected / expected_photon_clusters) * 100}")

    tot_exp_neutron.append(expected_neutron_clusters)
    tot_exp_photon.append(expected_photon_clusters)
    rem_neutron.append(neutron_counts)
    rem_neutron_err.append(neutron_counts_err)
    rem_photon.append(photon_multiplicity_hist.sum())
    rem_photon_err.append(np.sqrt(photon_multiplicity_hist.sum()))  # Assuming Poisson statistics for error estimation

    # Getting simulation multeplicity data from simulation from simulated_cluster_multeplicity.txt
    simulation_multiplicity_data = {}
    with open("./simulated_cluster_multiplicity.txt", "r", encoding="utf-8") as sim_file:
        lines = sim_file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                mult = int(parts[0]) + 1
                count = int(parts[1])
                simulation_multiplicity_data[mult] = count
    
    # Plotting grouped bars per multiplicity: Total, Neutron, Photon
    plt.figure(figsize=(8,6))
    # choose multiplicity bins to display (1..6 for consistency with previous xlim)
    mult_range = np.arange(1, 7)
    total_counts = cluster_multiplicity_hist.reindex(mult_range, fill_value=0).to_numpy()
    photon_counts = photon_multiplicity_hist.reindex(mult_range, fill_value=0).to_numpy()
    neutron_counts = total_counts - photon_counts

    n = len(mult_range)
    ind = np.arange(n)
    bar_width = 0.25

    plt.bar(ind - bar_width, total_counts, width=bar_width, color='navy', label='Total Clusters', alpha=0.7)
    plt.bar(ind, neutron_counts, width=bar_width, color='dodgerblue', label='Neutron Clusters', alpha=0.7)
    plt.bar(ind + bar_width, photon_counts, width=bar_width, color='greenyellow', label='Photon Clusters', alpha=0.8)

    plt.xlabel('Cluster Multiplicity')
    plt.ylabel('Counts')
    plt.xticks(ind, mult_range)
    plt.xlim(-1, n)
    plt.title(f'Cluster Multiplicity - THR  {THR}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_PATH + f"cluster_multiplicity_comparison_grouped_vcasn{i}.png", dpi=300)

    # Plotting total cluster multiplicity
    plt.figure(figsize=(8,6))
    plt.bar(cluster_multiplicity_hist.index, cluster_multiplicity_hist.values, label='Total Clusters', alpha=0.7, color='navy')
    #plt.bar(simulation_multiplicity_data.keys(), simulation_multiplicity_data.values(), label='Simulation', alpha=0.7)
    plt.xlabel('Cluster Multiplicity')
    plt.ylabel('Counts')
    plt.title(f'Cluster Multiplicity - THR  {THR}')
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(SAVE_PATH + f"total_cluster_multiplicity_vcasn{i}.png", dpi=300)

    # Plotting only the photon multiplicity histogram with logarithmic scale
    plt.figure(figsize=(8,6))
    plt.bar(photon_multiplicity_hist.index, photon_multiplicity_hist.values, label='Photon Clusters', alpha=0.6, color = 'navy')
    plt.xlabel('Cluster Multiplicity')
    plt.ylabel('Counts')
    plt.title(f'Photon Cluster Multiplicity - THR  {THR}')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(SAVE_PATH + f"photon_cluster_multiplicity_log_vcasn{i}.png", dpi=300)

    # Plotting the difference between photon multepicity histogram and simulation
    # plt.figure(figsize=(8,6))
    # sim_mults = np.array(list(simulation_multiplicity_data.keys()))
    # sim_counts = np.array(list(simulation_multiplicity_data.values()))
    # photon_mults = photon_multiplicity_hist.index.to_numpy()
    # photon_counts = photon_multiplicity_hist.values
    # # Interpolating simulation counts to photon multiplicity bins
    # sim_counts_interp = np.interp(photon_mults, sim_mults, sim_counts)
    # difference = photon_counts - sim_counts_interp
    # plt.bar(photon_mults, np.abs(difference), label='Difference (Photon - Simulation)', alpha=0.7)
    # plt.xlabel('Cluster Multiplicity')
    # plt.ylabel('Difference in Number of Clusters')
    # plt.title('Difference Between Photon Cluster Multiplicity and Simulation')
    # plt.legend()
    # plt.tight_layout()

    exp_mult_1_to_4 = cluster_multiplicity_hist.reindex(range(1, 5), fill_value=0).to_numpy()
    print(exp_mult_1_to_4)
    sim_mult_1_to_4 = np.array([simulation_multiplicity_data.get(mult, 0) for mult in range(1, 5)])
    print(sim_mult_1_to_4)
    simulated_neutron_multiplicity.append(int(np.sum(exp_mult_1_to_4 - sim_mult_1_to_4)))

print(" ")
print("VCASN - Total neutron expected - Total neutron removed")
for vca, exp, rem, sim in zip(vcasn, tot_exp_neutron, rem_neutron, simulated_neutron_multiplicity):
    print(vca, " ", exp, " ", rem, " ", sim)
print("VCASN - Total photon expected - Total photon evaluated")
for vca, exp, rem in zip(vcasn, tot_exp_photon, rem_photon):
    print(vca, " ", exp, " ", rem)


def set_stacked_bottom_xaxes(primary_ax, secondary_ax, x_values):
    thr_labels = [str(63 - x) for x in x_values]
    vcasn_labels = [str(x) for x in x_values]

    primary_ax.set_xlabel('THR', fontsize=13)
    primary_ax.set_xticks(x_values)
    primary_ax.set_xticklabels(thr_labels)
    primary_ax.tick_params(axis='x', pad=4, length=3)

    secondary_ax.set_xlabel('VCASN', fontsize=13)
    secondary_ax.set_xticks(x_values)
    secondary_ax.set_xticklabels(vcasn_labels)
    secondary_ax.tick_params(axis='x', pad=4, length=3)
    secondary_ax.xaxis.set_ticks_position('bottom')
    secondary_ax.xaxis.set_label_position('bottom')
    secondary_ax.spines['bottom'].set_position(('outward', 40))
    secondary_ax.spines['top'].set_visible(False)
    secondary_ax.set_xlim(primary_ax.get_xlim())

# Plotting the difference between total neutron expected and total neutron removed for each VCASN
plt.figure(figsize=(8,6))
ax = plt.gca()
tot_exp_neutron = np.array(tot_exp_neutron)
tot_exp_photon = np.array(tot_exp_photon)
rem_photon = np.array(rem_photon)
rem_photon_err = np.array(rem_photon_err)
rem_neutron = np.array(rem_neutron)
rem_neutron_err = np.array(rem_neutron_err)
photon_difference = tot_exp_photon - rem_photon
photon_difference_err = np.sqrt(tot_exp_photon + rem_photon_err**2)
neutron_difference = tot_exp_neutron - rem_neutron
neutron_difference_err = np.sqrt(tot_exp_neutron + rem_neutron_err**2)
plt.errorbar(vcasn, neutron_difference, yerr=neutron_difference_err, color='darkred', alpha=0.7, marker='o', capsize=5)
plt.errorbar(vcasn, photon_difference, yerr=photon_difference_err, color='darkblue', alpha=0.7, marker='o', capsize=5)
plt.ylabel('Difference in Number of Clusters')
plt.title('Difference Between Total Expected cluster and Total Removed')
secax = ax.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
set_stacked_bottom_xaxes(ax, secax, vcasn)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.subplots_adjust(bottom=0.31)
plt.savefig("./plots/difference_expected_removed.png", dpi=300)

# Plotting the percentual difference between total neutron expected and total neutron removed for each VCASN
plt.figure(figsize=(8,6))
ax = plt.gca()
photon_ratios = 100 * (photon_difference / tot_exp_photon)
photon_ratios_err = photon_ratios * np.sqrt(photon_difference_err**2 / photon_difference**2 + tot_exp_photon/tot_exp_photon**2)
neutron_ratios = 100 * (neutron_difference / tot_exp_neutron)
neutron_ratios_err = neutron_ratios * np.sqrt(neutron_difference_err**2 / neutron_difference**2 + tot_exp_neutron/tot_exp_neutron**2)
plt.errorbar(vcasn, neutron_ratios, yerr=np.abs(neutron_ratios_err), color='darkred', alpha=0.7, fmt='o', capsize=5, label = 'Neutrons')
plt.errorbar(vcasn, photon_ratios, yerr=np.abs(photon_ratios_err), color='darkblue', alpha=0.7, fmt='o', capsize=5, label = "Photons")
plt.ylabel('Percentual Difference in Number of Clusters (%)')
plt.title('Difference Between Total Expected cluster and Total Removed')
secax = ax.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
set_stacked_bottom_xaxes(ax, secax, vcasn)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.subplots_adjust(bottom=0.31)
plt.legend()
plt.savefig("./plots/percentual_difference_expected_removed.png", dpi=300)
plt.show()
