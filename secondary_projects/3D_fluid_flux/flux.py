import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from cycler import cycler

from Physics_lab import LabH as helper # Importing helper functions

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

# ======= MAIN FUNCTION ================
def main():
    # resolve the relative path to the data file in the same folder as the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data.txt')  # Data NO probe
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data file not found: {data_path}')

    # Reading data from file
    # Data style: x, y, flux, flux_error
    x, y, flux, flux_error = helper.read_data_xyz_errorz(data_path)

    # 3D scatter plot with error bars on z-axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, flux, c=flux, cmap='viridis', label='Data points')

    # Error bars on z-axis
    for xi, yi, zi, err in zip(x, y, flux, flux_error):
        ax.plot([xi, xi], [yi, yi], [zi - err, zi + err], color='gray', alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Flux')
    ax.set_title('3D Flux Data with Z Error Bars')

    outpath = os.path.join(base_dir, '3D_flux.png')
    plt.savefig(outpath, dpi=150)
    plt.legend()
    plt.tight_layout()
    plt.show()
	

if __name__ == '__main__':
	main()
