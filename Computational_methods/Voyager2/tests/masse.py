import numpy as np

mass = np.array([825, 1.98841e+30, 3.301e+23, 4.8675e+24, 6.04568e+24, 6.4171e+23, 1.89858e+27, 5.6847e+26, 8.6811e+25, 1.02409e+26], dtype=np.float64)
mass0 = np.array([825, 1.988410e+30, 3.301e+23, 4.8675e+24, 5.9726e+24, 6.4171e+23, 1.8982e+27, 5.6834e+26, 8.681e+25, 1.024e+26])
moons = mass - mass0

massa_riotta = moons*mass0/(moons + mass0)

print(massa_riotta)