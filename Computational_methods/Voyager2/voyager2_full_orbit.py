import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants
from mpl_interactions import ioff, panhandler, zoom_factory
from numba import njit
from datetime import datetime, timedelta

# ============== GENERAL VARIABLES DEFINITION ==============

# Mass array considering the mass of the single planet system
#voyager2 - sun - mercury - venus - earth - mars - jupiter - saturn - uranus - neptune
mass = np.array([825, 1.98841e+30, 3.301e+23, 4.8675e+24, 6.04568e+24, 6.4171e+23, 1.89858e+27, 5.6847e+26, 8.6811e+25, 1.02409e+26], dtype=np.float64)
ua = 1.496e+11 # Astronomical unit
G = constants.G # Gravitational constant
count = 0
idx_found = []
idx_found_flag = False

# ============== USEFULL FUNCTIONS =======================

# Function to compute the orbital velocity
def vel_orb(dist):
    """Get the tangential velocity of a planet in orbit around the sun - Parameter: distance"""
    return  np.sqrt(mass[1]*G/dist)

# Function to compute the orbital period
def temp_orb(dist):
    """Get the orbital period of a circular orbit around the sun - Parameter: distance"""
    return 2*np.pi*np.sqrt(dist**3/(mass[1]*G))

# Function to visualize the computation time progressive
def time_progressive():
    global count
    for i in range(count):
        print("█", end="", flush=True)
    count += 1

# ============== INITIAL PARAMETERS SETTING ==============

# Function to set the initial parameters(positions and velocities) for the simulation
def set_initial_par(tstep):
    """Return planets_vector - spacecraft_vector - Initial_simulation_time(datetime)"""

    # Defining a vector for the Voyager 2 spacecraft
    spacecraft = np.zeros((tstep, 6), dtype = np.float64)

    # Defining a vector for each planet to simulate[ 8 objects - tstep point to simulate - 6 parameters]
    y = np.zeros((9, tstep,6), dtype = np.float64)

    # DATA ORIGIN: 1977-08-21 15:32:00 - https://ssd.jpl.nasa.gov/horizons/
    # Setting the initial positions and velocities(3D) for the planets
    y[0, 0] = [3.367107777747086e+08, -6.577270346042727e+08, -9.077003721988382e+06, 1.390533190887513e+01, -2.586244414562326e-01, -3.775929494275127e-01] # Sun
    y[1, 0] = [2.454102984319427e+10, -6.253475904215566e+10, -7.284910826512132e+09, 3.562425641387389e+04, 2.021374161404849e+04, -1.620268961396410e+03] # Mercury
    y[2, 0] = [6.183582174997857e+10, 8.812590034782256e+10, -2.352030104596667e+09, -2.888638626349391e+04, 1.978757370701472e+04, 1.937555211543902e+03] # Venus   
    y[3, 0] = [1.297227034766012e+11, -7.911086962834534e+10, -1.229743213462830e+07, 1.497458563207181e+04, 2.536102345569274e+04, 9.405773425665842e-01] # Earth   
    y[4, 0] = [1.550779921884853e+11, 1.538180600795949e+11, -5.854617965897546e+08, -1.617623593907502e+04, 1.920777802730366e+04, 8.005812460247031e+02] # Mars    
    y[5, 0] = [1.224039940672899e+11, 7.513147905197932e+11, -5.835915175926865e+09, -1.304927572505666e+04, 2.704862616033336e+03, 2.811363647690763e+02] # Jupiter 
    y[6, 0] = [-1.067101342017617e+12, 8.632000084440186e+11, 2.730690647535652e+10, -6.596007880347378e+03, -7.534834038342594e+03, 3.939302572985932e+02] # Saturn 
    y[7, 0] = [-2.082426817441853e+12, -1.842915809823593e+12, 2.017664729493368e+10, 4.462513930525587e+03, -5.418009047801510e+03, -7.805646950723010e+01] # Uranus
    y[8, 0] = [-1.131090781790213e+12, -4.386684561917005e+12, 1.163847833461907e+11, 5.228459426280597e+03, -1.326745879809566e+03, -9.313404118778423e+01] # Neptune

    # Setting the initial position and velocity for the Voyager 2 spacecraft
    spacecraft[0] = [1.299432920966462e+11, -7.828934081046111e+10, 2.768578653356545e+08, 1.733991264158666e+04, 3.470347470774246e+04, 3.269173166451550e+03] # spacecraft

        
    # Initial simulation date
    start_date = datetime(1977, 8, 21, 15, 32, 0)

    # Return the two vectors 
    return y, spacecraft, start_date

# ============== SIMULATION HELP FUNCTIONS ===============

# Function to compute planet's velocity and acceleration
@njit # To increase performance
def func(y, x, planet):
    """All planets vector - Planet index - Single planet vector"""

    f = np.zeros(6, dtype=np.float64)
    # r_sun = y[0, :3] # Getting sun position
    r = planet[:3] # Single planet position relative to the sun
    f[:3] = planet[3:]
    # f[3:] = -mass[1]*G*r/((np.linalg.norm(r))**3) # Gravitational force only by the sun

    # Compute planets acceleration
    acceleration = np.zeros(3, dtype=np.float64)

    # boolean vector True if idx[k] != x
    idx = np.arange(y.shape[0]) != x

    # Getting planets positions and masses by boolean indexing
    r2 = y[np.arange(y.shape[0])[idx], :3]
    m2 = mass[np.arange(y.shape[0])[idx] + 1]

    # Computing the distance between the studied planet and others
    diff = r2 - r # Vectorial variable(each row identify a single planet)
    n = diff.shape[0]
    dist = np.empty(n, dtype=np.float64)
    for j in range(n):
        dist[j] = np.sqrt(diff[j,0]**2 + diff[j,1]**2 + diff[j,2]**2)

    # Computing the acceleration components for each planet 
    all_accelerations = np.zeros_like(diff)
    for j in range(n):
        if dist[j] > 0:
            all_accelerations[j] = (m2[j] * diff[j]) / (dist[j]**3)
    # Get the resulting acceleration
    acceleration = np.zeros(3, dtype=np.float64)
    for j in range(n):
        acceleration += all_accelerations[j]
    f[3:] = G * acceleration

    return f

# Function to compute spacecraft velocity and acceleration
@njit # To increase performance
def func_spacecraft(y, spacecraft):
    "All planets vector - Spacecraft vector"

    f = np.zeros(6, dtype=np.float64)
    r = spacecraft[:3] # Spacecraft position
    f[:3] = spacecraft[3:]

    # # Compute the spacecraft's acceleration only by the sun
    # f[3:] = -mass[1]*G*r/((np.linalg.norm(r))**3)

    # Including the gravitational interaction from other planets
    # Getting planets positions and masses
    r2 = y[:, :3]
    m2 = mass[1:]

    # Computing the distance between the spacecraft and the planets
    diff = r2 - r # Vectorial variable(each row identify a single planet)
    n = diff.shape[0]
    dist = np.empty(n, dtype=np.float64)
    for j in range(n):
        dist[j] = np.sqrt(diff[j,0]**2 + diff[j,1]**2 + diff[j,2]**2)

    # Computing the acceleration components for each planet 
    all_accelerations = np.zeros_like(diff)
    for j in range(n):
        if dist[j] > 0:
            all_accelerations[j] = (m2[j] * diff[j]) / (dist[j]**3)
    # Get the resulting acceleration
    acceleration = np.zeros(3, dtype=np.float64)
    for j in range(n):
        acceleration += all_accelerations[j]
    f[3:] += G * acceleration

    return f

# Function to compute the total energy of the spacecraft
@njit # To increase performance
def get_energy(y, spacecraft):
    "All planets vector - Spacecraft vector"

    r = spacecraft[:3] # Spacecraft position

    # Computing the spacecraft kinetic energy
    H = 0.5 * mass[0] * (np.linalg.norm(spacecraft[3:])**2)

    # Compute the spacecraft potential energy
    # Getting planets positions and masses
    r2 = y[:, :3]
    m2 = mass[1:]

    # Computing the distance between the spacecraft and the planets
    diff = r2 - r # Vectorial variable(each row identify a single planet)
    n = diff.shape[0]
    dist = np.empty(n, dtype=np.float64)
    for j in range(n):
        dist[j] = np.sqrt(diff[j,0]**2 + diff[j,1]**2 + diff[j,2]**2)

    # Computing the interaction potential energy for each planet 
    for j in range(n):
        if dist[j] > 0:
            H += -G*(m2[j]*mass[0]) / (dist[j])

    return H

# Function to compute the delta_time between each time step
def get_delta_time(y, spacecraft, reference, i):
    """Planets vector - Spacecraft vector - Reference time - increment number"""
    global idx_found
    global idx_found_flag

    # Computing distances between selected planets and the spacecraft
    dist_jupiter = np.linalg.norm(y[5, i, :3] - spacecraft[i, :3])
    dist_saturn = np.linalg.norm(y[6, i, :3] - spacecraft[i, :3])
    dist_earth = np.linalg.norm(y[3, i, :3] - spacecraft[i, :3])

    # Founding the minimum distance 
    min_dist = min(dist_jupiter, dist_saturn, dist_earth)

    # Changing delta time if the distance is less than ua/5
    if min_dist < ua:
        if idx_found_flag == False:
            idx_found = np.append(idx_found, i)
            idx_found_flag = True
        return reference / (200000 * 20)
    else:
        if idx_found_flag == True:
            idx_found = np.append(idx_found, i)
            idx_found_flag = False
        return reference / (200000 * 5)

# ============== RUNGE KUTTA =============================

# Function to execute the Runge - Kutta method to solve the differential equations for planets and spacecraft
@njit # To increase performance
def runge_kutta(y, spacecraft, dt, i):
    """Runge-Kutta method: planets vector - spacecraft vector - delta_time - for index - type(True = planets and spacecraft, False = only planets)"""

    y_return = np.zeros((y.shape[0], 6), dtype = np.float64)

    # Applying Runge - Kutta method to solve the planet's differential equation(for each planet)
    for k in range(y.shape[0]):
        Y1 = y[k, i,:]
        Y2 = Y1 + func(y[:, i, :], k, Y1)*dt/2
        Y3 = Y1 + func(y[:, i, :], k, Y2)*dt/2
        Y4 = Y1 + func(y[:, i, :], k, Y3)*dt
        y_return[k] = Y1+(func(y[:, i, :], k, Y1) + 2*func(y[:, i, :], k, Y2) + 2*func(y[:, i, :], k, Y3)+func(y[:, i, :], k, Y4))*dt/6

    # Applying Runge - Kutta method to solve the spacecraft's differential equation
    Y1 = spacecraft[i]
    Y2 = Y1 + func_spacecraft(y[:, i, :], Y1)*dt/2
    Y3 = Y1 + func_spacecraft(y[:, i, :], Y2)*dt/2
    Y4 = Y1 + func_spacecraft(y[:, i, :], Y3)*dt
    spacecraft_return = Y1+(func_spacecraft(y[:, i, :], Y1) + 2*func_spacecraft(y[:, i, :], Y2) + 2*func_spacecraft(y[:, i, :], Y3)+func_spacecraft(y[:, i, :], Y4))*dt/6

    # Return planets and spacecraft positions
    return y_return, spacecraft_return

# Function to execute the Runge - Kutta method to solve the differential equations for only planets
@njit # To increase performance
def runge_kutta_planets_orbit(y, dt, i):
    """Runge-Kutta method: planets vector - delta_time - for index"""

    y_return = np.zeros((9, 6), dtype = np.float64)

    # Applying Runge - Kutta method to solve the planet's differential equation(for each planet)
    for k in range(9):
        Y1 = y[k, i,:]
        Y2 = Y1 + func(y[:, i, :], k, Y1)*dt/2
        Y3 = Y1 + func(y[:, i, :], k, Y2)*dt/2
        Y4 = Y1 + func(y[:, i, :], k, Y3)*dt
        y_return[k] = Y1+(func(y[:, i, :], k, Y1) + 2*func(y[:, i, :], k, Y2) + 2*func(y[:, i, :], k, Y3)+func(y[:, i, :], k, Y4))*dt/6

    # Return only planets positions
    return y_return

# ============== EXPLICIT EULERO =========================
# Function to execute the Explicit Eulero method to solve the differential equations for only planets
@njit # To increase performance
def explicit_eulero_planets_orbit(y, dt, i):
    """Explicit Eulero method: planets vector - delta_time - for index"""

    y_return = np.zeros((9, 6), dtype = np.float64)

    # Applying Explicit Eulero method to solve the planet's differential equation(for each planet)
    for k in range(9):
        Y1 = y[k, i,:]
        y_return[k] = Y1 + func(y[:, i, :], k, Y1)*dt

    # Return only planets positions
    return y_return

# ============== PLOTTING FUNCTION =======================
# Function to plot the planets and spacecraft trajectory
def plot_trajectory(planets, voy, tstep):
    """Planets vector - Spacecraft vector - Animation flag = True to animate the trajectory"""

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=None, hspace=None)

    ax.set_title("Voyager 2 Trajectory in the Solar System", pad=20, fontsize=14)
    ax.set_xlim3d([-6*ua, 6*ua])
    ax.set_ylim3d([-6*ua, 6*ua])
    ax.set_zlim3d([-6*ua, 6*ua])

    sun, = ax.plot([], [], [],'.', markersize=11, color = 'yellow')
    mercury, = ax.plot([], [], [],'-', markersize=9, color = 'gray')
    mercurydot, = ax.plot([], [], [],'.', markersize=9, color = 'gray')
    venus, = ax.plot([], [], [],'-', markersize=9, color = 'goldenrod')
    venusdot, = ax.plot([], [], [],'.', markersize=9, color = 'goldenrod')
    earth, = ax.plot([], [], [],'-', markersize=9, color = 'green')
    earthdot, = ax.plot([], [], [],'.', markersize=9, color = 'green')
    mars, = ax.plot([], [], [],'-', markersize=9, color = 'red')
    marsdot, = ax.plot([], [], [],'.', markersize=9, color = 'red')
    jupiter, = ax.plot([], [], [],'-', markersize=9, color = 'brown')
    jupiterdot, = ax.plot([], [], [],'.', markersize=9, color = 'brown')
    saturn, = ax.plot([], [], [],'-', markersize=9, color = 'orange')
    saturndot, = ax.plot([], [], [],'.', markersize=9, color = 'orange')
    uranus, = ax.plot([], [], [],'-', markersize=9, color = 'lightblue')
    uranusdot, = ax.plot([], [], [],'.', markersize=9, color = 'lightblue')
    neptune, = ax.plot([], [], [],'-', markersize=9, color = 'blue')
    neptunedot, = ax.plot([], [], [],'.', markersize=9, color = 'blue')

    voyager2, = ax.plot([],[],[],'-', markersize=9, color = 'white')
    voyager2dot, = ax.plot([],[],[],'.', markersize=9, color = 'white')

    def animate(frame):
        step = 300
        if((frame > idx_found[0] and frame < idx_found[1]) 
        or (frame > idx_found[2] and frame < idx_found[3])):
            i = frame * 2000
        else: 
            i = frame * step
        # Selecting only valid frames
        i = min(i, planets.shape[1]-1)
        trail = 1000  # planet trail length

        sun.set_data(planets[0, max(0,i-trail):i, 0], planets[0, max(0,i-trail):i, 1])
        sun.set_3d_properties(planets[0, max(0,i-trail):i, 2])
        mercury.set_data(planets[1, max(0,i-trail):i, 0], planets[1, max(0,i-trail):i, 1])
        mercury.set_3d_properties(planets[1, max(0,i-trail):i, 2])
        mercurydot.set_data([planets[1, i, 0]], [planets[1, i, 1]])
        mercurydot.set_3d_properties([planets[1, i, 2]])
        venus.set_data(planets[2, max(0,i-trail):i, 0], planets[2, max(0,i-trail):i, 1])
        venus.set_3d_properties(planets[2, max(0,i-trail):i, 2])
        venusdot.set_data([planets[2, i, 0]], [planets[2, i, 1]])
        venusdot.set_3d_properties([planets[2, i, 2]])
        earth.set_data(planets[3, max(0,i-trail):i, 0], planets[3, max(0,i-trail):i, 1])
        earth.set_3d_properties(planets[3, max(0,i-trail):i, 2])
        earthdot.set_data([planets[3, i, 0]], [planets[3, i, 1]])
        earthdot.set_3d_properties([planets[3, i, 2]])
        mars.set_data(planets[4, max(0,i-trail):i, 0], planets[4, max(0,i-trail):i, 1])
        mars.set_3d_properties(planets[4, max(0,i-trail):i, 2])
        marsdot.set_data([planets[4, i, 0]], [planets[4, i, 1]])
        marsdot.set_3d_properties([planets[4, i, 2]])
        jupiter.set_data(planets[5, max(0,i-trail):i, 0], planets[5, max(0,i-trail):i, 1])
        jupiter.set_3d_properties(planets[5, max(0,i-trail):i, 2])
        jupiterdot.set_data([planets[5, i, 0]], [planets[5, i, 1]])
        jupiterdot.set_3d_properties([planets[5, i, 2]])
        saturn.set_data(planets[6, max(0,i-trail):i, 0], planets[6, max(0,i-trail):i, 1])
        saturn.set_3d_properties(planets[6, max(0,i-trail):i, 2])
        saturndot.set_data([planets[6, i, 0]], [planets[6, i, 1]])
        saturndot.set_3d_properties([planets[6, i, 2]])
        uranus.set_data(planets[7, max(0,i-trail):i, 0], planets[7, max(0,i-trail):i, 1])
        uranus.set_3d_properties(planets[7, max(0,i-trail):i, 2])
        uranusdot.set_data([planets[7, i, 0]], [planets[7, i, 1]])
        uranusdot.set_3d_properties([planets[7, i, 2]])
        neptune.set_data(planets[8, max(0,i-trail):i, 0], planets[8, max(0,i-trail):i, 1])
        neptune.set_3d_properties(planets[8, max(0,i-trail):i, 2])
        neptunedot.set_data([planets[8, i, 0]], [planets[8, i, 1]])
        neptunedot.set_3d_properties([planets[8, i, 2]])
        voyager2.set_data(voy[:i, 0], voy[:i, 1])
        voyager2.set_3d_properties(voy[:i, 2])
        voyager2dot.set_data([voy[i, 0]], [voy[i, 1]])
        voyager2dot.set_3d_properties([voy[i, 2]])
        return earth, jupiter, saturn, uranus, neptune, earthdot, jupiterdot, saturndot, uranusdot, neptunedot, sun, voyager2, mars, marsdot, venus, venusdot, mercury, mercurydot
        
    anim = animation.FuncAnimation(fig, animate, repeat=True, frames=tstep-2, interval=1, blit=False)
    disconnect_zoom = zoom_factory(ax)
    pan_handler = panhandler(fig)

    plt.show()
# ============== MAIN ====================================
def main():

    # Method selector
    method_type = {1: runge_kutta, 2: explicit_eulero_planets_orbit, 3: "comparison"}
    
    tstep = 360000 # Total time steps to compute
    
    time = [] # Time vector
    H = np.zeros(tstep) # Total energy list
    K = np.zeros(tstep) # Kinetic energy list

    # Plotting variables
    planet_names = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    colors = ['Yellow', 'gray', 'goldenrod', 'green', 'red', 'brown', 'orange', 'lightblue', 'blue']

    # TCM check flags
    time_check_done = np.array([False, False, False, False], dtype=bool)
    
    # Get the initial parameters
    planets, voy, TIME = set_initial_par(tstep)

    # Setting the initial display for simulation setting
    print("============= VOYAGER 2 TRAJECTORY SIMULATION =============")
    print("Simulation start date: ", str(TIME))
    print(" ")
    print("Select the simulation: ")
    print("1 - Runge Kutta (4th order) voyager 2 trajectory")
    print("2 - Planets orbit simulation")
    print("3 - Planets orbit comparison Runge-kutta - Explicit Eulero")
    method = int(input("Method: "))
    if method not in method_type.keys():
        print("Method not valid, setting Runge Kutta as default")
        method = 1

    # Loading bar
    print(" ")
    print("0 %                                                                  100 %")
    print("┌──────────────────────────────────────────────────────────────────┐")

    # Voyager 2 trajectory simulation using Runge Kutta method[Solar system]
    if method == 1:
        
        # Getting the reference time for delta time (in metri)
        reference_time = temp_orb(np.linalg.norm(planets[6, 0, :3]))

        # Simulation loop
        for i in range(tstep - 1):
            
            # Updating loading bar
            if(i%30000 == 0):
                time_progressive()
 
            # TCM maneuvers execution - For each TCM velocity and mass are changed
            # TCM maneuver during earth - jupitar navigation
            if not time_check_done[0] and TIME > datetime(1977, 10, 12): # TCM 1
                time_check_done[0] = True
                voy[i, 3:] += (0.05619961313385602, -0.06559218449808668, -0.11879086804134076)
                mass[0] = 824.93
            if not time_check_done[1] and TIME > datetime(1978, 6, 4): # TCM 2
                time_check_done[1] = True
                voy[i, 3:] += (2.507599814178363, 22.336311449151935, 2.8694242283308915)
                mass[0] = 814.48
            # TCM maneuver during jupiter pre-approach
            if not time_check_done[2] and TIME > datetime(1979, 5, 30): # TCM 3
                time_check_done[2] = True
                voy[i, 3:] += (-9.254002568829023, 11.697305500343369, -1.8206746499206394)
                mass[0] = 807.63
            if not time_check_done[3] and TIME > datetime(1979, 6, 28): # TCM 4
                time_check_done[3] = True
                voy[i, 3:] += (-10.823419538086299, 47.08880790957682, -8.854442214513924)
                mass[0] = 785.62

            # Compute the delta time for the current time step
            dt = get_delta_time(planets, voy, reference_time, i)
            
            # Update the planets and spacecraft position using Runge Kutta method
            planets[:, i+1], voy[i+1] = runge_kutta(planets, voy, dt, i)

            # Storing the first value for time and energy values
            if i == 0: 
                time.append(TIME)
                H[i] = get_energy(planets[:, i, :], voy[i, :])
                K[i] = 0.5 * mass[0] * (np.linalg.norm(voy[i, 3:])**2)

            # Local update of time variable
            TIME = TIME + timedelta(seconds=dt)
            time.append(TIME)

            # Computing total and kinetic energy of the spacecraft
            H[i+1] = get_energy(planets[:, i+1, :], voy[i+1, :])
            K[i+1] = 0.5 * mass[0] * (np.linalg.norm(voy[i+1, 3:])**2)

        # Showing minimum distance from jupiter and saturn
        distance_giove = np.linalg.norm(planets[5, :, :3] - voy[:, :3], axis=1)
        min_dist = np.min(distance_giove)
        min_index = np.argmin(distance_giove)
        print(" ")
        print(f"Minimum distance from jupiter: {min_dist:.2e} m at step {min_index}")

        distance_saturno = np.linalg.norm(planets[6, :, :3] - voy[:, :3], axis=1)
        min_dist_sat = np.min(distance_saturno)
        min_index_sat = np.argmin(distance_saturno)
        print(f"Minimum distance from saturn: {min_dist_sat:.2e} m at step {min_index_sat}")

        # Plotting voyager 2 velocities in function of time
        fig = plt.figure(figsize=(10,6))
        plt.plot(time, np.linalg.norm(voy[:, 3:], axis=1), color = "Red", label="Voyager 2")
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Voyager 2 velocity [relative to the sun]')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        
        # Adding text about TCM
        tcm_dates = [datetime(1977, 10, 12), datetime(1978, 6, 4), datetime(1979, 5, 30), datetime(1979, 6, 28)]
        tcm_labels = ["TCM 1", "TCM 2", "TCM 3", "TCM 4"]
        for tcm_date, tcm_label in zip(tcm_dates, tcm_labels):
            # Find the index of the closest time greater than or equal to tcm_date
            idx = next((i for i, t in enumerate(time) if isinstance(t, datetime) and t >= tcm_date), None)
            if idx is not None:
                plt.axvline(x=time[idx], color='gray', linestyle='--', alpha=0.7) # Vertical line
                plt.text(time[idx], np.linalg.norm(voy[idx, 3:]), f"{tcm_label}\n{tcm_date.strftime('%Y-%m-%d')}", color='white', fontsize=8, rotation=45, va='bottom', ha='left', backgroundcolor='black')

        # Plotting voyager 2 energy in function of time
        fig = plt.figure(figsize=(10,6))
        plt.plot(time, H, color = "Blue", label="Total Energy")
        plt.plot(time, K, color = "Cyan", label="Kinetic Energy")
        plt.plot(time, H - K, color = "Gold", label="Potential Energy")
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.title('Voyager 2 energy')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()

        plt.show() 

        plot_trajectory(planets, voy, tstep)

    # Simulating planets orbits
    if method == 2 or method == 3:

        tstep = 35000 # Redefining the time steps for planets orbit simulation

        # Get the initial parameters
        planets, voy, TIME = set_initial_par(tstep)
        planets_eulero = planets.copy() # Copy of the planets vector for Explicit Eulero method

        # Compute the delta time for each planet(1/20000 orbit time)
        dt = temp_orb(np.linalg.norm(planets[7, 0, :3]))/15000 # delta time is setted constant using saturn orbit time

        # Simulation loop
        for j in range(tstep - 1): # For each time step

            # Updating loading bar
            if(j%3000 == 0):
                time_progressive()

            # Update the planets position using Runge Kutta method
            planets[:, j+1] = runge_kutta_planets_orbit(planets, dt, j)

            if method == 3: 
                # Update the planets position using Explicit Eulero method
                planets_eulero[:, j+1] = explicit_eulero_planets_orbit(planets, dt, j)

                # Updating time vector
                if j == 0:
                    time.append(0)
                else:
                    time.append(time[j-1] + dt)

        # Plotting the difference for each planet in function of time
        if method == 2: 

            plt.style.use('dark_background')
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(projection='3d')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=None, hspace=None)
            ax.set_xlim3d([-25*ua, 25*ua])
            ax.set_ylim3d([-25*ua, 25*ua])
            ax.set_zlim3d([-25*ua, 25*ua])
            for i in range(9):
                ax.plot(planets[i, :len(planets[i, :, 1])-1, 0], planets[i, :len(planets[i, :, 1])-1, 1], planets[i, :len(planets[i, :, 1])-1, 2], color=colors[i], label=planet_names[i], alpha=0.7)
            ax.set_title('3D planets orbit (Runge-Kutta)', color='white')
            ax.legend(loc='upper left', fontsize=8)
            disconnect_zoom = zoom_factory(ax)
            pan_handler = panhandler(fig)

            plt.show()

        if method == 3:
            # Computing the difference between the two methods for each planet
            planets_diff = np.linalg.norm(planets[:, :len(time)] - planets_eulero[:, :len(time)], axis=2)  # shape (8, N)

            plt.figure(figsize=(10,6))
            for i in range(9):
                plt.plot(time[1:], planets_diff[i,1:], color=colors[i], label=planet_names[i])
            plt.yscale('log')
            plt.xlabel('Time [s]')
            plt.ylabel('Difference RK4 - Eulero [m]')
            plt.title('Difference between Runge-Kutta and Eulero for each planet')
            plt.legend()
            plt.grid(True, which='both', ls='--', alpha=0.5)
            plt.tight_layout()

            plt.show()
        
if __name__ == "__main__":
    main()