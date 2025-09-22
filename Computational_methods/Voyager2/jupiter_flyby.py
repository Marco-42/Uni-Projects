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
mass = np.array([814.48, 1.89858e+27, 1.4819e+23, 1.0759e+23, 8.9319e+22, 4.7998e+22, 1.98841e+30], dtype=np.float64)
ua = 1.496e+11 # Astronomical unit (non più usata per conversione)
G = constants.G 
count = 0
idx_found = np.zeros(4, dtype=np.float64)
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
    y = np.zeros((6, tstep,6), dtype = np.float64)

    # DATA ORIGIN: 1979-05-28 00:00:00 - https://ssd.jpl.nasa.gov/horizons/
    y[0, 0] = [1.194937010862643e+04, -8.738727107864237e+04, -2.566485604696652e+03, 8.846303824873099e-01, -7.163865887574347e-02, 8.544789440473610e-03] # Jupiter
    y[1, 0] = [1.032394759322124e+09, 2.875644256047898e+08, 2.123968056205666e+07, -2.910891335089839e+03, 1.046238548713686e+04, 3.213811605692785e+02] # Ganymede
    y[2, 0] = [-1.631591786464455e+09, 9.666654100805386e+08, 1.206893691797130e+07, -4.158385802638955e+03, -6.995725118931221e+03, -2.898877490371148e+02] # Callisto
    y[3, 0] = [-3.591970193734830e+08, 2.193541019125491e+08, 2.570967896674090e+06, -8.997061048834632e+03, -1.484469080029191e+04, -6.527974360111948e+02] # Io
    y[4, 0] = [6.658239230115716e+08, -6.217121201697216e+06, 4.107326551288272e+06, 5.936649705614654e+01, 1.383836088281645e+04, 5.343751784140168e+02] # Europa
    y[5, 0] = [5.545845926469857e+11, -5.701369464228733e+11, -1.007921204636288e+10, 9.528926732129548e+03, 8.513565359170738e+03, -2.484972593365398e+02] # Sun

    # Setting the initial position and velocity for the Voyager 2 spacecraft
    spacecraft[0] = [4.009661609804929e+09, -3.285045464505006e+10, 3.149595461333949e+09, -5.620548467401283e+02, 8.077612952628357e+03, -8.006082852800689e+02] # spacecraft
    
    # Initial simulation date
    start_date = datetime(1979, 5, 28, 0, 0, 0)

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

    y_return = np.zeros((y.shape[0], 6), dtype = np.float64)

    # Applying Runge - Kutta method to solve the planet's differential equation(for each planet)
    for k in range(y.shape[0]):
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

    y_return = np.zeros((y.shape[0], 6), dtype = np.float64)

    # Applying Explicit Eulero method to solve the planet's differential equation(for each planet)
    for k in range(y.shape[0]):
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

    ax.set_title("Voyager 2 Trajectory in the Jupiter System", pad=20, fontsize=14)
    ax.set_xlim3d([-ua/70, ua/70])
    ax.set_ylim3d([-ua/70, ua/70])
    ax.set_zlim3d([-ua/70, ua/70])

    jupiter, = ax.plot([], [], [],'.', markersize=11, color = 'peru')
    jupiterdot, = ax.plot([], [], [],'.', markersize=11, color = 'peru')
    ganymede, = ax.plot([], [], [],'-', markersize=9, color = 'gray')
    ganymededot, = ax.plot([], [], [],'.', markersize=9, color = 'gray')
    callisto, = ax.plot([], [], [],'-', markersize=9, color = 'goldenrod')
    callistodot, = ax.plot([], [], [],'.', markersize=9, color = 'goldenrod')
    io, = ax.plot([], [], [],'-', markersize=9, color = 'red')
    iodot, = ax.plot([], [], [],'.', markersize=9, color = 'red')
    europa, = ax.plot([], [], [],'-', markersize=9, color = 'lightblue')
    europadot, = ax.plot([], [], [],'.', markersize=9, color = 'lightblue')

    voyager2, = ax.plot([],[],[],'-', markersize=9, color = 'white')
    voyager2dot, = ax.plot([],[],[],'.', markersize=9, color = 'white')
    
    def animate(frame):
        step = 300
        if((frame > idx_found[0] and frame < idx_found[1]) 
        or (frame > idx_found[2] and frame < idx_found[3])):
            i = frame * 10000
        else: 
            i = frame * step
        # Selecting only valid frames
        i = min(i, planets.shape[1]-1)
        trail = 3000  # planet trail length

        jupiterdot.set_data([planets[0, i, 0]], [planets[0, i, 1]])
        jupiterdot.set_3d_properties([planets[0, i, 2]])
        ganymede.set_data(planets[1, max(0,i-trail):i, 0], planets[1, max(0,i-trail):i, 1])
        ganymede.set_3d_properties(planets[1, max(0,i-trail):i, 2])
        ganymededot.set_data([planets[1, i, 0]], [planets[1, i, 1]])
        ganymededot.set_3d_properties([planets[1, i, 2]])
        callisto.set_data(planets[2, max(0,i-trail):i, 0], planets[2, max(0,i-trail):i, 1])
        callisto.set_3d_properties(planets[2, max(0,i-trail):i, 2])
        callistodot.set_data([planets[2, i, 0]], [planets[2, i, 1]])
        callistodot.set_3d_properties([planets[2, i, 2]])
        io.set_data(planets[3, max(0,i-trail):i, 0], planets[3, max(0,i-trail):i, 1])
        io.set_3d_properties(planets[3, max(0,i-trail):i, 2])
        iodot.set_data([planets[3, i, 0]], [planets[3, i, 1]])
        iodot.set_3d_properties([planets[3, i, 2]])
        europa.set_data(planets[4, max(0,i-trail):i, 0], planets[4, max(0,i-trail):i, 1])
        europa.set_3d_properties(planets[4, max(0,i-trail):i, 2])
        europadot.set_data([planets[4, i, 0]], [planets[4, i, 1]])
        europadot.set_3d_properties([planets[4, i, 2]])
        voyager2.set_data(voy[max(0,i-20000):i, 0], voy[max(0,i-20000):i, 1])
        voyager2.set_3d_properties(voy[max(0,i-20000):i, 2])
        voyager2dot.set_data([voy[i, 0]], [voy[i, 1]])
        voyager2dot.set_3d_properties([voy[i, 2]])
        return jupiter, ganymede, callisto, io, europa, jupiterdot, ganymededot, callistodot, iodot, europadot, voyager2, voyager2dot
        
    anim = animation.FuncAnimation(fig, animate, repeat=True, frames=tstep-2, interval=1, blit=False)
    disconnect_zoom = zoom_factory(ax)
    pan_handler = panhandler(fig)

    # anim.save('voyager2_jupiter_system.gif', writer='pillow', fps=60)

    plt.show()


# ============== MAIN ====================================
def main():

    # Method selector
    method_type = {1: runge_kutta, 2: explicit_eulero_planets_orbit, 3: "comparison"}
    
    tstep = 350000 # Total time steps to compute
    
    time = [] # Time vector
    H = np.zeros(tstep) # Total energy list
    K = np.zeros(tstep) # Kinetic energy list

    # Plotting variables
    planet_names = ["Jupiter", "Ganymede", "Callisto", "Io", "Europa", "Sun"]
    colors = ['peru', 'gray', 'goldenrod', 'red', 'lightblue', 'yellow']

    # TCM check flags
    time_check_done = np.array([True, True, False, False], dtype=bool)
    
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
        
        # Setting delta time
        dt = 15
        
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
                voy[i, 3:] += (-25.30560423460293, 25.938101730890395, 0.4606972643019791)
                mass[0] = 798.05
            if not time_check_done[3] and TIME > datetime(1979, 6, 28): # TCM 4
                time_check_done[3] = True
                voy[i, 3:] += (-10.823419538086299, 47.08880790957682, -8.854442214513924)
                mass[0] = 776.30

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
        distance_giove = np.linalg.norm(planets[0, :, :3] - voy[:, :3], axis=1)
        min_dist = np.min(distance_giove)
        min_index = np.argmin(distance_giove)
        print(f"Minimum distance from jupiter: {min_dist:.2e} m at step {min_index}")

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
        tcm_dates = [datetime(1979, 5, 30), datetime(1979, 6, 28)]
        tcm_labels = ["TCM 3", "TCM 4"]
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
        dt = 60 # delta time is setted constant 

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
            # ax.set_xlim3d([ua/70, ua/70])
            # ax.set_ylim3d([ua/70, ua/70])
            # ax.set_zlim3d([ua/70, ua/70])
            for i in range(5):
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
            for i in range(6):
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
