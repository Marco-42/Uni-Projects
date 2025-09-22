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
mass = np.array([721.9, 1.98841e+30, 3.301e+23, 4.8673e+24, 6.04568e+24, 6.4169e+23, 1.89858e+27, 5.6847e+26, 8.6811e+25, 1.02409e+26], dtype=np.float64)

ua = 1.496e+11 # Astronomical unit (non più usata per conversione)
tstep = 350000 # Total time steps to compute
G = constants.G 
count = 0
idx_found = np.array(4, dtype=np.float64)
idx_found_flag = False

# ============== USEFULL FUNCTIONS ==============

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
def set_initial_par():
    """Return planets_vector - spacecraft_vector - Initial_simulation_time(datetime)"""

    # Defining a vector for each planet to simulate[ 8 objects - tstep point to simulate - 6 parameters]
    y = np.zeros((8, tstep,6), dtype = np.float64)

    # Defining a vector for the Voyager 2 spacecraft
    spacecraft = np.zeros((tstep, 6), dtype = np.float64)

    # DATA ORIGIN: 1977-08-21 15:32:00 - https://ssd.jpl.nasa.gov/horizons/
    # Setting the initial positions and velocities(3D) for the planets
    y[0, 0] = [2.420431906541956e+10, -6.187703200755139e+10, -7.275833822790146e+09, 3.561035108196501e+04, 2.021400023848995e+04, -1.619891368446982e+03] # Mercury
    y[0, 0] = [2.420431906541956e+10, -6.187703200755139e+10, -7.275833822790146e+09, 3.561035108196501e+04, 2.021400023848995e+04, -1.619891368446982e+03] # Mercury
    y[1, 0] = [6.149911097220387e+10, 8.878362738242684e+10, -2.342953100874681e+09, -2.890029159540279e+04, 1.978783233145617e+04, 1.937932804493329e+03] # Venus
    y[2, 0] = [1.293859926988265e+11,-7.845314259374106e+10, -3.220428412642330e+06, 1.496068030016293e+04, 2.536128208013420e+04, 1.318170291993326] # Earth-Moon system
    y[3, 0] = [1.547412814107106e+11, 1.544757871141991e+11, -5.763847928677648e+08, -1.619014127098390e+04, 1.920803665174512e+04, 8.009588389741307e+02] # Mars
    y[4, 0] = [1.220672832895152e+11, 7.519725175543975e+11, -5.826838172204912e+09, -1.306318105696554e+04, 2.705121240474793e+03, 2.815139577185036e+02] # Jupiter system
    y[5, 0] = [-1.067438052795392e+12, 8.638577354786228e+11, 2.731598347907853e+10, -6.609913212256253e+03,-7.534575413901138e+03, 3.943078502480204e+02] # Saturn system
    y[6, 0] = [-2.082763528219628e+12, -1.842258082788988e+12, 2.018572429865563e+10, 4.448608598616712e+03, -5.417750423360054e+03, -7.767887655780248e+01] # Uranus
    y[7, 0] = [-1.131427492567988e+12, -4.386026834882401e+12, 1.163938603499129e+11, 5.214554094371723e+03, -1.326487255368110e+03, -9.275644823835677e+01] # Neptune

    # Setting the initial position and velocity for the Voyager 2 spacecraft
    spacecraft[0] = [1.296064082167793e+11, -7.763260650298794e+10, 2.860000012703910e+08, 1.732600826660910e+04, 3.470373361262651e+04, 3.269550717841025e+03]
    
    # Initial simulation date
    start_date = datetime(1977, 8, 21, 15, 32, 0)

    # Return the two vectors 
    return y, spacecraft, start_date

# ============== SIMULATION HELP FUNCTIONS ==============

# Function to compute planet's velocity and acceleration
@njit # To increase performance
def func(y, x, planet):
    """All planets vector - Planet index - Single planet vector"""

    f = np.zeros(6, dtype=np.float64)
    r = planet[:3] # Single planet position
    f[:3] = planet[3:]
    f[3:] = -mass[1]*G*r/((np.linalg.norm(r))**3) # Gravitational force only by the sun

    # Compute planets acceleration
    acceleration = np.zeros(3, dtype=np.float64)

    # boolean vector True if idx[k] != x
    idx = np.arange(y.shape[0]) != x

    # Getting planets positions and masses by boolean indexing
    r2 = y[np.arange(y.shape[0])[idx], :3]
    m2 = mass[np.arange(y.shape[0])[idx] + 2]

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
    f[3:] += G * acceleration

    return f

# Function to compute spacecraft velocity and acceleration
@njit # To increase performance
def func_spacecraft(y, spacecraft):
    "All planets vector - Spacecraft vector"

    f = np.zeros(6, dtype=np.float64)
    r = spacecraft[:3] # Spacecraft position
    f[:3] = spacecraft[3:]

    # Compute the spacecraft's acceleration only by the sun
    f[3:] = -mass[1]*G*r/((np.linalg.norm(r))**3)

    # Including the gravitational interaction from other planets
    # Getting planets positions and masses
    r2 = y[:, :3]
    m2 = mass[2:]

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

# Function to compute the delta_time between each time step
def get_delta_time(y, spacecraft, reference, i):
    """Planets vector - Spacecraft vector - Reference time - increment number"""
    global idx_found
    global idx_found_flag

    # Computing distances between selected planets and the spacecraft
    dist_jupiter = np.linalg.norm(y[4, i, :3] - spacecraft[i, :3])
    dist_saturn = np.linalg.norm(y[5, i, :3] - spacecraft[i, :3])
    dist_earth = np.linalg.norm(y[2, i, :3] - spacecraft[i, :3])

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

# ============== RUNGE KUTTA ==============

# Function to execute the Runge - Kutta method to solve the differential equations
@njit # To increase performance
def runge_kutta(y, spacecraft, dt, i):
    """Runge-Kutta method: planets vector - spacecraft vector - delta_time - for index"""

    y_return = np.zeros((8, 6), dtype = np.float64)

    # Applying Runge - Kutta method to solve the planet's differential equation(for each planet)
    for k in range(8):
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

# ============== MAIN ==============
def main():

    # Get the initial parameters
    planets, voy, TIME = set_initial_par()
    
    # Getting the reference time for delta time (in metri)
    reference_time = temp_orb(np.linalg.norm(planets[5, 0, :3]))
    print("0 %                                                                           100 %")
    # Flag per eseguire l'if solo una volta
    time_check_done = np.array([False, False, False, False, False, False, False], dtype=bool)
    
    for i in range(tstep - 1):

        if(i%50000 == 0):
            time_progressive()

        # TCM maneuver during earth - jupitar navigation
        if not time_check_done[0] and TIME > datetime(1977, 10, 12): # TCM 1
            time_check_done[0] = True
            #voy[i+1, 3:] += 0.85*voy[i+1, 3:]/10000 # 0.0085 % of velocity
            voy[i, 3:] += (-0.3524034958877209, -1.3341693211401662, -0.11479562983218727)
            # print(np.linalg.norm((-0.3524034958877209, -1.3341693211401662, -0.11479562983218727))*100/np.linalg.norm(voy[i, 3:]))
        if not time_check_done[1] and TIME > datetime(1978, 6, 4): # TCM 2
            time_check_done[1] = True
            #voy[i+1, 3:] += 0.9*voy[i+1, 3:]/10000 # 0.009 % of velocity
            voy[i, 3:] += (3.5627933752309104, 16.149974912401888, 2.7027908147108093)
            # print(np.linalg.norm((3.5627933752309104, 16.149974912401888, 2.7027908147108093))*100/np.linalg.norm(voy[i, 3:]))
        # TCM maneuver during jupiter pre-approach
        if not time_check_done[2] and TIME > datetime(1979, 5, 30): # TCM 3
            time_check_done[2] = True
            #print(np.linalg.norm(voy[i+1, 3:])/100)
            # voy[i+1, 3:] += -np.linalg.norm(voy[i+1, 3:])*np.cross(voy[i+1, :3]/np.linalg.norm(voy[i+1, :3]), voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:]))/10000
            voy[i, 3:] += (-3.7600933801904546, 6.773486815637552, -2.3559608762223627)
            # print(np.linalg.norm((-3.0019597100405804, -0.5031942278084234, -0.6820768054122128))*100/np.linalg.norm(voy[i, 3:]))
        if not time_check_done[3] and TIME > datetime(1979, 6, 28): # TCM 4
            time_check_done[3] = True
            #print(np.linalg.norm(voy[i+1, 3:])/100)
            # voy[i+1, 3:] += -np.linalg.norm(voy[i+1, 3:])*np.cross(voy[i+1, :3]/np.linalg.norm(voy[i+1, :3]), voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:]))/10000
            voy[i, 3:] += (-15.148583227244217, 70.26566070973384, -11.677958319829983)
        # TCM maneuver during jupiter post-approach
        # if TIME >= datetime(1979, 7, 7, 23) and TIME <= datetime(1979, 7, 8):
        #     print(TIME)
        #     print(voy[i, 3:])

        # Compute the delta time for the current time step
        dt = get_delta_time(planets, voy, reference_time, i)
        #dt = 300

        planets[:, i+1], voy[i+1] = runge_kutta(planets, voy, dt, i)

        # Local update of time variable
        TIME = TIME + timedelta(seconds=dt)

        # if(np.linalg.norm(planets[4, i+1, :3] - voy[i+1, :3]) > np.linalg.norm(planets[4, i, :3] - voy[i, :3])):
        #    print(TIME)
        #    break 

        # TCM maneuver during earth - jupitar navigation
        # if not time_check_done[0] and TIME > datetime(1977, 10, 11): # TCM 1
        #     time_check_done[0] = True
        #     #voy[i+1, 3:] += 0.85*voy[i+1, 3:]/10000 # 0.0085 % of velocity
        #     voy[i+1, 3:] += (0.12663415121767618, 1.200525196064333, 0.09610910706566352)
        #     print(np.linalg.norm((0.12663415121767618, 1.200525196064333, 0.09610910706566352))*100/np.linalg.norm(voy[i+1, 3:]))
        # if not time_check_done[1] and TIME > datetime(1978, 6, 3): # TCM 2
        #     time_check_done[1] = True
        #     #voy[i+1, 3:] += 0.9*voy[i+1, 3:]/10000 # 0.009 % of velocity
        #     voy[i+1, 3:] += (-3.155872215380156, -9.99033645999091, -2.2433295744724426)
        #     print(np.linalg.norm((-3.155872215380156, -9.99033645999091, -2.2433295744724426))*100/np.linalg.norm(voy[i+1, 3:]))
        # # if not time_check_done[2] and TIME > datetime(1979, 2, 10, 0, 0, 0): # TCM 3
        #     time_check_done[2] = True
        #     #print(np.linalg.norm(voy[i+1, 3:])/100)
        #     # voy[i+1, 3:] += -np.linalg.norm(voy[i+1, 3:])*np.cross(voy[i+1, :3]/np.linalg.norm(voy[i+1, :3]), voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:]))/10000
        #     voy[i+1, 3:] += (-1.31195471993205, -20.15985136999955, -3.3408491603209023)
        #     print(np.linalg.norm((-1.31195471993205, -20.15985136999955, -3.3408491603209023))*100/np.linalg.norm(voy[i+1, 3:]))

        # # TCM maneuver during jupiter post-approach(pre-approach TCM 3 and TCM 4 was only inclination corrections)
        # if not time_check_done[3] and TIME > datetime(1979, 7, 9, 23, 0, 0): # TCM 5
        #     time_check_done[3] = True
        #     #print(np.linalg.norm(voy[i+1, 3:])/100)
        #     # voy[i+1, 3:] += -np.linalg.norm(voy[i+1, 3:])*np.cross(voy[i+1, :3]/np.linalg.norm(voy[i+1, :3]), voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:]))/10000
        #     voy[i+1, 3:] += (0, 0, -2*np.linalg.norm(voy[i+1, 3:])/10000)

            
        # # TCM maneuver during jupiter - saturn navigation
        # if not time_check_done[4] and TIME > datetime(1981, 2, 26): # TCM 7
        #     time_check_done[4] = True
        #     #voy[i+1, 3:] += (0, 0, -0.574)
        #     voy[i+1, 3:] += -0.574*np.cross(voy[i+1, :3]/np.linalg.norm(voy[i+1, :3]), voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:]))  # Correzione normale
        #     #voy[i+1, 3:] += -0.574*voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:])
        # if not time_check_done[5] and TIME > datetime(1981, 7, 19): # TCM 7
        #     time_check_done[5] = True
        #     voy[i+1, 3:] += (0, 0, -2.352)
        #     voy[i+1, 3:] += -2.2352*np.cross(voy[i+1, :3]/np.linalg.norm(voy[i+1, :3]), voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:]))  # Correzione normale
        #     #voy[i+1, 3:] += -0.574*voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:])
        # if not time_check_done[6] and TIME > datetime(1981, 8, 18): # TCM 7
        #     time_check_done[6] = True
        #     voy[i+1, 3:] += (0, 0, -2.2352)
        #     voy[i+1, 3:] += -2.2352*np.cross(voy[i+1, :3]/np.linalg.norm(voy[i+1, :3]), voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:]))  # Correzione normale
        #     #voy[i+1, 3:] += -0.574*voy[i+1, 3:]/np.linalg.norm(voy[i+1, 3:])

        # if(i == 167583):
        #     print(TIME)

        # if not time_check_done[4] and TIME > datetime(1979, 7, 9):  # TCM 3-4-5
        #     time_check_done[4] = True
        #    # print(f"La simulazione ha superato l'11 ottobre 1979: {TIME}")
        #     print(vrn_to_xyz(-22.77, 0.31, 0, voy[i+1, :3], planets[4, i+1, :3], vel_orbitale=voy[i+1, 3:]))
        #     voy[i+1, 3:] += vrn_to_xyz(-22.77, 0.31, 0, voy[i+1, :3], planets[4, i+1, :3], vel_orbitale=voy[i+1, 3:])
    


    distance_giove = np.linalg.norm(planets[4, :, :3] - voy[:, :3], axis=1)
    min_dist = np.min(distance_giove)
    min_index = np.argmin(distance_giove)
    print(f"Distanza minima da Giove: {min_dist:.2e} m al passo {min_index}")

    distance_saturno = np.linalg.norm(planets[5, :, :3] - voy[:, :3], axis=1)
    min_dist_sat = np.min(distance_saturno)
    min_index_sat = np.argmin(distance_saturno)
    print(f"Distanza minima da Saturno: {min_dist_sat:.2e} m al passo {min_index_sat}")

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=None, hspace=None)
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
        step = 1000
        if((frame > idx_found[0] and frame < idx_found[1]) 
           or (frame > idx_found[2] and frame < idx_found[3])):
            i = frame * 3000
        else: 
            i = frame * step
        trail = 1000  # lunghezza della scia visibile per i pianeti
        sun.set_data([0],[0])
        sun.set_3d_properties([0])
        # Mostra solo una scia breve per ogni pianeta
        mercury.set_data(planets[0, max(0,i-trail):i, 0], planets[0, max(0,i-trail):i, 1])
        mercury.set_3d_properties(planets[0, max(0,i-trail):i, 2])
        mercurydot.set_data([planets[0, i, 0]], [planets[0, i, 1]])
        mercurydot.set_3d_properties([planets[0, i, 2]])
        venus.set_data(planets[1, max(0,i-trail):i, 0], planets[1, max(0,i-trail):i, 1])
        venus.set_3d_properties(planets[1, max(0,i-trail):i, 2])
        venusdot.set_data([planets[1, i, 0]], [planets[1, i, 1]])
        venusdot.set_3d_properties([planets[1, i, 2]])
        earth.set_data(planets[2, max(0,i-trail):i, 0], planets[2, max(0,i-trail):i, 1])
        earth.set_3d_properties(planets[2, max(0,i-trail):i, 2])
        earthdot.set_data([planets[2, i, 0]], [planets[2, i, 1]])
        earthdot.set_3d_properties([planets[2, i, 2]])
        mars.set_data(planets[3, max(0,i-trail):i, 0], planets[3, max(0,i-trail):i, 1])
        mars.set_3d_properties(planets[3, max(0,i-trail):i, 2])
        marsdot.set_data([planets[3, i, 0]], [planets[3, i, 1]])
        marsdot.set_3d_properties([planets[3, i, 2]])
        jupiter.set_data(planets[4, max(0,i-trail):i, 0], planets[4, max(0,i-trail):i, 1])
        jupiter.set_3d_properties(planets[4, max(0,i-trail):i, 2])
        jupiterdot.set_data([planets[4, i, 0]], [planets[4, i, 1]])
        jupiterdot.set_3d_properties([planets[4, i, 2]])
        saturn.set_data(planets[5, max(0,i-trail):i, 0], planets[5, max(0,i-trail):i, 1])
        saturn.set_3d_properties(planets[5, max(0,i-trail):i, 2])
        saturndot.set_data([planets[5, i, 0]], [planets[5, i, 1]])
        saturndot.set_3d_properties([planets[5, i, 2]])
        uranus.set_data(planets[6, max(0,i-trail):i, 0], planets[6, max(0,i-trail):i, 1])
        uranus.set_3d_properties(planets[6, max(0,i-trail):i, 2])
        uranusdot.set_data([planets[6, i, 0]], [planets[6, i, 1]])
        uranusdot.set_3d_properties([planets[6, i, 2]])
        neptune.set_data(planets[7, max(0,i-trail):i, 0], planets[7, max(0,i-trail):i, 1])
        neptune.set_3d_properties(planets[7, max(0,i-trail):i, 2])
        neptunedot.set_data([planets[7, i, 0]], [planets[7, i, 1]])
        neptunedot.set_3d_properties([planets[7, i, 2]])
        # La sonda mostra la traiettoria completa
        voyager2.set_data(voy[:i, 0], voy[:i, 1])
        voyager2.set_3d_properties(voy[:i, 2])
        voyager2dot.set_data([voy[i, 0]], [voy[i, 1]])
        voyager2dot.set_3d_properties([voy[i, 2]])
        return earth, jupiter, saturn, uranus, neptune, earthdot, jupiterdot, saturndot, uranusdot, neptunedot, sun, voyager2, mars, marsdot, venus, venusdot, mercury, mercurydot

    anim = animation.FuncAnimation(fig, animate, repeat=True, frames=tstep-2, interval=1)
    # anim.save('orbita.gif', writer='Pillow', fps=30)

    disconnect_zoom = zoom_factory(ax)
    pan_handler = panhandler(fig)
    plt.show()

if __name__ == "__main__":
    main()