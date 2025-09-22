import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants
from mpl_interactions import ioff, panhandler, zoom_factory


#voyager - sole - terra - marte - giove - saturno - urano - nettuno
#massa = [721.9, 1.988410e+30, 5.97219e+24, 1.89819e+27, 5.6834e+26, 8.681e+25, 1.024e+26] #only earth
massa = [721.9, 1.988410e+30, 6.04568e+24, 6.4171e+23, 1.89819e+27, 5.6834e+26, 8.681e+25, 1.024e+26] #moon-earth system

ua = 1.496e+11
tstep = 14000
delta = float(0)

def vel_orb(dist):
    """Restituisce la velocità tangenziale in un punto dell'orbita attorno al sole - Parametro: distanza"""
    return  np.sqrt(massa[1]*constants.G/dist)

def temp_orb(dist):
    """Restituisce il periodo orbitale di un orbita circolare attorno al sole - Parametro: distanza"""
    return 2*np.pi*np.sqrt(dist**3/(massa[1]*constants.G))

def func(y, M): 
    """"Restituisce l'accelerazione - Parametri: vettore(posizione, velocità) in 3d e massa generatrice del campo"""
    f = np.zeros((6), dtype = np.float64)
    r = y[:3]
    f[:3] = y[3:]
    f[3:] = -M*constants.G*r/((np.linalg.norm(r))**3)
    return f

def func_voy(y, voy):
    f = np.zeros((6), dtype = np.float64)
    r = voy[:3]
    a = y[2, :3]-r #giove
    b = y[0, :3]-r #terra
    c = y[3, :3]-r #saturno
    d = y[1, :3]-r #marte
    e = y[4, :3]-r #urano
    g = y[5, :3]-r #nettuno
    f[:3] = voy[3:]
    f[3:] = -massa[1]*constants.G*r/((np.linalg.norm(r))**3)+massa[2]*constants.G*b/((np.linalg.norm(b))**3)+massa[4]*constants.G*a/((np.linalg.norm(a))**3)+massa[5]*constants.G*c/((np.linalg.norm(c))**3)+massa[3]*constants.G*d/((np.linalg.norm(d))**3)+massa[6]*constants.G*e/((np.linalg.norm(e))**3)+massa[7]*constants.G*g/((np.linalg.norm(g))**3)
    return f

def main(): 
#   (rx, ry, rz, vx, vy, vz)  per 5 pianeti nell'ordine(terra - marte - giove - saturno - urano - nettuno)
#   Si considerano le posizioni di partenza all'afelio dell'orbita
    y = np.zeros((6, tstep,6), dtype = np.float64)
    planets = np.zeros((6, tstep, 3), dtype = np.float64)
    voy = np.zeros((tstep, 6), dtype = np.float64)
    t = []
#   1977-08-20 15:32:32.1830
    y[0, 0] = [1.280754507981821e+11,-8.063223054221845e+10, -3.333806160446256e+06, 1.538647658886928e+04, 2.509818990291462e+04, 1.307230850967400] #earth-moon system
    #y[0, 0] = [1.293886667715476e+11,-7.844952451639074e+10, -3.463729096457362e+06, 1.495022350622355e+04, 2.536866680617683e+04, 0.4113356176898009] #only earth
    y[1, 0] = [1.533351483204997e+11, 1.561280217126437e+11, -5.071557534510717e+08, -1.635890858940308e+04, 1.903790398788446e+04, 8.015523324009024e+02] #--> body center
    y[2, 0] = [1.209384878347845e+11, 7.522053967032398e+11, -5.802508836417437e+09, -1.306633836957094e+04, 2.685599415908918e+03, 2.816650009811117e+02] #--> body center
    y[3, 0] = [-1.068008945052454e+12, 8.632065821317078e+11, 2.735004645263809e+10, -6.605184960633535e+03,-7.538418208606297e+03, 3.941868840101117e+02] #--> body center
    y[4, 0] = [-2.082379120616198e+12, -1.842726134821691e+12, 2.017901238749790e+10, 4.449715524238801e+03, -5.416787343682951e+03, -7.768943705667453e+01]
    y[5, 0] = [-1.130976949260969e+12, -4.386141420888277e+12, 1.163858455813677e+11, 5.214689122432943e+03, -1.325966592651084e+03, -9.277060168212825e+01]

    voy[0] = [1.280860880954407e+11, -8.063043539493687e+10, -1.355532101832330e+06, 2.168074852106179e+04, 3.697930846868999e+04, 5.268488333819866e+03]
              

    dt = temp_orb(np.sqrt(y[3, 0, 0]**2 + y[3, 0, 1]**2 + y[3, 0, 2]**2))/(tstep*3)
  
    for k in range(6):
        for i in range(tstep -1):
            Y1 = np.array(y[k, i])
            Y2 = Y1 +func(Y1, massa[1])*dt/2
            Y3 = Y1 + func(Y2, massa[1])*dt/2
            Y4 = Y1 + func(Y3, massa[1])*dt
            y[k, i+1] = Y1+(func(Y1, massa[1]) + 2*func(Y2, massa[1]) + 2*func(Y3, massa[1])+func(Y4, massa[1]))*dt/6
            planets[k, i, 0] = y[k, i, 0]
            planets[k, i, 1] = y[k, i, 1]
            planets[k, i, 2] =  y[k, i, 2]
            
            if(k == 0):
                delta = dt
                t.append(i*dt)

    for i in range(tstep-1):
        Y1 = np.array(voy[i])
        Y2 = Y1 +func_voy(planets[:, i, :], Y1)*dt/2
        Y3 = Y1 + func_voy(planets[:, i, :], Y2)*dt/2
        Y4 = Y1 + func_voy(planets[:, i, :], Y3)*dt
        voy[i+1] = Y1+(func_voy(planets[:, i, :], Y1) + 2*func_voy(planets[:, i, :], Y2) + 2*func_voy(planets[:, i, :], Y3)+func_voy(planets[:, i, :], Y4))*dt/6        

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
   # ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=None, hspace=None)
    ax.set_xlim3d([-5*ua, 5*ua])
    ax.set_ylim3d([-5*ua, 5*ua])
    ax.set_zlim3d([-5*ua, 5*ua])
    
    sun, = ax.plot([], [], [],'.', markersize=11, color = 'yellow')
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
    
    def animate(i):
        i = i*10
        sun.set_data([0],[0])
        sun.set_3d_properties([0])
        earth.set_data(planets[0, :i, 0], planets[0, :i, 1])
        earth.set_3d_properties(planets[0, :i, 2])
        earthdot.set_data([planets[0, i, 0]], [planets[0, i, 1]])
        earthdot.set_3d_properties([planets[0, i, 2]])
        mars.set_data(planets[1, :i, 0], planets[1, :i, 1])
        mars.set_3d_properties(planets[1, :i, 2])
        marsdot.set_data([planets[1, i, 0]], [planets[1, i, 1]])
        marsdot.set_3d_properties([planets[1, i, 2]])
        jupiter.set_data(planets[2, :i, 0], planets[2, :i, 1])
        jupiter.set_3d_properties(planets[2, :i, 2])
        jupiterdot.set_data([planets[2, i, 0]], [planets[2, i, 1]])
        jupiterdot.set_3d_properties([planets[2, i, 2]])
        saturn.set_data(planets[3, :i, 0], planets[3, :i, 1])
        saturn.set_3d_properties(planets[3, :i, 2])
        saturndot.set_data([planets[3, i, 0]], [planets[3, i, 1]])
        saturndot.set_3d_properties([planets[3, i, 2]])
        uranus.set_data(planets[4, :i, 0], planets[4, :i, 1])
        uranus.set_3d_properties(planets[4, :i, 2])
        uranusdot.set_data([planets[4, i, 0]], [planets[4, i, 1]])
        uranusdot.set_3d_properties([planets[4, i, 2]])
        neptune.set_data(planets[5, :i, 0], planets[5, :i, 1])
        neptune.set_3d_properties(planets[5, :i, 2])
        neptunedot.set_data([planets[5, i, 0]], [planets[5, i, 1]])
        neptunedot.set_3d_properties([planets[5, i, 2]])
        voyager2.set_data(voy[:i, 0], voy[:i, 1])
        voyager2.set_3d_properties(voy[:i, 2])
        voyager2dot.set_data([voy[i, 0]], [voy[i, 1]])
        voyager2dot.set_3d_properties([voy[i, 2]])
        return earth, jupiter, saturn, uranus, neptune, earthdot, jupiterdot, saturndot, uranusdot, neptunedot, sun, voyager2

    anim = animation.FuncAnimation(fig, animate, repeat=True, frames = tstep, interval=1)
    #anim.save('orbita.gif', writer='Pillow', fps=30)

    disconnect_zoom = zoom_factory(ax)
    pan_handler = panhandler(fig)
    plt.show()

if __name__ == "__main__":
    main()