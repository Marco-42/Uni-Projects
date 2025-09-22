import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants
from mpl_interactions import ioff, panhandler, zoom_factory


#voyager - sole - terra - giove - saturno - urano - nettuno
#massa = [721.9, 1.988410e+30, 5.97219e+24, 1.89819e+27, 5.6834e+26, 8.681e+25, 1.024e+26] #only earth
massa = [721.9, 1.988410e+30, 6.04568e+24, 1.89819e+27, 5.6834e+26, 8.681e+25, 1.024e+26] #moon-earth system
#terra - giove - saturno - urano - nettuno
inclinazione = [0, 1.3, 2.5, 0.8, 1.8]
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
    a = y[1, :3]-r #giove
    b = y[0, :3]-r #terra
    c = y[2, :3]-r #saturno
    f[:3] = voy[3:]
    f[3:] = -massa[1]*constants.G*r/((np.linalg.norm(r))**3)+massa[2]*constants.G*b/((np.linalg.norm(b))**3)+massa[3]*constants.G*a/((np.linalg.norm(a))**3)+massa[4]*constants.G*c/((np.linalg.norm(c))**3)
    return f

def main(): 
#   (rx, ry, rz, vx, vy, vz)  per 5 pianeti nell'ordine(terra - giove - saturno - urano - nettuno)
#   Si considerano le posizioni di partenza all'afelio dell'orbita
    y = np.zeros((5, tstep,6), dtype = np.float64)
    planets = np.zeros((5, tstep, 3), dtype = np.float64)
    voy = np.zeros((tstep, 6), dtype = np.float64)
    t = []

   # y[0, 0] = [1.318979846056757e+11,-7.402099973641272e+10, -3.356075921379030e+06, 1.408222185406100e+04, 2.586813426342016e+04, 8.636839086921810e-01] #only earth
    y[0, 0] = [1.318973605817352e+11,-7.402541961170338e+10, -2.990768063254654e+06, 1.409502254263077e+04, 2.586644144636632e+04, 1.338944780846063] #earth-moon system
    y[1, 0] = [1.198087911578624e+11,  7.524367175268956e+11, -5.778152900470793e+09, -1.306946748845915e+04, 2.666064436903565e+03, 2.818154651206821e+02] #--> body center
   # y[1, 0] = [1.198086988073409e+11,  7.524366536310722e+11, -5.778156157933772e+09, -1.306788551193488e+04, 2.665818848548563e+03, 2.818263722904029e+02]
    y[2, 0] = [-1.068579746720888e+12, 8.625547334948143e+11, 2.738411795566380e+10, -6.600451849964506e+03,-7.542260028657253e+03, 3.940657058105259e+02] #--> body center
   # y[2, 0] = [-1.068579689711188e+12, 8.625544828858907e+11, 2.738424381595993e+10, -6.599176072432937e+03,-7.542143269092019e+03, 3.938832291134009e+02]
    y[3, 0] = [-2.082828979672745e+12, -1.842427428590381e+12, 2.018368216005743e+10, 4.461354281666044e+03, -5.419034890417917e+03, -7.804523385425410e+01]
    y[4, 0] = [-1.131562025778750e+12, -4.386564954458217e+12, 1.163931770114503e+11, 5.228313841361846e+03, -1.327310422299496e+03, -9.311906046824170e+01]

    voy[0] = [1.325250317089844e+11, -7.159364107513539e+10, 8.497599672614373e+08, 1.644287509793228e+04, 3.517556258110445e+04,  3.256293961011894e+03]


    dt = temp_orb(np.sqrt(y[2, 0, 0]**2 + y[2, 0, 1]**2 + y[2, 0, 2]**2))/(tstep*3)
    print(dt)
    for k in range(5):
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
    jupiter, = ax.plot([], [], [],'-', markersize=9, color = 'red')
    jupiterdot, = ax.plot([], [], [],'.', markersize=9, color = 'red')
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
        jupiter.set_data(planets[1, :i, 0], planets[1, :i, 1])
        jupiter.set_3d_properties(planets[1, :i, 2])
        jupiterdot.set_data([planets[1, i, 0]], [planets[1, i, 1]])
        jupiterdot.set_3d_properties([planets[1, i, 2]])
        saturn.set_data(planets[2, :i, 0], planets[2, :i, 1])
        saturn.set_3d_properties(planets[2, :i, 2])
        saturndot.set_data([planets[2, i, 0]], [planets[2, i, 1]])
        saturndot.set_3d_properties([planets[2, i, 2]])
        uranus.set_data(planets[3, :i, 0], planets[3, :i, 1])
        uranus.set_3d_properties(planets[3, :i, 2])
        uranusdot.set_data([planets[3, i, 0]], [planets[3, i, 1]])
        uranusdot.set_3d_properties([planets[3, i, 2]])
        neptune.set_data(planets[4, :i, 0], planets[4, :i, 1])
        neptune.set_3d_properties(planets[4, :i, 2])
        neptunedot.set_data([planets[4, i, 0]], [planets[4, i, 1]])
        neptunedot.set_3d_properties([planets[4, i, 2]])
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