import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants
from mpl_interactions import ioff, panhandler, zoom_factory


#voyager - sole - terra - marte - giove - saturno - urano - nettuno
#massa = [721.9, 1.988410e+30, 5.97219e+24, 1.89819e+27, 5.6834e+26, 8.681e+25, 1.024e+26] #only earth
massa = [721.9, 1.988410e+30, 6.04568e+24, 6.4171e+23, 1.89819e+27, 5.6834e+26, 8.681e+25, 1.024e+26, 4.8685e+24, 3.302e+23] #moon-earth system

ua = 1.496e+11
tstep = 7000
delta = float(0)

def vel_orb(dist):
    """Restituisce la velocità tangenziale in un punto dell'orbita attorno al sole - Parametro: distanza"""
    return  np.sqrt(massa[1]*constants.G/dist)

def temp_orb(dist):
    """Restituisce il periodo orbitale di un orbita circolare attorno al sole - Parametro: distanza"""
    return 2*np.pi*np.sqrt(dist**3/(massa[1]*constants.G))

def func(y, x, planet): 
    """"Restituisce l'accelerazione - Parametri: vettore(posizione, velocità) in 3d e massa generatrice del campo"""
    f = np.zeros((6), dtype = np.float64)
    r = planet[:3]
    f[:3] = planet[3:]
    f[3:] = -massa[1]*constants.G*r/((np.linalg.norm(r))**3)
    for i in range(8):
        if(i != x):
            f[3:] += massa[i+2]*constants.G*(y[i, :3]-r)/((np.linalg.norm((y[i, :3]-r)))**3)
    return f

def func2(y, M): 
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
    t = y[6, :3]-r #venere
    o = y[7, :3]-r #mercurio
    f[:3] = voy[3:]
    f[3:] = -massa[1]*constants.G*r/((np.linalg.norm(r))**3)+massa[2]*constants.G*b/((np.linalg.norm(b))**3)+massa[4]*constants.G*a/((np.linalg.norm(a))**3)+massa[5]*constants.G*c/((np.linalg.norm(c))**3)+massa[3]*constants.G*d/((np.linalg.norm(d))**3)+massa[6]*constants.G*e/((np.linalg.norm(e))**3)+massa[7]*constants.G*g/((np.linalg.norm(g))**3)+massa[8]*constants.G*t/((np.linalg.norm(t))**3)+massa[9]*constants.G*o/((np.linalg.norm(o))**3)
    return f

def main(): 
#   (rx, ry, rz, vx, vy, vz)  per 5 pianeti nell'ordine(terra - marte - giove - saturno - urano - nettuno)
#   Si considerano le posizioni di partenza all'afelio dell'orbita
    y = np.zeros((8, tstep,6), dtype = np.float64)
    planets = np.zeros((8, tstep, 3), dtype = np.float64)
    voy = np.zeros((tstep, 6), dtype = np.float64)
    t = []
#   1977-08-21 15:32:00
    y[0, 0] = [1.293859926988265e+11,-7.845314259374106e+10, -3.220428412642330e+06, 1.496068030016293e+04, 2.536128208013420e+04, 1.318170291993326] #earth-moon system
    y[1, 0] = [1.547412814107106e+11, 1.544757871141991e+11, -5.763847928677648e+08, -1.619014127098390e+04, 1.920803665174512e+04, 8.009588389741307e+02] #--> body center
    y[2, 0] = [1.220672832895152e+11, 7.519725175543975e+11, -5.826838172204912e+09, -1.306318105696554e+04, 2.705121240474793e+03, 2.815139577185036e+02] #--> body center
    y[3, 0] = [-1.067438052795392e+12, 8.638577354786228e+11, 2.731598347907853e+10, -6.609913212256253e+03,-7.534575413901138e+03, 3.943078502480204e+02] #--> body center
    y[4, 0] = [-2.082763528219628e+12, -1.842258082788988e+12, 2.018572429865563e+10, 4.448608598616712e+03, -5.417750423360054e+03, -7.767887655780248e+01]
    y[5, 0] = [-1.131427492567988e+12, -4.386026834882401e+12, 1.163938603499129e+11, 5.214554094371723e+03, -1.326487255368110e+03, -9.275644823835677e+01]
    y[6, 0] = [6.149911097220387e+10, 8.878362738242684e+10, -2.342953100874681e+09, -2.890029159540279e+04, 1.978783233145617e+04, 1.937932804493329e+03]
    y[7, 0] = [2.420431906541956e+10, -6.187703200755139e+10, -7.275833822790146e+09, 3.561035108196501e+04, 2.021400023848995e+04, -1.619891368446982e+03]
    voy[0] = [1.296064082167793e+11, -7.763260650298794e+10, 2.860000012703910e+08, 1.732600826660910e+04, 3.470373361262651e+04, 3.269550717841025e+03]
              
    a = 1
    dt = temp_orb(np.sqrt(y[a, 0, 0]**2 + y[a, 0, 1]**2 + y[a, 0, 2]**2))/(tstep-1)
  
    for k in range(8):
        for i in range(tstep -1):
            Y1 = np.array(y[k, i,:])
            Y2 = Y1 +func(y[:, i, :], k, Y1)*dt/2
            Y3 = Y1 + func(y[:, i, :], k, Y2)*dt/2
            Y4 = Y1 + func(y[:, i, :], k, Y3)*dt
            y[k, i+1] = Y1+(func(y[:, i, :], k, Y1) + 2*func(y[:, i, :], k, Y2) + 2*func(y[:, i, :], k, Y3)+func(y[:, i, :], k, Y4))*dt/6
            planets[k, i, 0] = y[k, i, 0]
            planets[k, i, 1] = y[k, i, 1]
            planets[k, i, 2] =  y[k, i, 2]

    satu = np.zeros((8, tstep, 3), dtype = np.float64)
    
    k = a
    for i in range(tstep -1):
            Y1 = np.array(y[k, i, :])
            Y2 = Y1 +func2(Y1, massa[1])*dt/2
            Y3 = Y1 + func2(Y2, massa[1])*dt/2
            Y4 = Y1 + func2(Y3, massa[1])*dt
            y[k, i+1] = Y1+(func2(Y1, massa[1]) + 2*func2(Y2, massa[1]) + 2*func2(Y3, massa[1])+func2(Y4, massa[1]))*dt/6
            satu[k, i, 0] = y[k, i, 0]
            satu[k, i, 1] = y[k, i, 1]
            satu[k, i, 2] =  y[k, i, 2]
        
    """
    for i in range(tstep-1):
        Y1 = np.array(voy[i])
        Y2 = Y1 +func_voy(planets[:, i, :], Y1)*dt/2
        Y3 = Y1 + func_voy(planets[:, i, :], Y2)*dt/2
        Y4 = Y1 + func_voy(planets[:, i, :], Y3)*dt
        voy[i+1] = Y1+(func_voy(planets[:, i, :], Y1) + 2*func_voy(planets[:, i, :], Y2) + 2*func_voy(planets[:, i, :], Y3)+func_voy(planets[:, i, :], Y4))*dt/6        
    """

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
   # ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=None, hspace=None)
    ax.plot(planets[k, :, 0], planets[k, :, 1], planets[k, :, 2],label='saturno tutto', linestyle='--', color='red')
    ax.plot(satu[k, :, 0], satu[k, :, 1], satu[k, :, 2],label='saturno', linestyle='--', color='white')
    """
    ax.set_xlim3d([-5*ua, 5*ua])
    ax.set_ylim3d([-5*ua, 5*ua])
    ax.set_zlim3d([-5*ua, 5*ua])
    
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
    
    def animate(i):
        i = i*10
        sun.set_data([0],[0])
        sun.set_3d_properties([0])
        mercury.set_data(planets[7, :i, 0], planets[7, :i, 1])
        mercury.set_3d_properties(planets[7, :i, 2])
        mercurydot.set_data([planets[7, i, 0]], [planets[7, i, 1]])
        mercurydot.set_3d_properties([planets[7, i, 2]])
        venus.set_data(planets[6, :i, 0], planets[6, :i, 1])
        venus.set_3d_properties(planets[6, :i, 2])
        venusdot.set_data([planets[6, i, 0]], [planets[6, i, 1]])
        venusdot.set_3d_properties([planets[6, i, 2]])
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
        return earth, jupiter, saturn, uranus, neptune, earthdot, jupiterdot, saturndot, uranusdot, neptunedot, sun, voyager2, mars, marsdot, venus, venusdot, mercury, mercurydot

    anim = animation.FuncAnimation(fig, animate, repeat=True, frames = tstep, interval=1)
    #anim.save('orbita.gif', writer='Pillow', fps=30)

    disconnect_zoom = zoom_factory(ax)
    pan_handler = panhandler(fig)"
    """
    plt.show()

if __name__ == "__main__":
    main()