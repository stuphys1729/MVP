import math
import time
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from optparse import OptionParser
import sys
import pickle
import copy
from mpl_toolkits.mplot3d import axes3d

from animators import Poisson_Animator as Animator
from animators import Poisson_Animator_After as Post_Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )


class Lattice():
    """

    """

    def __init__(self, x, y, z, dels, e0, init_cond=None, omega=1):

        self.x = x
        self.y = y
        self.z = z

        self.dels = dels
        self.omega = omega
        self.e0 = e0

        self.rho = np.zeros( (x,y,z) )

        if init_cond == "point":
            self.rho[int(x/2),int(y/2),int(z/2)] = 1
        else:
            sys.exit("Initial conditions not recognised, aborting...")

        self.phi = np.zeros( (x,y,z) )
        self.new_phi = np.zeros( (x,y,z) )


    def update_phi_jacobi(self):
        l = self.phi
        diff = 0
        ds2 = self.dels**2
        # don't change the boundaries (keep them at 0)
        for i in range(1, self.x-1):
            iup     = (i + 1)
            idown   = (i - 1)
            for j in range(1, self.y-1):
                jup     = (j + 1)
                jdown   = (j - 1)
                for k in range(1, self.z-1):
                    kup     = (k + 1)
                    kdown   = (k - 1)

                    neighbours = [ l[iup,j,k], l[idown,j,k], l[i,jup,k],
                                    l[i,jdown,k], l[i,j,kup], l[i,j,kdown] ]

                    self.new_phi[i,j,k] = (1/6)*(sum(neighbours) + ds2*self.rho[i,j,k])
                    diff += abs(self.new_phi[i,j,k] - self.phi[i,j,k])

        self.phi = copy.deepcopy(self.new_phi)
        return diff


    def update_phi_gauss(self):
        l = self.phi
        ds2 = self.dels**2
        diff = 0
        # don't change the boundaries (keep them at 0)
        for i in range(1, self.x-1):
            iup     = (i + 1)
            idown   = (i - 1)
            for j in range(1, self.y-1):
                jup     = (j + 1)
                jdown   = (j - 1)
                for k in range(1, self.z-1):
                    kup     = (k + 1)
                    kdown   = (k - 1)

                    neighbours = [ l[iup,j,k], l[idown,j,k], l[i,jup,k],
                                    l[i,jdown,k], l[i,j,kup], l[i,j,kdown] ]

                    temp = (1/6)*(sum(neighbours) + ds2*self.rho[i,j,k])
                    diff += abs(self.phi[i,j,k] - temp)
                    self.phi[i,j,k] = temp

        return diff


    def update_phi_SOR(self, omega=None):
        l = self.phi
        ds2 = self.dels**2
        diff = 0
        if omega:
            w = omega
        else:
            w = self.omega
        # don't change the boundaries (keep them at 0)
        for i in range(1, self.x-1):
            iup     = (i + 1) % self.x
            idown   = (i - 1) % self.x
            for j in range(1, self.y-1):
                jup     = (j + 1) % self.y
                jdown   = (j - 1) % self.y
                for k in range(1, self.z-1):
                    kup     = (k + 1) % self.z
                    kdown   = (k - 1) % self.z

                    neighbours = [ l[iup,j,k], l[idown,j,k], l[i,jup,k],
                                    l[i,jdown,k], l[i,j,kup], l[i,j,kdown] ]

                    temp = (1-w)*self.phi[i,j,k] + w*(1/6)*(sum(neighbours) + ds2*self.rho[i,j,k])
                    diff += abs(self.phi[i,j,k] - temp)
                    self.phi[i,j,k] = temp

        return diff

    def get_potential_from(self, ri, rj, rk):
        r_list = []
        phi_list = []
        for i in range(1, self.x-1):
            for j in range(1, self.y-1):
                for k in range(1, self.z-1):
                    #(self.dels**3)*
                    r_x = ri - i; r_y = rj - j; r_z = rk - k
                    r = math.sqrt( r_x**2 + r_y**2 + r_z**2 )
                    #r = math.sqrt( ((ri - i)**2 + (rj - j)**2 + (rk - k)**2) )
                    r_list.append(r)
                    phi_list.append(self.phi[i,j,k])

        return r_list, phi_list

    def get_E_field(self):
        l = self.phi
        ds = self.dels

        E_xlist = np.zeros( (self.x-2, self.x-2, self.z-2) )
        E_ylist = np.zeros( (self.x-2, self.x-2, self.z-2) )
        E_zlist = np.zeros( (self.x-2, self.x-2, self.z-2) )

        for i in range(1, self.x-1):
            for j in range(1, self.y-1):
                for k in range(1, self.z-1):

                    E_xlist[i-1,j-1,k-1] = (l[i+1,j,k] - l[i-1,j,k]) / (2*ds)
                    E_ylist[i-1,j-1,k-1] = (l[i,j+1,k] - l[i,j-1,k]) / (2*ds)
                    E_zlist[i-1,j-1,k-1] = (l[i,j,k+1] - l[i,j,k-1]) / (2*ds)

        return E_xlist, E_ylist, E_zlist



def main():
    parser = OptionParser("Usage: >> python poisson.py [options] <data_file>")
    parser.add_option("-a", action="store_true", default=False,
        help="Use this option to enable animation")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (default: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (default: 50)")
    parser.add_option("-z", action="store", dest="z", default=50, type="int",
        help="Use this to specify the z-axis size (default: 50)")
    parser.add_option("-n", action="store", dest="n_runs", default=1000, type="int",
        help="Use this to specify the number of runs (default: 1000)")
    parser.add_option("-i", action="store", default="point", type="string",
        help="Use this to specify the initial condition")
    parser.add_option("-m", action="store", default="s", type="string",
        help="Use this to specify the update method (default: SOR)")
    parser.add_option("-t", action="store", default=0.01, type="float",
        help="Use this to specify the finish tolerance (default: 0.01)")
    parser.add_option("-w", action="store", default=1.0, type="float",
        help="Use this to specify the value of omega (default: 1.0)")
    parser.add_option("--anim", action="store_true", default=False,
        help="Use this option along with a data file to animate from it")

    (options, args) = parser.parse_args()
    if options.anim:
        with open(args[0], 'rb') as f:
            data = pickle.load(f)
        show_animation(data)
        return

    num_runs = options.n_runs
    anim = options.a
    init_cond = options.i
    tolerance = options.t
    omega = options.w

    x = options.x
    y = options.y
    z = options.z

    lattice = Lattice(x, y, z, 0.1, 1.0, init_cond, omega)

    if options.m == "j":
        method = lattice.update_phi_jacobi
    elif options.m == "g":
        method = lattice.update_phi_gauss
    elif options.m == "s":
        method = lattice.update_phi_SOR
    else:
        raise AttributeError("Method type not identified")

    if anim:
        lattice_queue = Queue()
        lattice_queue.put( (copy.deepcopy(lattice.phi[:,:,int(z/2)])) )

        animator = Animator(lattice_queue)

        animator_proc = Process(target=animator.animate)
        animator_proc.start()

    for i in range(num_runs):
        #diff = lattice.update_phi_jacobi()
        #diff = lattice.update_phi_gauss()
        diff = method()
        if (i % 5 == 0):
            if anim:
                lattice_queue.put( (copy.deepcopy(lattice.phi[:,:,int(z/2)])) )
            print("Sweep number {0:8d} | Diff: {1:7.02f}".format(i, diff))
        if diff <= tolerance:
            break



    r_list, phi_list = lattice.get_potential_from(int(x/2), int(y/2), int(z/2))

    if anim:
        animator_proc.join()
        plt.clf()
    #plt.scatter(r_list, phi_list)
    #plt.show()

    #plt.clf()
    #time.sleep(0.5)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make the grid
    X, Y, Z = np.meshgrid(np.arange(1, x-1),
                          np.arange(1, y-1),
                          np.arange(1, z-1))

    Ex, Ey, Ez = lattice.get_E_field()

    ax.quiver(X, Y, Z, Ex, Ey, Ez, length=0.5, normalize=True)

    plt.show()

def show_animation(data):
    pass

if __name__ == '__main__':
    main()
