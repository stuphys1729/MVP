import math
import time
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process, Queue
from optparse import OptionParser
import sys
import pickle
import copy

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Animator():

    def __init__(self, lattice_queue):

        self.queue  = lattice_queue

        data = lattice_queue.get()

        self.fig, self.ax_array = plt.subplots()
        cont = self.ax_array.contourf(data, vmin=0, vmax=1)
        self.cbar = plt.colorbar(cont)

    def update(self, i):

        if (self.queue.empty()):
            time.sleep(0.1)
        else:
            self.ax_array.cla()
            data = self.queue.get()
            vmax = np.max(data)
            vmin = np.min(data)
            cont = self.ax_array.contourf(data, vmin=vmin, vmax=vmax)
            self.cbar.set_clim(vmin=vmin, vmax=vmax)
            #self.cbar.draw_all()
            return [cont, self.cbar]

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

class Lattice():
    """

    """

    def __init__(self, x, y, dels, delt, D, K, sigma):

        self.x = x
        self.y = y
        self.dels = dels
        self.delt = delt

        self.D = D
        self.K = K

        self.K_del  = self.K/(self.dels**2)
        self.dtD_ds = (self.delt*self.D)/(self.dels**2)

        self.phi = np.random.uniform(0.4, 0.6, size=(x,y) )

        self.new_phi = np.zeros( (x, y) )

        self.rho = self.calculate_rho(sigma)


    def calculate_rho(self, sigma):

        rho = np.zeros( (self.x,self.y) )

        for i in range(self.x):
            for j in range(self.y):
                r_x = int(self.x/2) - i
                r_y = int(self.y/2) - j
                r = r_x**2 + r_y**2

                rho[i,j] = np.exp(-r/sigma)

        return rho

    def update_phi(self):

        l = self.phi
        dt = self.delt
        for i in range(self.x):
            iup     = (i + 1) % self.x
            idown   = (i - 1) % self.x
            for j in range(self.y):
                jup     = (j + 1) % self.y
                jdown   = (j - 1) % self.y

                neighbours = [ l[iup,j], l[idown,j], l[i,jup], l[i,jdown] ]

                self.new_phi[i,j] = l[i,j] + self.dtD_ds*(sum(neighbours) - 4*l[i,j]) + dt*(self.rho[i,j] - self.K*l[i,j])

        self.phi = np.copy(self.new_phi)

    def update_advection(self, v0):

        l = self.phi
        dt = self.delt
        for i in range(self.x):
            iup     = (i + 1) % self.x
            idown   = (i - 1) % self.x
            for j in range(self.y):
                jup     = (j + 1) % self.y
                jdown   = (j - 1) % self.y

                neighbours = [ l[iup,j], l[idown,j], l[i,jup], l[i,jdown] ]

                self.new_phi[i,j] = l[i,j] + self.dtD_ds*(sum(neighbours) - 4*l[i,j]) + dt*(self.rho[i,j] - self.K*l[i,j] + v0*np.sin((2*np.pi*j)/(self.x))*(l[idown,j]-l[iup,j]) )

        self.phi = np.copy(self.new_phi)

def main():

    parser = OptionParser("Usage: >> python practice.py [options] <data_file>")
    parser.add_option("-a", action="store_true", default=False,
        help="Use this option to enable animation")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (default: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (default: 50)")
    parser.add_option("-n", action="store", dest="n_runs", default=1000, type="int",
        help="Use this to specify the number of runs (default: 1000)")
    parser.add_option("--sigma", action="store", default=10, type="float",
        help="Use this to specify the value of sigma (default: 10)")
    parser.add_option("-k", action="store", default=0.01, type="float",
        help="Use this option to specify the value of Kappa (default 0.01)")
    parser.add_option("-d", action="store", default=1, type="float",
        help="Use this option to specify the value of D (default 1)")
    parser.add_option("-v", action="store", default=0.5, type="float",
        help="Use this option to specify the value of v0 (default 0.5)")

    (options, args) = parser.parse_args()
    anim = options.a
    x = options.x
    y = options.y
    num_runs = options.n_runs
    D = options.d
    s = options.sigma
    k = options.k
    v = options.v

    lattice = Lattice(x, y, 1, 0.1, D, k, s)

    interval = 10
    if anim:
        lattice_queue = Queue()
        lattice_queue.put( (np.copy(lattice.phi)) )

        animator = Animator(lattice_queue)

        animator_proc = Process(target=animator.animate)
        animator_proc.start()

    for i in range(num_runs):
        #lattice.update_phi()
        lattice.update_advection(v)
        if (i % interval == 0):
            #print("Step: {} | phi[20,20]: {}".format(i, lattice.phi[20,20]))
            lattice_queue.put( np.copy(lattice.phi) )


if __name__ == '__main__':
    main()
