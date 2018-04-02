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

from animators import Cahn_Hill_Animator as Animator
from animators import Cahn_Hill_Animator_After as Post_Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Lattice():
    """

    """

    def __init__(self, x, y, dels, delt, a, K, M, init_cond=None):

        self.x = x
        self.y = y
        self.dels = dels
        self.delt = delt

        self.a = a
        self.K = K
        self.M = M

        self.K_del  = self.K/(self.dels**2)
        self.dtM_ds = (self.delt*self.M)/(self.dels**2)

        if not init_cond:
            self.phi = np.random.uniform(-0.1, 0.1, size=(x,y) )
        elif abs(init_cond) < 1:
            self.phi = np.random.uniform(init_cond-0.1, init_cond+0.1, size=(x,y) )
        else:
            sys.exit("Initial conditions not recognised, aborting...")

        self.new_phi = np.zeros( (x, y) )

        self.mu = np.zeros( (x,y) )


    def update_mu(self, i, j):
        a = self.a
        l = self.phi
        iup     = (i + 1) % self.x
        jup     = (j + 1) % self.y
        idown   = (i - 1) % self.x
        jdown   = (j - 1) % self.y
        neighbours = [ l[iup,j], l[idown,j], l[i,jup], l[i,jdown] ]

        self.mu[i,j] = -a*l[i,j] + a*l[i,j]**3 - self.K_del*(sum(neighbours)- 4*l[i,j])


    def update_phi(self, i, j):

        l = self.mu
        iup     = (i + 1) % self.x
        jup     = (j + 1) % self.y
        idown   = (i - 1) % self.x
        jdown   = (j - 1) % self.y
        neighbours = [ l[iup,j], l[idown,j], l[i,jup], l[i,jdown] ]

        self.new_phi[i,j] = self.phi[i,j] + self.dtM_ds*(sum(neighbours) - 4*l[i,j])

    def get_free_energy(self):
        a = self.a
        K = self.K
        l = self.phi
        ds = self.dels

        f = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(self.x):
            for j in range(self.y):

                iup     = (i + 1) % self.x
                jup     = (j + 1) % self.y
                idown   = (i - 1) % self.x
                jdown   = (j - 1) % self.y

                grad_phi_i = (l[iup,j]-l[idown,j])/(2*ds)

                grad_phi_j = (l[i,jup]-l[i,jdown])/(2*ds)

                grad_phi_sq = grad_phi_i**2 + grad_phi_j**2

                f += (-a/2)*l[i,j]**2 + (a/4)*l[i,j]**4 + (K/2)*grad_phi_sq

        return f

    def update(self):

        for i in range(self.x):
            for j in range(self.y):
                self.update_mu(i,j)

        for i in range(self.x):
            for j in range(self.y):
                self.update_phi(i,j)

        self.phi = copy.deepcopy(self.new_phi)
        #self.phi = self.new_phi



def main():

    parser = OptionParser("Usage: >> python cahn_hill.py [options] <data_file>")
    parser.add_option("-a", action="store_true", default=False,
        help="Use this option to enable animation")
    parser.add_option("-f", action="store_true", default=False,
        help="Use this option to enable free energy plot at the end")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (default: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (default: 50)")
    parser.add_option("-n", action="store", dest="n_runs", default=10000, type="int",
        help="Use this to specify the number of runs (default: 10000)")
    parser.add_option("-i", action="store", default=0.0, type="float",
        help="Use this to specify the initial value of phi (default: 0.0)")
    parser.add_option("--anim", action="store_true", default=False,
        help="Use this option along with a data file to animate from it")
    parser.add_option("--interval", action="store", dest="interval", default=100, type="int",
        help="Use this to specify the gap between data grabs (default: 100)")

    (options, args) = parser.parse_args()
    if options.anim:
        with open(args[0], 'rb') as f:
            data = pickle.load(f)
        show_animation(data)
        return

    x = options.x
    y = options.y
    num_runs = options.n_runs
    anim = options.a
    init_cond = options.i
    f_plot = options.f
    interval = options.interval

    lattice = Lattice(x, y, 1.0, 2.0, 0.1, 0.1, 0.1, init_cond)
    #print(lattice.phi)

    if anim:
        lattice_queue = Queue()
        lattice_queue.put( (copy.deepcopy(lattice.phi)) )

        animator = Animator(lattice_queue)

        animator_proc = Process(target=animator.animate)
        animator_proc.start()

        f_list = []
        for i in range(num_runs):
            lattice.update()
            if (i % interval == 0):
                f = lattice.get_free_energy()
                f_list.append(f)
                lattice_queue.put( copy.deepcopy(lattice.phi) )
                print("Sweep number {0:8d} | Free Energy: {1:3.2f}".format(i,f))

    else:

        lattices = []
        lattices.append( copy.deepcopy(lattice.phi))

        f_list = []
        for i in range(num_runs):
            lattice.update()
            if (i % interval) == 0:
                f = lattice.get_free_energy()
                f_list.append(f)
                lattices.append( copy.deepcopy(lattice.phi) )
                print("Sweep number {0:8d} | Free Energy: {1:7.02f}".format(i,f))
                print(lattice.phi.max())
                print(lattice.phi.min())

        data = [ lattices, f_list, interval]

        file_name = 'data_im_s{}_r{}_i{}.pickle'.format(lattice.x, num_runs, init_cond)
        with open(file_name, 'wb') as f:
            pickle.dump(lattices, f, pickle.HIGHEST_PROTOCOL)
        print("Wrote file: " + file_name)

    if f_plot:
        if anim:
            animator_proc.join()

        plt.clf()

        x_list = np.linspace(0, num_runs, len(f_list))
        plt.plot(x_list, f_list)
        plt.show()

def show_animation(data):

    if len(data) != 3: # Old version of the pickling
        anim = Post_Animator(data)
        anim.animate()

    lattices = data[0]
    f_energy = data[1]
    x_list = np.linspace(0, num_runs, len(f_list))
    interval = data[2]

    anim = Post_Animator(lattices, interval)
    anim.animate()

if __name__ == '__main__':
    main()
