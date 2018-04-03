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

from animators import Poisson_Animator as Animator
from animators import Poisson_Animator_After as Post_Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Lattice():
    """

    """

    def __init__(self, x, y, dels, mu0, init_cond=None, omega=1):

        self.x = x
        self.y = y

        self.ds = dels
        self.omega = omega
        self.mu0 = mu0

        self.J = np.zeros( (x,y) )

        if init_cond == "wire":
            self.J[int(x/2),int(y/2)] = 1
        else:
            sys.exit("Initial conditions not recognised, aborting...")

        self.A = np.zeros( (x,y) )
        self.new_A = np.zeros( (x,y) )



    def update_A_SOR(self, omega=None):
        l = self.A
        ds2mu0 = (self.ds**2)*self.mu0
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

                neighbours = [ l[iup,j], l[idown,j], l[i,jup], l[i,jdown] ]

                temp = (1-w)*self.A[i,j] + w*(1/4)*(sum(neighbours) + ds2mu0*self.J[i,j])
                diff += abs(self.A[i,j] - temp)
                self.A[i,j] = temp

        return diff

    def get_potential_from(self, ri, rj):
        r_list = []
        A_list = []
        for i in range(1, self.x-1):
            for j in range(1, self.y-1):
                r_x = ri - i; r_y = rj - j;
                r = math.sqrt( r_x**2 + r_y**2 )
                r_list.append(r)
                A_list.append(self.A[i,j])

        return r_list, A_list

    def get_B_field(self):
        l = self.A
        ds = self.ds

        B_xlist = np.zeros( (self.x-2, self.y-2) )
        B_ylist = np.zeros( (self.x-2, self.y-2) )

        for i in range(1, self.x-1):
            for j in range(1, self.y-1):

                Bx = (l[i,j+1] - l[i,j-1]) / (2*ds)
                By = - (l[i+1,j] - l[i-1,j]) / (2*ds)

                # Normalisation
                B_ylist[i-1,j-1] = Bx / np.sqrt( (Bx**2 + By**2) )
                B_xlist[i-1,j-1] = By / np.sqrt( (Bx**2 + By**2) )

        return B_xlist, B_ylist



def main():
    parser = OptionParser("Usage: >> python poisson_magnetic.py [options] <data_file>")
    parser.add_option("-a", action="store_true", default=False,
        help="Use this option to enable animation")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (default: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (default: 50)")
    parser.add_option("-n", action="store", dest="n_runs", default=1000, type="int",
        help="Use this to specify the maximum number of runs (default: 10000)")
    parser.add_option("-i", action="store", default="wire", type="string",
        help="Use this to specify the initial condition")
    parser.add_option("-m", action="store", default="s", type="string",
        help="Use this to specify the update method (default: SOR)")
    parser.add_option("-t", action="store", default=0.001, type="float",
        help="Use this to specify the finish tolerance (default: 0.001)")
    parser.add_option("-w", action="store", default=1.94, type="float",
        help="Use this to specify the value of omega (default: 1.94)")
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

    lattice = Lattice(x, y, 0.1, 1.0, init_cond, omega)

    if anim:
        lattice_queue = Queue()
        lattice_queue.put( (copy.deepcopy(lattice.A)) )

        animator = Animator(lattice_queue)

        animator_proc = Process(target=animator.animate)
        animator_proc.start()

    for i in range(num_runs):
        diff = lattice.update_A_SOR()
        if (i % 5 == 0):
            if anim:
                lattice_queue.put( (copy.deepcopy(lattice.A)) )
            print("Sweep number {0:8d} | Diff: {1:.02e}".format(i, diff))
        if diff <= tolerance:
            break

    print("Converged after {} steps".format(i))

    r_list, A_list = lattice.get_potential_from( int(x/2), int(y/2) )

    if anim:
        animator_proc.join()
        plt.clf()
    plt.scatter(r_list, A_list)
    ax = plt.gca()
    ax.set_ylim(0, max(A_list)+max(A_list)*0.01)
    plt.show()


    plt.clf()
    time.sleep(0.1)

    # Make the grid
    X, Y = np.meshgrid( np.arange(1, x-1),
                        np.arange(1, y-1) )

    Ex, Ey = lattice.get_B_field()

    plt.quiver(X, Y, Ex, Ey, width=0.001)

    plt.show()


def show_animation(data):
    pass

if __name__ == '__main__':
    main()
