import math
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, set_start_method
from optparse import OptionParser
import sys
import pickle
import copy

from animator import Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Lattice():
    """
    This class models a 2D lattice of magnetic spins, with a variable x and y
    number of spins, energy constant 'J', boltzman constant 'k' and temperature
    'T'.
    """

    def __init__(self, x, y, J, k, T, h, init_cond=None):

        self.x = x
        self.y = y
        self.J = J
        self.k = k
        self.T = T
        self.h = h
        self.eq_sweeps = 300 # Sweeps before taking measurements
        self.measure_freq = 10 # Frequency of measurements in sweeps
        self.bootstrap_samples = 100 # How many bootstrap runs to do
        self.prob_cache = {} # A store for common exponential values

        if (not init_cond):
            self.lattice = np.random.choice([1, -1], size=(self.x, self.y))
        elif (init_cond == "all"):
            self.lattice = np.ones((self.x, self.y), np.int8)
            print(self.lattice)
        elif (init_cond == "half"):
            self.lattice = np.ones((self.x, self.y), np.int8)
            for i in range(int(self.x/2)):
                self.lattice[i] = [-1 for j in range(self.y)]

        self.fileHandle = open("x{}_s{}.data", 'w')

    def write_to_file(self, *args):
        """
        Function expects values that are to be printed are passed as tuples:
        (value, format)
        For integers, `format` is a single value, the total number of digits
        For floats, it is a 2-value list (or tuple), first for the leading
        digits, second for decimal places.
        """
        for arg in args:
            val, form = arg
            t = type(val)
            if t == int:
                self.fileHandle.write( ("{0:" + str(form) + "d}\t").format(val))
            elif t == float or t == np.float64:
                self.fileHandle.write( ("{0: " + str(form[0]) + "." + str(form[1]) + "f}\t").format(val))
            elif t == str:
                self.fileHandle.write( ("{0:>" + str(form) + "s}\t").format(val))
            else:
                raise AttributeError(t)

        self.fileHandle.write("\n")

    def should_change(self, dE):
        """ A function to perform the (second part of) Markov test"""
        if dE in self.prob_cache:
            prob = self.prob_cache[dE]
        else:
            prob = math.exp(-dE/(self.k*self.T))
            self.prob_cache[dE] = prob
        dart = np.random.random()

        if (dart <= prob):
            return True
        else:
            return False


    def flip_delta_E(self, i ,j):
        """ Find the energy change from one spin flip """
        lattice = self.lattice

        iup     = (i + 1) % self.x
        idown   = (i - 1) % self.x
        jup     = (j + 1) % self.y
        jdown   = (j - 1) % self.y

        dE = -self.J * (lattice[i,j] * -1) * ( lattice[iup,j] + lattice[idown,j]
                                            + lattice[i,jup] + lattice[i,jdown])
        dE = 2*dE - self.h * (lattice[i,j] * -1) # field contribution
            #^ Avoids double counting unneccarily
        return dE

    def flip(self, i, j):
        self.lattice[i,j] = -1 * self.lattice[i,j]

    def evolve_Glauber(self):

        # Pick a spin at random
        ri = np.random.randint(self.x)
        rj = np.random.randint(self.y)

        dE = self.flip_delta_E(ri, rj)
        #print("Energy change is: " + str(dE))

        # If this change reduced the energy, accept it
        if (dE <= 0): self.flip(ri, rj)
        # Otherwise, accept probabilistically
        elif self.should_change(dE): self.flip(ri, rj)

    def total_magnetisation(self):
        return self.lattice.sum()

    def staggered_magnetisation(self):
        Ms = 0
        for i in range(self.x):
            for j in range(self.y):
                sgn = (-1)**((i+1) + (j+1))
                Ms += sgn * self.lattice[i,j]

        return Ms

def main():
    parser = OptionParser("Usage: >> python main.py [options]")
    parser.add_option("--na", action="store_true", default=False,
        help="Use this option to disable animation")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (def: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (def: 50)")
    parser.add_option("-t", action="store", dest="T", default=1.0, type="float",
        help="Use this to specify the temperature (def: 1.0)")
    parser.add_option("-n", action="store", dest="n_runs", default=10000, type="int",
        help="Use this to specify the number of runs (def: 10000)")
    parser.add_option("-j", action="store", dest="J", default=-1.0, type="float",
        help="Use this to specify the energy constant (def: -1.0)")
    parser.add_option("-k", action="store", default=1.0, type="float",
        help="Use this to specify a value for the boltzman constant")
    parser.add_option("-i", action="store", default=None,
        help="Use this to specify an initial condition:\t\t\t"
            + "\'all\' for all spins starting the same\t\t\t\t"
            + "\'half\' for a 50/50 split of spins")
    parser.add_option("-m", action="store", default=1.0, type="float", dest='h',
        help="Use this to specify a value for the magnetic field (default 0)")


    (options, args) = parser.parse_args()
    x = options.x
    y = options.y
    should_animate = not options.na
    num_runs = options.n_runs#
    temp = options.T
    J = options.J
    k = options.k
    h = options.h
    init_cond = options.i

    lattice = Lattice(x, y, J, k, temp, h, init_cond)

    run_sim(lattice, should_animate, num_runs)

def run_sim(lattice, should_animate, num_runs):

    plot_M_list = [lattice.total_magnetisation()]
    plot_Ms_list = [lattice.staggered_magnetisation()]

    if should_animate:
        lattice_queue = Queue()
        lattice_queue.put( (lattice.lattice[:], plot_Ms_list, plot_Ms_list) )
        animator = Animator(lattice_queue, lattice.measure_freq, num_runs)

        animator_proc = Process(target=animator.animate)
        animator_proc.start()

    for i in range(num_runs):
        for j in range(lattice.x*lattice.y):
            lattice.evolve_Glauber()
        if i % lattice.measure_freq == 0:
            plot_M_list.append(lattice.total_magnetisation())
            plot_Ms_list.append(lattice.staggered_magnetisation())
            if should_animate:
                package = (np.copy(lattice.lattice), plot_M_list[:], plot_Ms_list[:])
                lattice_queue.put(package)

    if should_animate: animator_proc.join()

    M_list = plot_M_list[int(np.ceil(lattice.eq_sweeps/lattice.measure_freq)):]
    Ms_list = plot_Ms_list[int(np.ceil(lattice.eq_sweeps/lattice.measure_freq)):]


if __name__ == '__main__':
    set_start_method("spawn")
    main()
