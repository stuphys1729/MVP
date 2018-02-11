import math
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
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

    def __init__(self, x, y, J, k, T, init_cond=None):

        self.x = x
        self.y = y
        self.J = J
        self.k = k
        self.T = T
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

    def total_energy(self):
        """ Finds the total energy of the lattice """

        lattice = self.lattice
        E = 0
        for i in range(self.x):
            iup = i + 1
            if (iup == self.x): iup = 0 # Periodic Boundary

            for j in range(self.y):

                jup = j + 1
                if (jup == self.y): jup = 0 # Periodic Boundary

                E += lattice[i,j] * (lattice[iup,j] + lattice[i,jup])
                # We only look up and right from each spin to avoid double
                # counting

        return -self.J * E

    def total_magnetisation(self):
        """ Sum of the individual spins """
        lattice = self.lattice

        M = 0
        for i in range(self.x):
            for j in range(self.y):
                M += self.lattice[i,j]

        return abs(M)

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

        return 2*dE # Avoids double counting unneccarily

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

    def are_neigbours(self, i1, j1, i2, j2):

        if (abs(i1-i2) < 2 and abs(j1-j2) <2):
            return True

        #periodic boundary consideration
        if (i1 == i2) and ((j1==0 and j2==self.y-1) or (j1==self.y-1 and j2==0)):
            return True
        if (j1 == j2) and ((i1==0 and i2==self.x-1) or (i1==self.x-1 and i2==0)):
            return True

        return False

    def swap_delta_E(self, i1, j1, i2, j2):
        """ Method assumes spins are different """

        if self.are_neigbours(i1, j1, i2, j2):
            return self.flip_delta_E(i1,j1) + self.flip_delta_E(i2,j2) +4*self.J

        return self.flip_delta_E(i1,j1) + self.flip_delta_E(i2,j2)


    def swap(self, i1, j1, i2, j2):
        temp = self.lattice[i1,j1]
        self.lattice[i1,j1] = self.lattice[i2,j2]
        self.lattice[i2,j2] = temp


    def evolve_Kawasaki(self):

        i1 = np.random.randint(self.x)
        j1 = np.random.randint(self.y)

        i2 = np.random.randint(self.x)
        j2 = np.random.randint(self.y)

        if (self.lattice[i1,j1] == self.lattice[i2,j2]): return 0

        dE = self.swap_delta_E(i1, j1, i2, j2)

        if (dE <= 0): self.swap(i1, j1, i2, j2)
        # Otherwise, accept probabilistically
        elif self.should_change(dE): self.swap(i1, j1, i2, j2)


    def calc_c(self, E_list):

        num_measures = len(E_list)
        E_bar = sum(E_list)/num_measures
        E_sq_list = [E**2 for E in E_list]
        E_sq_bar = sum(E_sq_list)/num_measures

        C = (1/(self.x*self.y*self.k*self.T**2))*(E_sq_bar - E_bar**2)

        return C, E_bar

    def calc_x(self, M_list):

        num_measures = len(M_list)
        M_bar = sum(M_list)/num_measures
        M_sq_list = [M**2 for M in M_list]
        M_sq_bar = sum(M_sq_list)/num_measures

        X = (1/(self.x*self.y*self.k*self.T**2))*(M_sq_bar - M_bar**2)

        return X, M_bar

    def bootstrap(self, master_list, observable):

        num_samples = 100
        sample_size = len(master_list)
        o_list = []
        for i in range(num_samples):
            sample_list = np.random.choice(master_list, sample_size)
            o, ignore = observable(sample_list)
            o_list.append(o)

        o_bar = sum(o_list)/num_samples
        o_sq_list = [o**2 for o in o_list]
        o_sq_bar = sum(o_sq_list)/num_samples

        if (o_sq_bar - o_bar**2) < 0:
            if np.isclose(o_sq_bar, o_bar**2, rtol=0.0001):
                return 0
            else:
                print("o_sq_bar: {0}\to_bar^2: {1}".format(o_sq_bar, o_bar**2))
                sys.exit("Got a negative argument to square root.")
        else:
            return np.sqrt(o_sq_bar - o_bar**2)


def main():

    parser = OptionParser("Usage: >> python main.py [options]")
    parser.add_option("-a", action="store_true", default=False,
        help="Use this option to enable animation")
    parser.add_option("-m", action="store", dest="method",
        help="Use this to specify the evolution method:\t\t"
            + "\'Glauber\'or \'G\'   : Glabuer Method\t\t\t"
            + "\'Kawasaki\' or \'K\' : Kawasaki Method")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (def: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (def: 50)")
    parser.add_option("-t", action="store", dest="T", default=1.0, type="float",
        help="Use this to specify the temperature (def: 1.0)")
    parser.add_option("-n", action="store", dest="n_runs", default=10000, type="int",
        help="Use this to specify the number of runs (def: 10000)")
    parser.add_option("-j", action="store", dest="J", default=1.0, type="float",
        help="Use this to specify the energy constant (def: 1.0)")
    parser.add_option("-k", action="store", default=1.0, type="float",
        help="Use this to specify a value for the boltzman constant")
    parser.add_option("-i", action="store",
        help="Use this to specify an initial condition:\t\t\t"
            + "\'all\' for all spins starting the same\t\t\t\t"
            + "\'half\' for a 50/50 split of spins")
    parser.add_option("-s", action="store_true", default=False,
        help="Use this option to run a sequence of simulations at different temperatures")
    parser.add_option("--plotnow", action="store_true", default=False,
        help="Use this option to specify whether to plot at the end of a sequence")
    parser.add_option("-p", action="store_true", default=False,
        help="Use this option along with a file name to plot from that data dump")

    # Get options
    (options, args) = parser.parse_args()
    if options.p:
        with open(args[0], 'rb') as f:
            data = pickle.load(f)
        plot_graph(data)
        return

    x = options.x
    y = options.y
    should_animate = options.a
    method = options.method
    num_runs = options.n_runs#
    temp = options.T
    J = options.J
    k = options.k
    init_cond = options.i
    sequence = options.s
    plot_now = options.plotnow

    if not init_cond:
        lattice = Lattice(x, y, J, k, temp)
    elif init_cond == "all":
        lattice = Lattice(x, y, J, k, temp, "all") # initialise all up
    elif init_cond == "half":
        lattice = lattice = Lattice(x, y, J, k, temp, "half") # initialise two bands
    else:
        sys.exit("You have entered an incorrect initial condition, please use\n"
        + "\'python main.py -h\' to get help on the options")

    if sequence:
        run_sims(lattice, method, should_animate, num_runs, plot_now)
    else:
        run_sim(lattice, method, should_animate, num_runs)

def run_sim(lattice, method, should_animate, num_runs):

    if (method == "Kawasaki") or (method == "K"):
        evolve = lattice.evolve_Kawasaki

    elif (method == "Glauber") or (method == "G"):
        evolve = lattice.evolve_Glauber

    else:
        sys.exit("You have entered an incorrect method, please use\n"
        + "\'python main.py -h\' to get help on the options")

    plot_E_list = [lattice.total_energy()]
    plot_M_list = [lattice.total_magnetisation()]

    if should_animate:
        lattice_queue = Queue()
        lattice_queue.put( (lattice.lattice[:], plot_E_list, plot_M_list) )
        animator = Animator(lattice_queue, lattice.measure_freq, num_runs)

        animator_proc = Process(target=animator.animate)
        animator_proc.start()

    for i in range(num_runs):
        for j in range(lattice.x*lattice.y):
            evolve()
        if i % lattice.measure_freq == 0:
            plot_E_list.append(lattice.total_energy())
            plot_M_list.append(lattice.total_magnetisation())
            if should_animate:
                #package = (copy.deepcopy(lattice.lattice), plot_E_list[:], plot_M_list[:])
                package = (lattice.lattice[:], plot_E_list[:], plot_M_list[:])
                lattice_queue.put(package)

    if should_animate: animator_proc.join() #lattice_queue.put("STOP")


    # Only keep the energy and Magnetisation after equilibriating
    E_list = plot_E_list[int(np.ceil(lattice.eq_sweeps/lattice.measure_freq)):]
    M_list = plot_M_list[int(np.ceil(lattice.eq_sweeps/lattice.measure_freq)):]

    M = None; X = None; Xerr = None;
    if (method == "Glauber") or (method == "G"):
        X, M = lattice.calc_x(M_list)
        Xerr = lattice.bootstrap(M_list, lattice.calc_x)

    C, E = lattice.calc_c(E_list)
    Cerr = lattice.bootstrap(E_list, lattice.calc_c)

    return E, C, M, X, Cerr, Xerr

def run_sims(lattice, method, should_animate, num_runs, plot_now):

    num_temps = 21
    dT = 0.2

    # setup toolbar
    sys.stdout.write("Running {} different temps: ".format(num_temps))
    sys.stdout.write("[%s]" % (" " * num_temps))
    sys.stdout.flush()
    sys.stdout.write("\b" * (num_temps+1)) # return to start of line, after '['

    E_list = []; C_list = []
    M_list = []; X_list = []
    Cerr_list = []; Xerr_list = []
    T_list = []
    for i in range(num_temps):
        E, C, M, X, Cerr, Xerr = run_sim(lattice, method, should_animate,
            num_runs)
        E_list.append(E); C_list.append(C)
        M_list.append(M); X_list.append(X)
        Cerr_list.append(Cerr); Xerr_list.append(Xerr)
        T_list.append(lattice.T)

        #format_results(lattice, E, C, M, X, Cerr, Xerr)
        sys.stdout.write("#")
        sys.stdout.flush()

        lattice.T += dT
        lattice.prob_cache = {}

    sys.stdout.write("\n")

    data = {
        'E_list' : E_list,
        'M_list' : M_list,
        'T_list' : T_list,
        'C_list' : C_list,
        'X_list' : X_list,
        'Xerr_list' : Xerr_list,
        'Cerr_list' : Cerr_list
    }

    if plot_now:
        plot_graph(data)
    else:
        file_name = 'data_m{}_s{}_r{}.pickle'.format(method, lattice.x, num_runs)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print("Wrote file: " + file_name)

    return

def plot_graph(data):

    E_list = data['E_list'][2:]
    M_list = data['M_list'][2:]
    T_list = data['T_list'][2:]
    C_list = data['C_list'][2:]
    X_list = data['X_list'][2:]
    Xerr_list = data['Xerr_list'][2:]
    Cerr_list = data['Cerr_list'][2:]

    if M_list[0]: # Checks if we are tracking magnetisation

        fig, axes = plt.subplots(2, 2)

        axes[0,0].plot(T_list, E_list, label="Energy", marker=".")
        axes[0,0].set_ylabel("Energy")
        axes[0,0].set_title("Lattice Energy")

        axes[0,1].plot(T_list, M_list, label="Magnetisation", marker=".")
        axes[0,1].set_ylabel("Magnetisation")
        axes[0,1].set_title("Lattice Magnetisation (Absolute)")

        axes[1,0].plot(T_list, C_list, label="Specific Heat")
        axes[1,0].errorbar(T_list, C_list, fmt='bo', yerr=Cerr_list, capsize=4)
        axes[1,0].set_xlabel("Temperature")
        axes[1,0].set_ylabel("Specific Heat")
        axes[1,0].set_title("Lattice Specific Heat")

        axes[1,1].plot(T_list, X_list, label="Susceptibility")
        axes[1,1].errorbar(T_list, X_list, fmt='bo', yerr=Xerr_list, capsize=4)
        axes[1,1].set_xlabel("Temperature")
        axes[1,1].set_ylabel("Susceptibility")
        axes[1,1].set_title("Lattice Susceptibility")

    else: # We shouldn't plot the magnetisation or Susceptibility

        fig, axes = plt.subplots(2, 1)

        axes[0].plot(T_list, E_list, label="Energy", marker=".")
        axes[0].set_ylabel("Energy")
        axes[0].set_title("Lattice Energy")

        axes[1].plot(T_list, C_list, label="Specific Heat")
        axes[1].errorbar(T_list, C_list, fmt='bo', yerr=Cerr_list, capsize=4)
        axes[1].set_xlabel("Temperature")
        axes[1].set_ylabel("Specific Heat")
        axes[1].set_title("Lattice Specific Heat")


    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    #plt.savefig("Size{}_Runs{}".format(lattice.x, num_runs), dpi=500)

    return

def format_results(lattice, E, C, M, X, Cerr, Xerr):
    if M:
        print("Temp: {0:.1f}\tE: {1:.3f}\tC: {2:.3f} +/- {5:.3f}\tM: {3:.3f}\tX: {4:.3f} +/- {6:.3f}".format(
            lattice.T, E, C, M, X, Cerr, Xerr))
    else:
        print("Temp: {0:.1f}\tE: {1:.3f}\tC: {2:.3f} +/- {3:.3f}".format(
            lattice.T, E, C, Cerr))

if __name__ == '__main__':
    main()
