import math
import time
import logging
import numpy as np
from multiprocessing import Process, Queue
from optparse import OptionParser, OptionGroup
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
from random import sample

from animators import SIRS_Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Lattice():
    """
    """

    def __init__(self, x, y, p1, p2, p3, immune_frac=None):

        self.x = x
        self.y = y

        self.inf_prob = p1
        self.rec_prob = p2
        self.sup_prob = p3

        self.lattice = np.random.choice( [0, 1], size=(x,y) )
        if immune_frac:
            num_immune = int(np.floor(immune_frac*self.x*self.y))
            indices = []
            for i in range(self.x):
                for j in range(self.y):
                    indices.append( (i,j) )
            selections = sample(indices, num_immune)
            for (i,j) in selections:
                self.lattice[i,j] = 3 # Immune


    def update(self, i, j):

        l = self.lattice

        iup     = (i + 1) % self.x
        jup     = (j + 1) % self.y
        idown   = (i - 1) % self.x
        jdown   = (j - 1) % self.y

        cell = l[i,j]
        dart = np.random.random()
        if cell == 0: # If succeptible
            if any([c == 1 for c in [ l[iup,j], l[idown,j],
                                            l[i,jup], l[i,jdown]]]):
                if (dart <= self.inf_prob):
                    l[i,j] = 1

        elif cell == 1: # If infected
            if (dart <= self.rec_prob):
                l[i,j] = 2

        elif cell == 2: # If recovered
            if (dart <= self.sup_prob):
                l[i,j] = 0

    def get_inf_frac(self):
        inf = 0
        for i in range(self.x):
            for j in range(self.y):
                if self.lattice[i,j] == 1:
                    inf += 1
        return inf/(self.x*self.y)

    def sweep(self):
        for i in range(self.x*self.y):
            ri = np.random.randint(self.x)
            rj = np.random.randint(self.y)

            self.update(ri, rj)

def get_var(vlist):
    n = len(vlist)
    v_bar = sum(vlist)/n
    v_sq_list = [v**2 for v in vlist]
    v_sq_bar = sum(v_sq_list)/n

    var = (v_sq_bar - v_bar**2)/n

    return v_bar, var

def main():

    parser = OptionParser("Usage: >> python main.py [options] [<data_file>]")
    parser.add_option("-a", action="store_true", default=False,
        help="Use this option to enable animation")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (default: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (default: 50)")
    parser.add_option("-n", action="store", dest="n_runs", default=5000, type="int",
        help="Use this to specify the number of runs (default: 5000)")
    parser.add_option("--p1", action="store", default=0.5, type="float",
        help="The probability for a susceptible actor to be infected")
    parser.add_option("--p2", action="store", default=0.5, type="float",
        help="The probability for an infected actor to recover")
    parser.add_option("--p3", action="store", default=0.5, type="float",
        help="The probability for a recovered actor to become susceptible")
    parser.add_option("-c", action="store_true", default=False,
        help="Use this option to create a contour plot")
    parser.add_option("--plotnow", action="store_true", default=False,
        help="Use this option to specify whether to plot at the end of a sequence")
    parser.add_option("-p", action="store_true", default=False,
        help="Use this option along with a file name to plot from that data dump")
    parser.add_option("--immune", action="store", default=None, type="float",
        help="Use this to specify the fraction of immune actors")
    parser.add_option("--immunesweep", action="store_true", default=False,
        help="Do a sweep of immune fractions")

    presets = OptionGroup(parser, "Presets")
    presets.add_option("--absorb", action="store_true", default=False,
        help="preset values for an absorbing state")
    presets.add_option("--dynamic", action="store_true", default=False,
        help="preset values for a dynamic equilibrium state")
    presets.add_option("--waves", action="store_true", default=False,
        help="preset values for an state with waves of infection")

    parser.add_option_group(presets)

    (options, args) = parser.parse_args()

    if options.p:
        with open(args[0], 'rb') as f:
            data = pickle.load(f)
        plot_graph(data)
        return

    num_runs = options.n_runs
    x = options.x
    y = options.y
    anim = options.a
    p1 = options.p1
    p2 = options.p2
    p3 = options.p3
    contour = options.c
    plot_now = options.plotnow
    immune_frac = options.immune
    imsweep = options.immunesweep

    if contour:
        contour_plot(x, y, num_runs)

    elif imsweep:
        immune_plot(x, y, num_runs)


    else:
        if options.absorb:
            p1 = 1.0; p2 = 0.1; p3 = 0
        elif options.dynamic:
            p1 = 0.5; p2 = 0.5; p3 = 0.5
        elif options.waves:
            p1 = 0.8; p2 = 0.1; p3 = 0.01

        lattice = Lattice(x, y, p1, p2, p3, immune_frac)

        if anim:
            lattice_queue = Queue()
            lattice_queue.put( (deepcopy(lattice.lattice)) )

            animator = SIRS_Animator(lattice_queue, num_runs)

            animator_proc = Process(target=animator.animate)
            animator_proc.start()

        for i in range(num_runs):
            lattice.sweep()
            if anim:
                lattice_queue.put( (deepcopy(lattice.lattice)) )

        if anim:
            animator_proc.join()
            plt.clf()

        return

def contour_plot(x, y, num_runs):
    num_points = 21
    interval = 0.05
    # setup toolbar
    sys.stdout.write("Running {} different points: ".format(num_points))
    sys.stdout.write("[%s]" % (" " * num_points))
    sys.stdout.flush()
    sys.stdout.write("\b" * ((num_points)+1)) # return to start of line, after '['

    p2 = 0.5
    p1 = 0
    frac_mat = []
    var_mat = []
    for i in range(num_points):
        p3 = 0
        rec_fracs = []
        rec_vars = []
        for j in range(num_points):
            lattice = Lattice(x, y, p1, p2, p3)
            frac_list = []
            for k in range(num_runs):
                lattice.sweep()
                if (k % 10) == 0 and k > 200:
                    frac_list.append(lattice.get_inf_frac())
            frac, var = get_var(frac_list)
            rec_fracs.append(frac)
            rec_vars.append(var)
            p3 += interval
        frac_mat.append(rec_fracs[:])
        var_mat.append(rec_vars[:])
        p1 += interval
        sys.stdout.write("#")
        sys.stdout.flush()

    print("")

    p1_list = list(np.linspace(0, 1, num_points))
    p3_list = p1_list[:]

    data = {
        'p1_list' : p1_list,
        'p3_list' : p3_list,
        'frac_mat': frac_mat,
        'var_mat' : var_mat
    }

    if plot_now:
        plot_graph(data)
    else:
        file_name = 'data_s{}_r{}_n{}.pickle'.format(lattice.x, num_runs, num_points)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print("Wrote file: " + file_name)

    return

def immune_plot(x, y, num_runs):
    num_points = 11
    interval = 0.05
    # setup toolbar
    sys.stdout.write("Running {} different points: ".format(num_points))
    sys.stdout.write("[%s]" % (" " * num_points))
    sys.stdout.flush()
    sys.stdout.write("\b" * ((num_points)+1)) # return to start of line, after '['

    p1 = 0.5; p2 = 0.5; p3 = 0.5
    im_frac = 0
    im_frac_list = []
    for i in range(num_points):
        lattice = Lattice(x, y, p1, p2, p3, im_frac)
        frac_list = []
        for j in range(num_runs):
            lattice.sweep()
            if (j % 10) == 0 and j > 1000:
                frac_list.append(lattice.get_inf_frac())

        av_frac = np.mean(frac_list)
        im_frac_list.append(av_frac)
        im_frac += interval
        sys.stdout.write("#")
        sys.stdout.flush()

    print("")

    im_list = list(np.linspace(0, 0.5, num_points))

    data = {
        'im_list'       : im_list,
        'im_frac_list'  : im_frac_list
    }

    if plot_now:
        plot_graph(data)
    else:
        file_name = 'data_im_s{}_r{}_n{}.pickle'.format(lattice.x, num_runs, num_points)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print("Wrote file: " + file_name)

    return

def plot_graph(data):

    if 'p1_list' in data:
        p1_list = data['p1_list']
        p3_list = data['p3_list']
        frac_mat= data['frac_mat']
        var_mat = data['var_mat']

        plt.contourf(p1_list, p3_list, frac_mat)
        plt.colorbar()
        plt.xlabel("P1 (Probability of Infection)")
        plt.ylabel("P3 (Probability of Susceptibility)")
        plt.title("Average Infection at P2=0.5")
        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        plt.show()
        #print(frac_mat)
        #print("")

        plt.clf()
        plt.contourf(p1_list, p3_list, var_mat)
        plt.colorbar()
        plt.xlabel("P1 (Probability of Infection)")
        plt.ylabel("P3 (Probability of Susceptibility)")
        plt.title("Average Infection Variance at P2=0.5")
        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        plt.show()
        #print(var_mat)

    else:
        im_list = data['im_list']
        im_frac_list = data['im_frac_list']

        plt.plot(im_list, im_frac_list)
        plt.xlabel("Immune Fraction")
        plt.ylabel("Average Infected Fraction")
        plt.show()

    return

if __name__ == '__main__':
    main()
