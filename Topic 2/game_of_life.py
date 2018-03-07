import math
import time
import logging
import numpy as np
from multiprocessing import Process, Queue
from optparse import OptionParser
import sys
from copy import deepcopy
import matplotlib.pyplot as plt

from animators import GoL_Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Lattice():
    """

    """

    def __init__(self, x, y, init_cond=None):

        self.x = x
        self.y = y

        if init_cond == "random":
            self.lattice = np.random.choice( [1, 0], size=(x,y) )
        elif init_cond == "glider":
            self.lattice = np.zeros( (x,y), dtype=np.int8)
            self.lattice[0,1] = 1
            self.lattice[1,2] = 1
            self.lattice[2,0] = 1
            self.lattice[2,1] = 1
            self.lattice[2,2] = 1

        elif init_cond == "block":
            self.lattice = np.zeros( (x,y), dtype=np.int8)
            self.lattice[1,1] = 1
            self.lattice[1,2] = 1
            self.lattice[2,1] = 1
            self.lattice[2,2] = 1

        elif init_cond == "blink":
            self.lattice = np.zeros( (x,y), dtype=np.int8)
            self.lattice[2,2] = 1
            self.lattice[2,3] = 1
            self.lattice[2,4] = 1

        else:
            sys.exit("Initial condition not recognised. Aborting...")

    def update(self, i, j):
        l = self.lattice

        iup     = (i + 1) % self.x
        jup     = (j + 1) % self.y
        idown   = (i - 1) % self.x
        jdown   = (j - 1) % self.y

        neighbours = [l[iup,j], l[iup,jup], l[i,jup], l[idown,jup], l[idown,j],
                        l[idown,jdown], l[i,jdown], l[iup,jdown]]

        num_alive = sum(neighbours)

        if l[i,j] == 1: # Cell is alive
            if (num_alive==2) or (num_alive==3):
                return 1 # Cell lives on
            else:
                return 0 # Cell dies
        else: # Cell is dead
            if num_alive == 3:
                return 1 # Cell is revived
            else:
                return 0 # Cell stays dead


    def evolve(self):
        temp = np.zeros( (self.x,self.y), dtype=np.int8)
        for i in range(self.x):
            for j in range(self.y):
                temp[i,j] = self.update(i,j)

        self.lattice = temp

    def get_CoM(self):
        """ """
        boundary = False
        points = []
        for i in range(self.x):
            for j in range(self.y):
                if self.lattice[i,j] == 1:
                    if (i < 3) or (i > self.x-3) or (j < 3) or (j > self.y-3):
                        boundary = True
                        break
                    else:
                        points.append( (i,j) ) # Identify live cells

        if boundary:
            return None
        com_x = 0
        com_y = 0
        for r in points:
            com_x += r[0]
            com_y += r[1]

        x = com_x/len(points)
        y = com_y/len(points)

        return (x,y)

def main():

    parser = OptionParser("Usage: >> python main.py [options]")
    parser.add_option("-a", action="store_true", default=False,
        help="Use this option to enable animation")
    parser.add_option("-i", action="store", dest="init", default="random",
        help="Use this to specify the initial condition:\t\t"
            + "\'random\'or \'r\'   : random allocation\t\t\t\t"
            + "\'glider\' or \'g\'  : simple glider\t\t\t\t"
            + "\'block\' or \'b\'   : absorbing block state\t\t\t"
            + "\'blink\' or \'i\'   : simple blinker")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (default: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (default: 50)")
    parser.add_option("-n", action="store", dest="n_runs", default=1000, type="int",
        help="Use this to specify the number of runs (def: 10000)")

    (options, args) = parser.parse_args()

    num_runs = options.n_runs
    init_cond = options.init
    x = options.x
    y = options.y
    anim = options.a

    lattice = Lattice(x, y, init_cond)

    if anim:
        lattice_queue = Queue()
        lattice_queue.put( (deepcopy(lattice.lattice)) )

        animator = GoL_Animator(lattice_queue, num_runs)

        animator_proc = Process(target=animator.animate)
        animator_proc.start()

    x_list = []; y_list = []
    for i in range(num_runs):
        lattice.evolve()
        if anim:
            lattice_queue.put( (deepcopy(lattice.lattice)) )
        if init_cond == "glider":
            com = lattice.get_CoM()
            if com:
                x, y = com
                x_list.append(x); y_list.append(y)

    if anim:
        animator_proc.join()

    if init_cond == "glider":
        if num_runs > 200:
            x_list = x_list[:150]
            y_list = y_list[:150]
            t_list = list(range(150))
        else:
            t_list = list(range(len(x_list)))

        x_coefs = np.polyfit(t_list, x_list, 1)
        y_coefs = np.polyfit(t_list, y_list, 1)
        print("x velocity: {}".format(x_coefs[0]))
        print("y velocity: {}".format(y_coefs[0]))
        v = np.sqrt(x_coefs[0]**2 + y_coefs[0]**2)
        print("Total velocity: {}".format(v))

        plt.clf()
        plt.title("Glider x Movement")
        plt.xlabel("Timestep")
        plt.ylabel("Glider x position")
        plt.scatter(t_list, x_list)

        m, c = x_coefs
        plt.plot(t_list, [m*x + c for x in t_list], color='r')
        plt.show()

if __name__ == '__main__':
    main()
