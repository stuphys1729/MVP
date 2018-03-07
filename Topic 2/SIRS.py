import math
import time
import logging
import numpy as np
from multiprocessing import Process, Queue
from optparse import OptionParser
import sys
from copy import deepcopy

from animators import SIRS_Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Lattice():
    """
    """

    def __init__(self, x, y, p1, p2, p3, init_conds=None):

        self.x = x
        self.y = y

        self.inf_prob = p1
        self.rec_prob = p2
        self.sup_prob = p3

        if not init_conds:
            self.lattice = np.random.choice( [0, 1], size=(x,y) )

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


    def sweep(self):
        for i in range(self.x*self.y):
            ri = np.random.randint(self.x)
            rj = np.random.randint(self.y)

            self.update(ri, rj)

def main():

    num_runs = 1000

    lattice = Lattice(50, 50, 0.8, 0.1, 0.01)
    lattice_queue = Queue()
    lattice_queue.put( (deepcopy(lattice.lattice)) )

    animator = SIRS_Animator(lattice_queue)

    animator_proc = Process(target=animator.animate)
    animator_proc.start()

    for i in range(num_runs):
        lattice.sweep()
        lattice_queue.put( (deepcopy(lattice.lattice)) )


if __name__ == '__main__':
    main()
