import math
import time
import logging
import numpy as np
from multiprocessing import Process, Queue
from optparse import OptionParser
import sys
from copy import deepcopy

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

        if not init_cond:
            self.lattice = np.random.choice( [1, 0], size=(x,y) )
        elif init_cond == "glider":
            self.lattice = np.zeroes( (x,y), dtype=np.int8)

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

def main():

    num_runs = 1000

    lattice = Lattice(50, 50)
    lattice_queue = Queue()
    lattice_queue.put( (deepcopy(lattice.lattice)) )

    animator = GoL_Animator(lattice_queue)

    animator_proc = Process(target=animator.animate)
    animator_proc.start()

    for i in range(num_runs):
        lattice.evolve()
        lattice_queue.put( (deepcopy(lattice.lattice)) )


if __name__ == '__main__':
    main()
