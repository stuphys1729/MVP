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
            self.phi = np.random.uniform(-0.1, 0.1, size=(x,y) )
            for i in range(self.x):
                for j in range(self.y):
                    self.phi[i,j] += init_cond
        else:
            sys.exit("Initial conditions not recognised, aborting...")

        self.new_phi = np.zeros( (self.x, self.y) )

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


    def update(self):

        for i in range(self.x):
            for j in range(self.y):
                self.update_mu(i,j)

        for i in range(self.x):
            for j in range(self.y):
                self.update_phi(i,j)

        self.phi = copy.deepcopy(self.new_phi)



def main():

    num_runs = 10000

    lattice = Lattice(100, 100, 1.0, 2.0, 0.1, 0.1, 0.1)

    lattice_queue = Queue()
    lattice_queue.put( (copy.deepcopy(lattice.phi)) )

    animator = Animator(lattice_queue, num_runs)

    animator_proc = Process(target=animator.animate)
    animator_proc.start()

    for i in range(num_runs):
        lattice.update()
        if (i % 10 == 0):
            lattice_queue.put( copy.deepcopy(lattice.phi) )

    return

if __name__ == '__main__':
    main()
