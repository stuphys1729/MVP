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

        self.new_phi = np.zeros( (self.x, self.y) )

        self.mu = np.zeros( (x,y) )
