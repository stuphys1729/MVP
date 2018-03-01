import math
import time
import logging
import numpy as np
from multiprocessing import Process, Queue
from optparse import OptionParser
import sys

from animators import SIRS_Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )
