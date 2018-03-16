import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Cahn_Hill_Animator():

    def __init__(self, lattice_queue, frames):

        self.queue  = lattice_queue
        self.frames = frames

        data = lattice_queue.get()

        self.fig, self.ax_array = plt.subplots()
        cont = self.ax_array.contourf(data, vmin=-1, vmax=1)

    def update(self, i):

        if (self.queue.empty()):
            time.sleep(0.1)
        else:
            data = self.queue.get()
            cont = self.ax_array.contourf(data, vmin=-1, vmax=1)
            return cont

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()