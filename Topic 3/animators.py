import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )

class Cahn_Hill_Animator():

    def __init__(self, lattice_queue):

        self.queue  = lattice_queue

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


class Cahn_Hill_Animator_After():

    def __init__(self, lattice_list):

        self.lattice_list = lattice_list

        self.fig, self.ax = plt.subplots()
        cont = self.ax.contourf(lattice_list[0], vmin=-1, vmax=1)

    def update(self, i):

        print(i)
        cont = self.ax.contourf(self.lattice_list[i], vmin=-1, vmax=1)
        return cont

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update,
                                        frames=len(self.lattice_list))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

class Poisson_Animator():

    def __init__(self, lattice_queue):

        self.queue  = lattice_queue

        data = lattice_queue.get()

        self.fig, self.ax_array = plt.subplots()
        cont = self.ax_array.contourf(data, vmin=0, vmax=0.5)

    def update(self, i):

        if (self.queue.empty()):
            time.sleep(0.1)
        else:
            data = self.queue.get()
            cont = self.ax_array.contourf(data, vmin=0, vmax=0.5)
            return cont

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

class Poisson_Animator_After():

    def __init__(self, lattice_list):

        self.lattice_list = lattice_list

        self.fig, self.ax = plt.subplots()
        cont = self.ax.contourf(lattice_list[0], vmin=0, vmax=0.5)

    def update(self, i):

        print(i)
        cont = self.ax.contourf(self.lattice_list[i], vmin=0, vmax=0.5)
        return cont

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update,
                                        frames=len(self.lattice_list))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
