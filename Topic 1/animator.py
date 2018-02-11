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

class Animator():

    def __init__(self, lattice_queue, measure_freq, max_sweeps):

        self.queue = lattice_queue
        self.measure_freq = measure_freq

        data = lattice_queue.get()
        lattice = data[0]
        E_list = data[1]
        M_list = data[2]
        self.step_list = [0]

        self.fig, self.ax_array = plt.subplots(2)
        self.mat = self.ax_array[0].matshow(lattice, vmin=-1, vmax=1)

        self.E_line = Line2D(E_list, self.step_list, color="blue",
            label="Total Energy")
        self.M_line = Line2D(M_list, self.step_list, color="red",
            label="Total Magnetisation")

        self.ax_array[1].set_xlim(0, max_sweeps)
        self.ax_array[1].set_xlabel("No. of Sweeps")
        self.ax_array[1].set_ylim(-2*len(lattice)*len(lattice[0])+1, 100)
        self.ax_array[1].set_ylabel("Total Energy")
        self.ax_array[1].add_line(self.E_line)

        self.ax_array = np.append(self.ax_array, self.ax_array[1].twinx() )
        self.ax_array[2].set_xlim(0, max_sweeps)
        self.ax_array[2].set_ylim(0, len(lattice)*len(lattice[0])+1)
        self.ax_array[2].set_ylabel("Total Magnetisation")
        self.ax_array[2].add_line(self.M_line)


    def update(self, i):
        #logging.debug("Trying to update plot")
        if (self.queue.empty()):
            time.sleep(0.1)
        else:
            data = self.queue.get()
            #print(data)
            if data == "STOP":
                sys.exit("Stopped Animator")
            lattice = data[0]
            E_list = data[1]
            M_list = data[2]
            self.mat.set_data(lattice)
            print(len(data[0]))
            self.step_list.append(self.step_list[-1]+self.measure_freq)
            self.E_line.set_data(self.step_list, E_list)
            self.M_line.set_data(self.step_list, M_list)

    def animate(self):

        anim = animation.FuncAnimation(self.fig, self.update)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        self.ax_array[1].legend()
        self.ax_array[2].legend()
        plt.show()
