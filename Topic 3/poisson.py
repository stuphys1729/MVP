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
from mpl_toolkits.mplot3d import axes3d
from pathlib import Path

from animators import Poisson_Animator as Animator
from animators import Poisson_Animator_After as Post_Animator

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-14s) %(message)s',
                    )


class Lattice():
    """

    """

    def __init__(self, x, y, z, dels, e0, init_cond=None, omega=1):

        self.x = x
        self.y = y
        self.z = z

        self.ds = dels
        self.omega = omega
        self.e0 = e0

        self.rho = np.zeros( (x,y,z) )

        if init_cond == "point":
            self.rho[int(x/2),int(y/2),int(z/2)] = 1
        else:
            sys.exit("Initial conditions not recognised, aborting...")

        self.phi = np.zeros( (x,y,z) )
        self.new_phi = np.zeros( (x,y,z) )


    def update_phi_jacobi(self):
        l = self.phi
        diff = 0
        ds2_e0 = (self.ds**2)/self.e0
        # don't change the boundaries (keep them at 0)
        for i in range(1, self.x-1):
            iup     = (i + 1)
            idown   = (i - 1)
            for j in range(1, self.y-1):
                jup     = (j + 1)
                jdown   = (j - 1)
                for k in range(1, self.z-1):
                    kup     = (k + 1)
                    kdown   = (k - 1)

                    neighbours = [ l[iup,j,k], l[idown,j,k], l[i,jup,k],
                                    l[i,jdown,k], l[i,j,kup], l[i,j,kdown] ]

                    self.new_phi[i,j,k] = (1/6)*(sum(neighbours) + ds2_e0*self.rho[i,j,k])
                    diff += abs(self.new_phi[i,j,k] - self.phi[i,j,k])

        self.phi = copy.deepcopy(self.new_phi)
        return diff


    def update_phi_gauss(self):
        l = self.phi
        ds2_e0 = (self.ds**2)/self.e0
        diff = 0
        # don't change the boundaries (keep them at 0)
        for i in range(1, self.x-1):
            iup     = (i + 1)
            idown   = (i - 1)
            for j in range(1, self.y-1):
                jup     = (j + 1)
                jdown   = (j - 1)
                for k in range(1, self.z-1):
                    kup     = (k + 1)
                    kdown   = (k - 1)

                    neighbours = [ l[iup,j,k], l[idown,j,k], l[i,jup,k],
                                    l[i,jdown,k], l[i,j,kup], l[i,j,kdown] ]

                    temp = (1/6)*(sum(neighbours) + ds2_e0*self.rho[i,j,k])
                    diff += abs(self.phi[i,j,k] - temp)
                    self.phi[i,j,k] = temp

        return diff


    def update_phi_SOR(self, omega=None):
        l = self.phi
        ds2_e0 = (self.ds**2)/self.e0
        diff = 0
        if omega:
            w = omega
        else:
            w = self.omega
        # don't change the boundaries (keep them at 0)
        for i in range(1, self.x-1):
            iup     = (i + 1) % self.x
            idown   = (i - 1) % self.x
            for j in range(1, self.y-1):
                jup     = (j + 1) % self.y
                jdown   = (j - 1) % self.y
                for k in range(1, self.z-1):
                    kup     = (k + 1) % self.z
                    kdown   = (k - 1) % self.z

                    neighbours = [ l[iup,j,k], l[idown,j,k], l[i,jup,k],
                                    l[i,jdown,k], l[i,j,kup], l[i,j,kdown] ]

                    temp = (1-w)*self.phi[i,j,k] + w*(1/6)*(sum(neighbours) + ds2_e0*self.rho[i,j,k])
                    diff += abs(self.phi[i,j,k] - temp)
                    self.phi[i,j,k] = temp

        return diff

    def get_potential_from(self, ri, rj, rk):
        r_list = []
        phi_list = []
        for i in range(1, self.x-1):
            for j in range(1, self.y-1):
                for k in range(1, self.z-1):
                    #(self.ds**3)*
                    r_x = ri - i; r_y = rj - j; r_z = rk - k
                    r = math.sqrt( r_x**2 + r_y**2 + r_z**2 )
                    #r = math.sqrt( ((ri - i)**2 + (rj - j)**2 + (rk - k)**2) )
                    r_list.append(r)
                    phi_list.append(self.phi[i,j,k])

        return r_list, phi_list

    def get_E_from(self, ri, rj, rk):
        r_list = []
        E_list = []
        Ex, Ey, Ez = self.get_E_field(3)
        N = len(Ex)
        for i in range(1, self.x-1):
            for j in range(1, self.y-1):
                for k in range(1, self.z-1):
                    r_x = ri - i; r_y = rj - j; r_z = rk - k
                    r = math.sqrt( r_x**2 + r_y**2 + r_z**2 )
                    #r = math.sqrt( ((ri - i)**2 + (rj - j)**2 + (rk - k)**2) )
                    r_list.append(r)
                    E = math.sqrt( Ex[i-1,j-1,k-1]**2 + Ey[i-1,j-1,k-1]**2 + Ez[i-1,j-1,k-1]**2 )
                    E_list.append(E)

        return r_list, E_list

    def get_E_field(self, d):

        l = self.phi
        ds = self.ds

        if d == 3:

            E_xlist = np.zeros( (self.x-2, self.x-2, self.z-2) )
            E_ylist = np.zeros( (self.x-2, self.x-2, self.z-2) )
            E_zlist = np.zeros( (self.x-2, self.x-2, self.z-2) )

            for i in range(1, self.x-1):
                for j in range(1, self.y-1):
                    for k in range(1, self.z-1):

                        E_xlist[i-1,j-1,k-1] = (l[i+1,j,k] - l[i-1,j,k]) / (2*ds)
                        E_ylist[i-1,j-1,k-1] = (l[i,j+1,k] - l[i,j-1,k]) / (2*ds)
                        E_zlist[i-1,j-1,k-1] = (l[i,j,k+1] - l[i,j,k-1]) / (2*ds)

            return E_xlist, E_ylist, E_zlist

        if d == 2:
            k = int(self.z/2)

            E_xlist = np.zeros( (self.x-2, self.y-2) )
            E_ylist = np.zeros( (self.x-2, self.y-2) )
            mag_list = np.zeros( (self.x, self.y) )

            for i in range(1, self.x-1):
                for j in range(1, self.y-1):

                    Ex = (l[i,j+1,k] - l[i,j-1,k]) / (2*ds)
                    Ey = (l[i+1,j,k] - l[i-1,j,k]) / (2*ds)

                    mag = np.sqrt( (Ex**2 + Ey**2) )
                    mag_list[i,j] = mag

                    # Normalisation
                    E_xlist[i-1,j-1] = Ex / mag
                    E_ylist[i-1,j-1] = Ey / mag

            return E_xlist, E_ylist, mag_list



def main():
    parser = OptionParser("Usage: >> python poisson.py [options] <data_file>")
    parser.add_option("-a", action="store_true", default=False,
        help="Use this option to enable animation")
    parser.add_option("-x", action="store", dest="x", default=50, type="int",
        help="Use this to specify the x-axis size (default: 50)")
    parser.add_option("-y", action="store", dest="y", default=50, type="int",
        help="Use this to specify the y-axis size (default: 50)")
    parser.add_option("-z", action="store", dest="z", default=50, type="int",
        help="Use this to specify the z-axis size (default: 50)")
    parser.add_option("-n", action="store", dest="n_runs", default=10000, type="int",
        help="Use this to specify the maximum number of runs (default: 10000)")
    parser.add_option("-i", action="store", default="point", type="string",
        help="Use this to specify the initial condition")
    parser.add_option("-m", action="store", default="s", type="string",
        help="Use this to specify the update method (default: SOR)")
    parser.add_option("-t", action="store", default=0.001, type="float",
        help="Use this to specify the finish tolerance (default: 0.001)")
    parser.add_option("-w", action="store", default=1.0, type="float",
        help="Use this to specify the value of omega (default: 1.0)")
    parser.add_option("--showplot", action="store_true", default=False,
        help="Use this option along with a data file to plot from")
    parser.add_option("--wplot", action="store_true", default=False,
        help="Use this option to run simulations with varying omega")
    parser.add_option("--index", action="store", default=None, type="int",
        help="Use this to specify only a certain w value from the static list")
    parser.add_option("--makepik", action="store_true", default=False,
        help="Use this option to turn a data file to a pickle file for plotting")

    (options, args) = parser.parse_args()
    if options.showplot:
        with open(args[0], 'rb') as f:
            data = pickle.load(f)
        show_plot(data)
        return

    if options.makepik:
        make_pickle(args[0])
        return

    num_runs = options.n_runs
    anim = options.a
    init_cond = options.i
    tolerance = options.t
    omega = options.w

    x = options.x
    y = options.y
    z = options.z

    if options.wplot:
        run_omegas(x, y, z, init_cond, num_runs, tolerance, options.index)
        return

    lattice = Lattice(x, y, z, 0.1, 1.0, init_cond, omega)

    if options.m == "j":
        method = lattice.update_phi_jacobi
    elif options.m == "g":
        method = lattice.update_phi_gauss
    elif options.m == "s":
        method = lattice.update_phi_SOR
    else:
        raise AttributeError("Method type not identified")

    lattice_queue = None
    if anim:
        lattice_queue = Queue()
        lattice_queue.put( (copy.deepcopy(lattice.phi[:,:,int(z/2)])) )

        animator = Animator(lattice_queue, 0.0008)

        animator_proc = Process(target=animator.animate)
        animator_proc.start()

    taken = run_sim(num_runs, tolerance, method, anim, lattice_queue, lattice)

    print("Converged after {} steps".format(taken))

    r_list, phi_list = lattice.get_potential_from(int(x/2), int(y/2), int(z/2))

    Ex, Ey, mag_list = lattice.get_E_field(2)

    ignore, E_mag_3d = lattice.get_E_from(int(x/2), int(y/2), int(z/2))

    data = [r_list, phi_list, Ex, Ey, mag_list, E_mag_3d]

    file_name = 'data_pos_s{}.pickle'.format(lattice.x)
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Wrote file: " + file_name)

    if anim:
        animator_proc.join()
        plt.clf()

    show_plot(data)

def run_sim(num_runs, tolerance, method, anim, lattice_queue=None, lattice=None):
    for i in range(num_runs):
        diff = method()
        if (i % 5 == 0):
            if anim:
                lattice_queue.put( (copy.deepcopy(lattice.phi[:,:,int(lattice.z/2)])) )
            if lattice != None:
                print("Sweep number {0:8d} | Diff: {1:.02e}".format(i, diff))
        if diff <= tolerance:
            break

    return i+1

def run_omegas(x, y, z, init_cond, num_runs, tolerance, index=None):

    w_list = [1.0, 1.2, 1.4, 1.6, 1.8, 1.85, 1.875, 1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99]

    if index == None:
        taken_list = []
        num_w = len(w_list)

        sys.stdout.write("Running {} different omega values: ".format(num_w))
        sys.stdout.write("[%s]" % (" " * num_w))
        sys.stdout.flush()
        sys.stdout.write("\b" * (num_w+1)) # return to start of line, after '['

        for w in w_list:

            lattice = Lattice(x, y, z, 0.1, 1.0, init_cond, w)

            taken = run_sim(num_runs, tolerance, lattice.update_phi_SOR, False) # No animation
            taken_list.append(taken)

            sys.stdout.write("#")
            sys.stdout.flush()
        sys.stdout.write("\n")

        data = [ w_list, taken_list ]

        file_name = 'data_pos_s{}_w{}.pickle'.format(lattice.x, len(w_list))
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print("Wrote file: " + file_name)

    else:
        w = w_list[index]
        print("Doing w at in index {} with value w={}".format(index, w))
        lattice = Lattice(x, y, z, 0.1, 1.0, init_cond, w)

        taken = run_sim(num_runs, tolerance, lattice.update_phi_SOR, False, None, lattice) # No animation

        out_file = Path("pos_s{}_w.data".format(lattice.x))
        if out_file.is_file(): # We might be the first to write to the file
            open_type = 'a'
        else:
            open_type = 'w'

        with open(out_file, open_type) as f:
            f.write("{} {}\n".format(w, taken))
        print("Wrote to file {} with type \'{}\'".format(out_file.name, open_type))
        return

def make_pickle(filename):

    w_list = []
    taken_list = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.split(" ")
            w_list.append(float(values[0]))
            taken_list.append(int(values[1]))
    data = [ w_list, taken_list ]
    out_file = filename + ".pickle"
    with open(out_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Wrote file: " + out_file)


def show_plot(data):

    if len(data) == 2:
        w_list = data[0]
        taken_list = data[1]

        indices = np.argsort(w_list)
        w_list = [w_list[i] for i in indices]
        taken_list = [taken_list[i] for i in indices]

        best = np.argmin(taken_list)
        print("Best Omega value: {} with {} runs".format(w_list[best], taken_list[best]))

        plt.plot(w_list, taken_list)
        plt.show()

    elif len(data) == 5:
        r_list, phi_list, Ex, Ey, mag_list = data
        print(max(phi_list))
        # potential plot
        """
        plt.scatter(r_list, phi_list)
        ax = plt.gca()
        ax.set_ylim(0, max(phi_list)+max(phi_list)*0.01)
        plt.show()

        plt.clf()
        time.sleep(0.1)
        """

        r_list = [np.log(r) for r in r_list]
        phi_list = [np.log(phi) for phi in phi_list]
        plt.scatter(r_list, phi_list)

        cutoff = 1.5
        grad, intercept = find_gradient(r_list, phi_list, cutoff)
        print("phi r-dependence: {}".format(grad))
        print("(with a cutoff of {})".format(cutoff))
        plt.plot([0, max(r_list)], [intercept, grad*max(r_list) + intercept], color='r')

        plt.show()


        # E-field plot
        plt.clf()
        time.sleep(0.1)

        x = len(Ex)+2; y = len(Ex[0])+2 # Doesn't include the boundaries
        X, Y = np.meshgrid( np.arange(1, x-1),
                            np.arange(1, y-1) )
        max_E = max(mag_list[int(x/2)])
        plt.contourf(mag_list, vmin=0, vmax=max_E, cmap="Oranges")
        plt.quiver(X, Y, Ex, Ey, width=0.001)

        plt.show()

        plt.clf()
        time.sleep(0.1)

        ri = rj = int( (len(Ex)+2) / 2 )
        r_list = []
        E_list = []
        N = len(Ex)
        for i in range(N):
            for j in range(N):
                r_x = ri - i; r_y = rj - j;
                r = math.sqrt( r_x**2 + r_y**2 )
                #r = math.sqrt( ((ri - i)**2 + (rj - j)**2 + (rk - k)**2) )
                r_list.append(r)
                E_list.append(mag_list[i,j])

        r_list = [np.log(r) for r in r_list]
        E_list = [np.log(E) for E in E_list]
        plt.scatter(r_list, E_list)

        cutoff = 1.5
        grad, intercept = find_gradient(r_list, E_list, cutoff)
        print("E r-dependence: {}".format(grad))
        print("(with a cutoff of {})".format(cutoff))
        plt.plot([0, max(r_list)], [intercept, grad*max(r_list) + intercept], color='r')

        plt.show()

    elif len(data) == 6:
        r_list, phi_list, Ex, Ey, mag_list, E_mag_3d = data

        n = len(phi_list)
        print(phi_list)

        r_list = [np.log(r) for r in r_list]
        phi_list = [np.log(phi) for phi in phi_list]
        plt.scatter(r_list, phi_list)

        cutoff = 1.5
        grad, intercept = find_gradient(r_list, phi_list, cutoff)
        print("phi r-dependence: {}".format(grad))
        print("(with a cutoff of {})".format(cutoff))
        plt.plot([0, max(r_list)], [intercept, grad*max(r_list) + intercept], color='r')

        plt.show()


        # E-field plot
        plt.clf()
        time.sleep(0.1)

        x = len(Ex)+2; y = len(Ex[0])+2 # Doesn't include the boundaries
        X, Y = np.meshgrid( np.arange(1, x-1),
                            np.arange(1, y-1) )
        max_E = max(mag_list[int(x/2)])
        plt.contourf(mag_list, vmin=0, vmax=max_E, cmap="Oranges")
        plt.quiver(X, Y, Ex, Ey, width=0.001)

        plt.show()

        plt.clf()
        time.sleep(0.1)


        E_list = [np.log(E) for E in E_mag_3d]
        plt.scatter(r_list, E_list)

        cutoff = 1.5
        grad, intercept = find_gradient(r_list, E_list, cutoff)
        print("E r-dependence: {}".format(grad))
        print("(with a cutoff of {})".format(cutoff))
        plt.plot([0, max(r_list)], [intercept, grad*max(r_list) + intercept], color='r')

        plt.show()

        """ This is the stuff for a 3d plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make the grid
        X, Y, Z = np.meshgrid(np.arange(1, x-1),
                              np.arange(1, y-1),
                              np.arange(1, z-1))

        Ex, Ey, Ez = lattice.get_E_field()

        ax.quiver(X, Y, Z, Ex, Ey, Ez, length=0.5, normalize=True)

        plt.show()
        """

def find_gradient(x_list, y_list, x_cutoff):
    indices = []
    for i in range(len(x_list)):
        if not np.isfinite(x_list[i]):
            continue
        if not np.isfinite(y_list[i]):
            continue
        if x_list[i] < x_cutoff:
            indices.append(i)

    x_data = [x_list[i] for i in indices]
    y_data = [y_list[i] for i in indices]

    coefs = np.polyfit(x_data, y_data, 1)

    return coefs

if __name__ == '__main__':
    main()
