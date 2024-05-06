import Loader.data_loader as Loader
import sys, os

sys.path.append(os.pardir)
import numpy as np
import scipy.io as io
import Drivers.VF_Driver as VFdriver

if __name__ == "__main__":

    data, Z0, frq, odc = Loader.load("./Data/test.mat")

    VFopts = {'asymp': 3,
              'plot': 0,
              'N': odc,
              'Niter1': 10,
              'Niter2': 5}
    s = 1j * 2 * np.pi * np.array(frq)

    poles = np.array([])  # Initial poles are automatically generated

    SER, rmserr, bigHfit, opts = VFdriver.drive(Z0, s, poles, VFopts)
    io.savemat('./Output/test.mat', SER)

