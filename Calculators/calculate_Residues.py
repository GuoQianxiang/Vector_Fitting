import sys, os

sys.path.append(os.pardir)
import numpy as np
from scipy.linalg import lstsq
import Drivers.VF_Driver as VFdriver
import Drivers.RP_Driver as RPdriver
import Loader.data_loader as Loader


# 【函数说明】poles残差计算
# 【参数说明】
# - 入参：Zi（n*n*n的矩阵），f0（1*n的list），Nfit（整数）
# - 出参：d（1*n的list），h（1*n的list），r（n*n*n的矩阵），a（1*n的list）
# 【功能说明】
def VF_PolesResidues(Zi, f0, Nfit):
    if len(f0) < 2:
        print("Invalid input for VF_PolesResidues:", f0)
        print('Parameters for Vector Fitting MUST be Multi-Frequency!!!')
        print('Vector Fitting Error')
        return None, None, None, None

    VFopts = {}
    VFopts['asymp'] = 3
    s = 1j * 2 * np.pi * np.array(f0)
    VFopts['plot'] = 0
    VFopts['N'] = Nfit
    VFopts['Niter1'] = 10
    VFopts['Niter2'] = 5

    poles = np.array([])  # Initial poles are automatically generated

    SER = VFdriver.drive(Zi, s, poles, VFopts)
    return SER, VFopts
    # RFopts = {}
    # RFopts['Niter_in'] = 5

    # SER, _ = RPdriver.drive(SER, s, RFopts)

    # r = np.zeros(Nfit)
    # a = np.zeros(Nfit)

    # d = SER['D']
    # h = SER['E']

    # for ik in range(Nfit):
    #     r[ik] = SER['R'][:, :, ik]
    #     a[ik] = SER['poles'][ik]

    # return d, h, r, a


if __name__ == '__main__':
    data, Z0, frq, odc = Loader.load("../Data/data.mat")
    print(Z0.shape)
    print(len(frq))
    SER, _ = VF_PolesResidues(Z0, frq, odc)
    print(SER)
