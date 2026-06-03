import numpy as np
import matplotlib.pyplot as plt

from ParaClass import ParaClass
from ParaClass import ParaClassMod
#  import models.USR_smooth as USR_smooth
#  import models.USR_full as USR_full

import P_zeta as P_zeta
import P_C_l as P_C_l
#  import f_m as f_m
π = np.pi


def plot_Gow():
    A = 4.05e-3  # to get f_PBH ~ 1
    #  A = 0.25
    Δ = 0.3
    k_star = 1.56e13  # Mpc-1
    k_star_r = 2.74
    K = 1
    C_c = 0.587
    Z = 6.88e16
    para = ParaClass(A, Δ, k_star, k_star_r, K, C_c, Z)

    #  P_zeta.test_USR_full(para)
    #  P_zeta.plot_P_ζ(para)
    #  P_zeta.plot_P_ζ_full(para)
    #  P_C_l.plot_all(para)
    #  P_C_l.plot_full(para)
    #  P_C_l.plot_integrand_2D(para)
    #  P_C_l.plot_2D_test()
    #  f_m.plot_f_m(para)
    #  f_m.get_f_total_all(para)


def plot_Cai():
    # Σ_YY is too large, not realistic ...
    A = 0.25  # Σ_YY ~ 0.02, just as the paper
    Δ = 0.3
    k_star = 1.56e13  # Mpc-1
    k_star_r = 2.74
    K = 1
    C_c = 0.587
    Z = 6.88e16
    para = ParaClass(A, Δ, k_star, k_star_r, K, C_c, Z)

    P_zeta.plot_P_ζ_Cai(para)
    #  P_C_l.plot_all(para)


def plot_test1():
    k_star = 1.56e13  # Mpc-1
    k_star_r = 2.74
    K = 1
    C_c = 0.587
    Z = 6.88e16
    fn_P = "../ms-code/results_mine2/P_k.npz"
    para = ParaClassMod(k_star, k_star_r, K, C_c, Z, fn_P)



if __name__ == '__main__':
    #  plot_Gow()
    #  plot_Cai()
    #  USR_full.test_inverse()
    plot_test1()
