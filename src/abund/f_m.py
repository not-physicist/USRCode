"""
    This file contains functions to calculate and plot the PBH abundance from
    some models.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from os.path import exists

from .ParaClass import ParaClass
from . import P_C_l as P_C_l
# from . import common as c
from .models import USR_only
from .models import USR_smooth
from .models import USR_step
from .models import USR_full

# from .common import nice_colors
from .. import common 

π = np.pi

def get_C_l_from_m(m, para):
    if m >= para.m_min and m <= para.m_max:
        coeff = [-3/8, +1, -para.C_c - (m/para.K)**(1/para.γ)]
        roots = np.roots(coeff)
        mask = (roots <= para.C_l_max)  # only need the upper bound
        roots = roots[mask]
        if roots.shape[0] == 1:  # only want one root in the possible range
            #  print(m, roots[0])
            return roots[0]
        else:
            return np.NaN
    else:
        return np.NaN

# @common.timer
def get_f(m, para, J1, J2, dir_n=None):
    '''
    get f(m) from m, which -> which PDF to use;
    expect m to be an array
    '''
    def _get_P_C_l(C_l):
        try:
            return P_C_l.get_P(C_l, J1, J2, para)
        except Exception:
            raise ValueError

    vget_C_l_from_m = np.vectorize(lambda x: get_C_l_from_m(x, para))
    C_l = vget_C_l_from_m(m)

    if dir_n is not None:
        # plot PDF of P_Cl
        fig, ax = plt.subplots()
        ax.plot(C_l, _get_P_C_l(C_l), c="k")
        #  ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$C_l$")
        ax.set_ylabel("$P(C_l)$")
        plt.savefig(dir_n + "P_Cl.pdf", bbox_inches="tight")
        plt.close()
    
    # assume Z is the Z of m=M_H at peak scale 
    # m^(-1/2) gives the scaling of Z
    f = para.Z*para.K*(C_l-3/8*C_l**2-para.C_c)**(para.γ+1)/(para.γ*(1-3/4*C_l)) \
        * _get_P_C_l(C_l)
    return f


def plot_f_m(para):
    m = np.logspace(-3, np.log10(para.m_max), 1000, base=10)[:-1]  # last element gives NaN C_l
    #  make sure the spacing for large m is linear; not necessary!
    #  m1 = np.logspace(-10, -1, 100000, base=10)[:-1]
    #  m2 = np.linspace(0.1, m_max, 1000)[:-1]
    #  m = np.concatenate((m1, m2))

    f_G = get_f(m, para, c.get_one, c.get_zero)
    f_N = get_f(m, para, lambda x: USR.get_J_1(x, para.YY), c.get_zero)
    f_upward1 = get_f(m, para, lambda x: USR_step.get_J_1(x, para.YY, -1.5), c.get_zero)
    f_upward2 = get_f(m, para, lambda x: USR_step.get_J_1(x, para.YY, -6), c.get_zero)
    f_USR_smooth1 = get_f(m, para, lambda x: USR_smooth.get_J_1(x, para.YY, -0.1), c.get_zero)
    f_USR_smooth2 = get_f(m, para, lambda x: USR_smooth.get_J_1(x, para.YY, -50), c.get_zero)

    fig, ax = plt.subplots()
    ax.plot(m*para.k_star_r**2, f_G, label="Gaussian")
    ax.plot(m*para.k_star_r**2, f_N, label=r"$\delta N$")
    ax.plot(m*para.k_star_r**2, f_upward1, label=r"upward $h=-1.5$")
    ax.plot(m*para.k_star_r**2, f_upward2, label=r"upward $h=-6$")
    ax.plot(m*para.k_star_r**2, f_USR_smooth1, label=r"USR-SR $\eta_V=-0.1$")
    ax.plot(m*para.k_star_r**2, f_USR_smooth2, label=r"USR-SR $\eta_V=-50$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e1)
    #  ax.set_ylim(1e-15, 1e2)
    ax.set_xlabel(r"$m/M_{k_*}$")
    ax.set_ylabel(r"$f(m)$")
    plt.tight_layout()
    plt.legend()
    plt.savefig("./fig/f(m).pdf")
    plt.close()

    ###############################################################################
    # f_PBH
    ###############################################################################

    # data points are presampled for plot already
    f_PBH_G = simpson(f_G, np.log(m))
    print('f_PBH with Gaussian PDF: ', f_PBH_G)
    f_PBH_N = simpson(f_N, np.log(m))
    print('f_PBH with puer USR PDF: ', f_PBH_N)
    f_PBH_up1 = simpson(f_upward1, np.log(m))
    print('f_PBH with USR with step h=-1.5: ', f_PBH_up1)
    f_PBH_up2 = simpson(f_upward2, np.log(m))
    print('f_PBH with USR with step h=-6: ', f_PBH_up2)
    f_PBH_USR_smooth1 = simpson(f_USR_smooth1, np.log(m))
    print('f_PBH with smooth USR η=-0.1: ', f_PBH_USR_smooth1)
    f_PBH_USR_smooth2 = simpson(f_USR_smooth2, np.log(m))
    print('f_PBH with smooth USR η=-0.1: ', f_PBH_USR_smooth2)

    fig, ax = plt.subplots()
    ax.plot(m*para.k_star_r**2, f_G/f_PBH_G, label="Gaussian")
    ax.plot(m*para.k_star_r**2, f_N/f_PBH_N, label=r"$\delta N$")
    ax.plot(m*para.k_star_r**2, f_upward1/f_PBH_up1, label="upward $h=-1.5$")
    ax.plot(m*para.k_star_r**2, f_upward2/f_PBH_up2, label="upward $h=-6$")
    ax.plot(m*para.k_star_r**2, f_USR_smooth1/f_PBH_USR_smooth1, label="USR-SR $\eta_V=-0.1$")
    ax.plot(m*para.k_star_r**2, f_USR_smooth2/f_PBH_USR_smooth2, label="USR-SR $\eta_V=-50$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-15, 1e2)
    ax.set_xlabel(r"$m/M_{k_*}$")
    ax.set_ylabel(r"$f(m)$")
    plt.tight_layout()
    plt.legend()
    plt.savefig("./fig/f(m)_f_pbh.pdf")
    plt.close()


def get_f_total(para, J1, J2, dir_n=None):
    # last element gives NaN C_l
    m = np.logspace(-3, np.log10(para.m_max), 10000, base=10)[:-1]
    f = get_f(m, para, J1, J2, dir_n)
    #  print(f)

    if dir_n is not None:
        # plot spectrum
        fig, ax = plt.subplots()
        ax.plot(m, f, c="k")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$m/M_H$")
        ax.set_ylabel("$f(m)$")
        plt.savefig(dir_n + "f.pdf", bbox_inches="tight")
        plt.close()

    f_tot = simpson(f, np.log(m))
    #  print('f_PBH: ', f_tot)
    return f_tot


def get_f_total_all(var):
    f_G = get_f_total(var, c.get_one, c.get_zero)
    f_N = get_f_total(var, lambda x: USR.get_J_1(x, var.YY), c.get_zero)
    f_upward1 = get_f_total(var, lambda x: USR_step.get_J_1(x, var.YY, -1.5), c.get_zero)
    f_upward2 = get_f_total(var, lambda x: USR_step.get_J_1(x, var.YY, -6), c.get_zero)
    f_USR_smooth1 = get_f_total(var, lambda x: USR_step.get_J_1(x, var.YY, -0.1), c.get_zero)
    #  f_USR_smooth2 = get_f_total(var, lambda x: USR_step.get_J_1(x, var.YY, -50), c.get_zero)
    return f_G, f_N, f_upward1, f_upward2, f_USR_smooth1


def plot_f_A():
    """
    plot f_PBH against A
    """
    A = np.logspace(-3, -1, base=10, num=50)
    f = np.zeros([A.shape[0], 5])

    fn = "f.npy"
    if exists(fn):
        f = np.load(fn)
    else:
        var = [ParaClass(x) for x in A]
        for i, var_i in enumerate(var):
            print(f"{i}/{A.shape[0]}", end='\r')
            f[i] = np.array([A[i], *get_f_total_all(var_i)])
            #  print(f[i])
        np.save(fn, f)

    fig, ax = plt.subplots()
    ax.plot(A, f[:, 1], label="Gaussian")
    ax.plot(A, f[:, 2], label=r"$\delta N$")
    ax.plot(A, f[:, 3], label="upward $h=-1.5$")
    ax.plot(A, f[:, 4], label="upward $h=-6$")
    ax.plot(A, f[:, 5], label=r"USR-SR $\eta_V=0.1$")

    ax.set_xlabel("$A$")
    ax.set_ylabel(r"$f_{\rm {PBH}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e-1)
    ax.set_ylim(1e-30, 1e20)

    plt.legend()
    plt.tight_layout()
    plt.savefig("./fig/f_A.pdf")

def plot_f_eta(para):
    """
    eta_V's effects on PBH mass spectrum
    """
    m = np.logspace(-3, np.log10(para.m_max), 1000, base=10)[:-1]  # last element gives NaN C_l

    f_G = get_f(m, para, c.get_one, c.get_zero)
    f_N = get_f(m, para, lambda x: USR_only.get_J_1(x), c.get_zero)
    f_1 = get_f(m, para, lambda x: USR_full.get_J_1(x, -0.1, -10), c.get_zero) 
    f_2 = get_f(m, para, lambda x: USR_full.get_J_1(x, -0.001, -10), c.get_zero) 
    f_3 = get_f(m, para, lambda x: USR_full.get_J_1(x, -0.1, -20), c.get_zero) 
    f_4 = get_f(m, para, lambda x: USR_full.get_J_1(x, -0.001, -20), c.get_zero) 

    fig, ax = plt.subplots()
    ax.plot(m*para.k_star_r**2, f_G, label="Gaussian", c="k", ls="--")
    ax.plot(m*para.k_star_r**2, f_N, label=r"$\delta N$", c="k")
    ax.plot(m*para.k_star_r**2, f_1, label=r"USR-SR $\eta_V = -0.1, h=0$", c=nice_colors[0], alpha=0.4)
    ax.plot(m*para.k_star_r**2, f_2, label=r"USR-SR $\eta_V = -0.001, h=0$", c=nice_colors[0], ls="--")
    ax.plot(m*para.k_star_r**2, f_3, label=r"USR-SR $\eta_V=-0.1, h=-20$", c=nice_colors[1], alpha=0.4)
    ax.plot(m*para.k_star_r**2, f_4, label=r"USR-SR $\eta_V=-0.001, h=-20$", c=nice_colors[1], ls="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e1)
    #  ax.set_ylim(1e-15, 1e2)
    ax.set_xlabel(r"$m/M_H$")
    ax.set_ylabel(r"$f(m)$")
    plt.tight_layout()
    plt.legend(fontsize=6)
    plt.savefig("./abund/fig/f(m)_eta.pdf")
    plt.close()

