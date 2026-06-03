"""
This file contains function for calculating and plotting P_C_l
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import quad_vec
from scipy.integrate import simpson

#  from ParaClass import ParaClass
#  from models.gaussian import get_P_G
from .models import USR_only
from .models import USR_smooth
from .models import USR_step
from .models import USR_beta
from .models import USR_full
# from . import common as c

# from julia import Main
#  Main.include("abund/test_2d_distribution.jl")

π = np.pi

def get_gaussian_2D(X, Y, para):
    """return 2D gaussian PDF; assuming zero mean vector

    Parameters
    ----------
    X : array_like, (1,)
        first random variable
    Y : array_like, (1,)
        second random variable
    para : object 'ParaClass'
        provides the variances

    Returns
    -------
    P: array_like, 1D
        the probabilities

    """
    V = np.array([X, Y]).T
    # for each C_l, compute the matrix multiplication;
    # use einsum instead of for-loop; much faster!
    P = 1/(2*π*np.sqrt(para.det)) * \
        np.exp(-np.einsum('ij,jk,ik->i', V, para.inv, V)/2)
    return P

#  @profile
def get_P_integrand(ζ_G, C_l, J1, J2, para):
    """TODO: Docstring for get_P_integrand.

    Parameters
    ----------
    ζ_G : scalar
    C_l : array_like, (1,)
    J1 : callable
    J2 : callable
    para : object 'ParaClass'

    Returns
    -------
    dP : array_like, (1,)

    """
    J1 = J1(ζ_G)
    if np.isnan(J1):
        # in case that J1 contains e.g. sqrt and
        # this particular ζ is out of the domain
        print("NaN J1")
        return 0
    else:
        X = -1/J1 * (3/4*C_l+2*para.XY*J2(ζ_G))
        Y = ζ_G * np.ones(X.shape)
        dP = 3/(4*np.abs(J1)) * get_gaussian_2D(X, Y, para)
        return dP


def get_P_integrand_test(ζ_G, C_l, J1, J2, para):
    """
    identical to get_P_integrand(), except
    J1 as argument of 2D gaussian is one
    """
    # expect ζ_G to be a scalar, C_l to be a 1D array
    J1 = J1(ζ_G)
    if np.isnan(J1):
        # in case that J1 contains e.g. sqrt and
        # this particular ζ is out of the domain
        return 0
    else:
        #  X = -1/c.get_one(ζ_G) * (3/4*C_l+2*para.XY*J2(ζ_G))
        X = -1/J1 * (3/4*C_l+2*para.XY*J2(ζ_G))
        Y = ζ_G * np.ones(X.shape)
        #  dP = 3/(4*np.abs(J1)) * get_gaussian_2D(X, Y, para)
        dP = 3/(4*np.abs(c.get_one(ζ_G))) * get_gaussian_2D(X, Y, para)
        #  print(dP.shape)
        #  print(ζ, J1(ζ), dP)
        return dP


#  @profile
def get_P(C_l, J1, J2, para):
    """
    get PDF of C_l

    Parameters
    ----------
    C_l : array_like
    J1 : callable
    J2 : callable
    para : object 'ParaClass'

    Returns
    -------
    P: array_like

    """
    # use quad seems to give zeros in some cases..
    P, err, infodict = quad_vec(get_P_integrand, -5, 5, epsrel=1e-20,
                                full_output=True, args=(C_l, J1, J2, para))
    #  print(P, err, infodict)
    return P
    # using sampled data
    #  ζ_G = np.linspace(-5, 5, 10000)
    #  dP2 = np.array([get_P_integrand(x, C_l, J1, J2, para) for x in ζ_G])
    #  P2 = simpson(dP2, ζ_G, axis=0)
    #  return P2


def plot_all(para):
    """
    Plot all P_C_l
    """
    C_l = np.linspace(-5, 5, 2000)
    P1 = get_P(C_l, c.get_one, c.get_zero, para)
    P2 = get_P(C_l, lambda x: USR.get_J_1(x), c.get_zero, para)
    P3 = get_P(C_l, lambda x: USR_step.get_J_1(x, para.YY, -1.5), c.get_zero, para)
    P4 = get_P(C_l, lambda x: USR_step.get_J_1(x, para.YY, -6), c.get_zero, para)
    P5 = get_P(C_l, lambda x: USR_smooth.get_J_1(x, para.YY, -0.1), c.get_zero, para)
    P6 = get_P(C_l, lambda x: USR_smooth.get_J_1(x, para.YY, -50), c.get_zero, para)
    P7 = get_P(C_l, lambda x: USR_beta.get_J_1(x, 0), c.get_zero, para)
    P8 = get_P(C_l, lambda x: USR_beta.get_J_1(x, -10), c.get_zero, para)
    P9 = get_P(C_l, lambda x: USR_smooth.get_J_1(x, para.YY, 0.1), c.get_zero, para)
    P10 = get_P(C_l, lambda x: USR_smooth.get_J_1(x, para.YY, 0.5), c.get_zero, para)
    print("Checking PDFs for C_l")
    c.v_check_PDF(C_l, [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10])

    fig, ax = plt.subplots()
    #  ax.plot(C_l, P3, label="upward, $h=-1.5$", c=c.nice_colors[0], ls="--")
    #  ax.plot(C_l, P4, label="upward, $h=-6$", c=c.nice_colors[0])
    ax.plot(C_l, P5, label=r"USR-SR, $\eta_v=-0.1$", c=c.nice_colors[1], ls="--")
    ax.plot(C_l, P6, label=r"USR-SR, $\eta_v=-50$", c=c.nice_colors[1])
    ax.plot(C_l, P7, label=r"USR-SR $\beta, h=0$", c=c.nice_colors[2], ls="--")
    ax.plot(C_l, P8, label=r"USR-SR $\beta, h=-10$", c=c.nice_colors[2])
    ax.plot(C_l, P9, label=r"USR-SR, $\eta_v=0.1$", c=c.nice_colors[3], ls="--")
    ax.plot(C_l, P10, label=r"USR-SR, $\eta_v=0.5$", c=c.nice_colors[3])

    ax.plot(C_l, P1, label="Gaussian", c="k", ls="--")
    ax.plot(C_l, P2, label=r"$\delta N$", c="k")

    ax.set_yscale("log")
    ax.set_ylim((1e-65, 10))
    ax.set_xlim((0, 2))
    ax.set_xlabel(r"$C_l$")
    ax.set_ylabel(r"$P(C_l)$")
    plt.tight_layout()
    plt.legend()
    plt.savefig("./abund/fig/P(C_l).pdf")
    plt.close()

    #  np.savetxt("./dat/P_C_l_G.dat", np.array([C_l, P_G]).T)
    #  np.savetxt("./dat/P_C_l_δN.dat", np.array([C_l, P_USR]).T)


def plot_full(para):
    C_l = np.linspace(-5, 5, 10000)
    P1 = get_P(C_l, c.get_one, c.get_zero, para)
    P2 = get_P(C_l, lambda x: USR_only.get_J_1(x), c.get_zero, para)
    P3 = get_P(C_l, lambda x: USR_full.get_J_1(x, -6, 0), c.get_zero, para)
    P4 = get_P(C_l, lambda x: USR_full.get_J_1(x, -6, -50), c.get_zero, para)
    P5 = get_P(C_l, lambda x: USR_full.get_J_1(x, -0.1, -1), c.get_zero, para)
    P6 = get_P(C_l, lambda x: USR_full.get_J_1(x, -0.1, -10), c.get_zero, para)
    print("Checking PDFs for C_l")
    c.v_check_PDF(C_l, [P1, P2, P3, P4, P5, P6])


    fig, ax = plt.subplots()
    ax.plot(C_l, P1, label="Gaussian", c="k", ls="--")
    ax.plot(C_l, P2, label="USR only", c="k")
    ax.plot(C_l, P3, label=r"USR, $\eta_V=-6, h=0$", c=c.nice_colors[0], ls="--")
    ax.plot(C_l, P4, label=r"USR, $\eta_V=-6, h=-50$", c=c.nice_colors[0])
    ax.plot(C_l, P5, label=r"USR, $\eta_V=-0.1, h=-1$", c=c.nice_colors[1], ls="--")
    ax.plot(C_l, P6, label=r"USR, $\eta_V=-0.1, h=-10$", c=c.nice_colors[1])

    ax.set_yscale("log")
    ax.set_ylim((1e-65, 10))
    ax.set_xlim((0, 2))
    ax.set_xlabel(r"$C_l$")
    ax.set_ylabel(r"$P[C_l]$")
    plt.tight_layout()
    plt.legend(ncol=1, bbox_to_anchor=(0.60, 0.4), fontsize=6)
    plt.savefig("./abund/fig/P(C_l)_full.pdf")
    plt.close()


def plot_integrand_2D(para, dir_n):
    ζ_G = np.linspace(-0.5, 0.5, 500)
    C_l = np.linspace(0, 2, 500)

    #  X, Y = np.meshgrid(ζ_G, C_l)
    dP0 = np.zeros((ζ_G.shape[0], C_l.shape[0]))
    dP1 = np.zeros((ζ_G.shape[0], C_l.shape[0]))

    for i, x in enumerate(ζ_G):
        dP0[:, i] = get_P_integrand(x, C_l, c.get_one, c.get_zero, para)
        dP1[:, i] = get_P_integrand_test(x, C_l, lambda x: USR_only.get_J_1(x),
                                         c.get_zero, para)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    _dP_cont = np.concatenate((dP0, dP1))
    #  print(np.amax(_dP_cont))
    zmin, zmax = 1e-100, +np.amax(_dP_cont)

    c1 = axes[0].pcolormesh(ζ_G, C_l, dP0, vmin=zmin, vmax=zmax,
                            shading="nearest", norm="log", rasterized=True)
    c2 = axes[1].pcolormesh(ζ_G, C_l, dP1, vmin=zmin, vmax=zmax,
                            shading="nearest", norm="log", rasterized=True)
    for ax in axes:
        ax.set_xlabel(r"$\zeta_G$")
        ax.set_ylabel(r"$C_l$")

    fig.colorbar(c1, ax=axes[1])
    fig.savefig(dir_n + "P_C_l_integrand.pdf", bbox_inches="tight")
    plt.close()
