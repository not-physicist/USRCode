"""
    This file contains functions to plot PDF of ζ.
"""

import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import scienceplots
plt.style.use("science")

from ..common import v_check_PDF
# from ..common import nice_colors

#  from ParaClass import ParaClass
from .models.gaussian import get_P_G
from .models import USR_only
from .models import USR_smooth
from .models import USR_step
from .models import USR_beta
from .models import USR_full

π = np.pi

def get_mean(x, PDF):
    """
    get the mean from (numerical) PDF
    """
    mean = simpson(x*PDF, x)
    return mean


def get_P_δN_tail(X, Σ):
    return 1/np.sqrt(2*π*Σ) * np.exp(-3*X)


def plot_P_ζ(para):
    """
    deprecated
    """
    X = np.linspace(-5, 5, 10000)
    Y1 = USR_only.get_P(X, para.YY)
    Y2 = get_P_G(X, para.YY)
    Y3 = USR_smooth.get_P(X, para.YY, -0.1)
    Y4 = USR_smooth.get_P(X, para.YY, -6)
    Y5 = USR_smooth.get_P(X, para.YY, -50)
    Y6 = USR_smooth.get_P(X, para.YY, 0.1)
    Y7 = USR_smooth.get_P(X, para.YY, 0.5)
    Y8 = USR_beta.get_P(X, para.YY, 0)
    Y9 = USR_beta.get_P(X, para.YY, -10)
    print("Checking PDFs for ζ")
    c.v_check_PDF(X, [Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9])

    fig, ax = plt.subplots()
    ax.plot(X, Y3, label=r"USR-SR, $\eta_V=-0.1, h=0$", c=c.nice_colors[1], ls="--")
    ax.plot(X, Y5, label=r"USR-SR, $\eta_V=-50, h=0$", c=c.nice_colors[1])
    ax.plot(X, Y6, label=r"USR-SR, $\eta_V=0.1, h=0$", c=c.nice_colors[3], ls="--")
    ax.plot(X, Y7, label=r"USR-SR, $\eta_V=0.5, h=0$", c=c.nice_colors[3])

    ax.plot(X, Y8, label=r"USR-SR $\beta$, $\eta_V = -6, h=0$", c=c.nice_colors[2], ls="--")
    #  ax.plot(X, Y4, label=r"USR-SR, $\eta_V=-6, h=0$", c=c.nice_colors[1], ls="--")
    ax.plot(X, Y9, label=r"USR-SR $\beta$, $\eta_V = -6, h=-10$", c=c.nice_colors[2])
    #  ax.plot(X, Y6, label="$\propto \exp(-3\mathcal{R})$, not conserved")
    # plot these on top
    ax.plot(X, Y1, label=r"Only USR", c="k")
    ax.plot(X, Y2, label="Gaussian", c="k", ls="--")

    ax.set_yscale("log")
    # original axes range
    #  ax.set_ylim((1e-85, 1))
    #  ax.set_xlim((0, 5))
    ax.set_ylim((1e-60, 1))
    ax.set_xlim((0, 1))
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$P(\zeta)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./fig/P(ζ).pdf")

    ax.set_ylim((1e-75, 1e2))
    ax.set_xlim((-1, 2))
    plt.tight_layout()
    plt.savefig("./fig/P(ζ)_zoomed_out.pdf")

    plt.close()


def test_USR_full(para):
    X = np.linspace(-5, 5, 10000)
    Y1 = USR.get_P(X, para.YY)
    Y2 = get_P_G(X, para.YY)
    Y3 = USR_smooth.get_P(X, para.YY, -0.1)
    Y4 = USR_full.get_P(X, para.YY, -0.1, 0)

    print("Checking PDFs for ζ")
    c.v_check_PDF(X, [Y1, Y2, Y3, Y4])

    fig, ax = plt.subplots()
    ax.plot(X, Y3, label="USR-SR smooth $h=0$", c=c.nice_colors[0])
    ax.plot(X, Y4, label="USR-SR full $h=0$", c=c.nice_colors[1], ls="--")
    ax.plot(X, Y1, label=r"$\delta N$", c="k")
    ax.plot(X, Y2, label="Gaussian", c="k", ls="--")

    ax.set_yscale("log")
    ax.set_ylim((1e-60, 1))
    ax.set_xlim((0, 1))
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$P(\zeta)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./fig/P(ζ)_check.pdf")

    plt.close()


def plot_P_ζ_full(para):
    X = np.linspace(-5, 5, 100000)
    Y1 = USR_only.get_P(X, para.YY)
    Y2 = get_P_G(X, para.YY)
    Y3 = USR_full.get_P_interp(X, para.YY, -6, 0)
    Y4 = USR_full.get_P_interp(X, para.YY, -6, -50)
    Y5 = USR_full.get_P_interp(X, para.YY, -20, 0)
    Y6 = USR_full.get_P_interp(X, para.YY, 0.5, 0)
    Y9 = USR_full.get_P_interp(X, para.YY, -0.1, -1)
    Y10 = USR_full.get_P_interp(X, para.YY, -0.1, -10)
    # Y11 = USR_full.get_P_interp(X, para.YY, 0.1, -10)
    # Y12 = USR_full.get_P_interp(X, para.YY, 0.1, -1)
    #  Y13 = USR_full.get_P_interp(X, para.YY, -6, 0)

    print("Checking PDFs for ζ")
    v_check_PDF(X, [Y1, Y2, Y3, Y4, Y5, Y6, Y9, Y10])

    fig, ax = plt.subplots()
    ax.plot(X, Y3, label=r"USR-SR, $\eta_V=-6, h=0$", c=nice_colors[0], ls="--")
    ax.plot(X, Y4, label=r"USR-SR, $\eta_V=-6, h=-50$", c=nice_colors[0])
    ax.plot(X, Y5, label=r"USR-SR, $\eta_V=-50, h=0$", c=nice_colors[2])
    ax.plot(X, Y6, label=r"USR-SR, $\eta_V=0.5, h=0$", c=nice_colors[2], ls="--")
    #  ax.plot(X, Y7, label=r"USR-SR, $\eta=-0.1, h=-10$", c=c.nice_colors[3])
    #  ax.plot(X, Y8, label=r"USR-SR, $\eta=-0.1, h=-1$", c=c.nice_colors[3], ls="--")
    ax.plot(X, Y9, label=r"USR-SR, $\eta_V=-0.1, h=-1$", c=nice_colors[1], ls="--")
    ax.plot(X, Y10, label=r"USR-SR, $\eta_V=-0.1, h=-10$", c=nice_colors[1])
    # ax.plot(X, Y11, label=r"USR-SR, $\eta_V=0.1, h=-10$", c=nice_colors[2], ls="--")
    # ax.plot(X, Y12, label=r"USR-SR, $\eta_V=0.1, h=-1$", c=nice_colors[2])
    #  ax.plot(X, Y13, label=r"USR-SR, $\eta=-6, h=0$", c=c.nice_colors[5], ls="--")
    # plot these on top
    ax.plot(X, Y1, label=r"Only USR", c="k")
    ax.plot(X, Y2, label="Gaussian", c="k", ls="--")

    ax.set_yscale("log")
    # original axes range
    #  ax.set_ylim((1e-85, 1))
    #  ax.set_xlim((0, 5))
    ax.set_ylim((1e-60, 1))
    ax.set_xlim((0, 1))
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$P[\zeta]$")
    plt.legend(ncol=1, bbox_to_anchor=(0.35, 0.45), fontsize=6)
    plt.tight_layout()
    plt.savefig("./abund/fig/P(ζ)_full.pdf")

    plt.close()


def plot_P_ζ_Cai(para):
    X = np.linspace(-5, 10, 1000)
    Y1 = USR.get_P(X, para.YY)
    #  print(Y1[0], Y1[-1])
    Y2 = get_P_G(X, para.YY)
    Y3 = USR_step.get_P(X, para.YY, -1.5)
    Y4 = USR_step.get_P(X, para.YY, -6)
    v_check_PDF(X, [Y1, Y2, Y3, Y4])

    mean1 = get_mean(X, Y1)
    mean2 = get_mean(X, Y2)
    mean3 = get_mean(X[Y3>0], Y3[Y3>0])
    mean4 = get_mean(X[Y4>0], Y2[Y4>0])
    #  print(mean1, mean2, mean3, mean4)

    fig, ax = plt.subplots()
    ax.plot(X-mean1, Y1, label="USR")
    ax.plot(X-mean2, Y2, label="Gaussian")
    ax.plot(X-mean3, Y3, label="upward, $h=-1.5$")
    ax.plot(X-mean4, Y4, label="upward, $h=-6$")

    ax.set_yscale("log")
    #  ax.set_ylim((1e-75, 1))
    #  ax.set_xlim((0, 5))
    ax.set_ylim((1e-8, 10))
    ax.set_xlim((0, 2))
    ax.set_xlabel(r"$\zeta - \langle \zeta \rangle$")
    ax.set_ylabel(r"$P(\zeta)$")
    plt.legend(ncol=1, bbox_to_anchor=(0.3, 1.0), fontsize=6)
    plt.tight_layout()
    plt.savefig("./fig/P(ζ)_Cai.pdf")
    plt.close()
