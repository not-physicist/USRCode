import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from .get_single import get_Ω

GW_SENS_DN = "data/PLIS/"
GW_SENS_FN = ["plis_BBO.dat", "plis_LISA.dat", "plis_ET.dat"]
GW_EXP_ANNO_COORD = [(1e0, 5e-17), (2.2e-5, 1e-10), (2e0, 1e-9)]
GW_EXP_ANNO_ROT = [55, -52, -90]


def _draw_gw_bounds(ax):
    """
    draw a GW bound curve defined in bound_fn
    """
    # for (i, j) in zip(GW_SENS_FN, GW_EXP_ANNO_COORD):
    for i in range(0, len(GW_SENS_FN)):
        fn = join(GW_SENS_DN, GW_SENS_FN[i])
        sens = np.genfromtxt(fn, comments="#").T
        f = 10**sens[0]
        omegah2 = 10**sens[1]
        ax.plot(f, omegah2, color="k")

        anno = GW_SENS_FN[i].replace("plis_", "").replace(".dat", "")
        ax.text(*GW_EXP_ANNO_COORD[i], anno, rotation=GW_EXP_ANNO_ROT[i])

    # for BBN bound
    ax.fill_between([1e-6, 1e6], 2e-6, 1, color="grey", alpha=0.5, linewidth=0.0)
    ax.annotate("BBN", [1e-3, 3e-6])

def test():
    fig, ax = plt.subplots()

    _draw_gw_bounds(ax)

    ax.plot(*get_Ω("find_N2/h=-20.0-left"))
    ax.plot(*get_Ω("find_N2/h=-20.0-right"))
    
    ax.set_xlim((1e-5, 1e1))
    ax.set_ylim((1e-18, 2e-5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f/\text{Hz}$")
    ax.set_ylabel(r"$\Omega_{\text{GW}, 0}h^2$")

    plt.savefig("find_N2/GW.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    test()
