"""
    This file contains functions for USR-SR with η=-6 and arbitrary h.
"""


import numpy as np


def get_P(x, Σ, h):
    η = -6
    β = 2*η/(2*η+h)
    α = -β/3 - 1/3
    sq = np.sqrt((β-1)**2+4*β*np.exp(-3*x))
    return 1/np.sqrt(2*np.pi*Σ)*np.exp(-1/(2*Σ)*(α/(2*β))**2*(-1-β+sq)**2 - 3*x) \
        * np.abs(3*α)/sq


def get_J_1(ζ_G, h):
    η = -6
    β = 2*η/(2*η+h)
    α = -β/3 - 1/3
    J1 = -1/3 * (β/(α+β*ζ_G) + 1/(α+ζ_G))
    return J1
