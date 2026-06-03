"""
    This file contains functions for a smooth USR-SR transition (h=0).
"""


import numpy as np
π = np.pi


def _get_α(η):
    s = np.sqrt(9 - 12*η)
    α = (-6/(s-3) - 1) / 3
    return α


def get_P(ζ, Σ, η):
    """
    PDF of ζ consider USR-SR smooth transition (ϵ=0)
    """
    α = _get_α(η)
    return 1/np.sqrt(2*π*Σ)*np.exp(-(1-np.exp(ζ/α))**2/(2*Σ)*α**2 + ζ/α)


#  def get_ζ_from_G(ζ_G, Σ, η):
    #  α = 1/η - 1/3
    #  ζ = α*np.log(1+ζ_G/α)
    #  return ζ


#  def get_P_from_G(ζ_G, Σ, η):
    #  return get_P_USR_smooth(get_ζ_USR_smooth_from_G(ζ_G, Σ, η), Σ, η)


def get_J_1(x, η):
    α = _get_α(η)
    return 1/(1+x/α)


def get_ζ_G_from_ζ(ζ, η):
    α = _get_α(η)
    return α*(np.exp(ζ/α)-1)
