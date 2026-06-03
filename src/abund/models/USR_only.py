"""
    This file contains functions for USR phase w/o transition.
"""


import numpy as np
π = np.pi


def get_P(ζ, Σ):
    '''
    PDF of USR δN
    '''
    return 1/np.sqrt(2*π*Σ)*np.exp(-(1-np.exp(-3*ζ))**2/(18*Σ) - 3*ζ)


#  def get_ζ_from_ζ_G(ζ_G):
    #  """
    #  the classical USR δN local transformation
    #  """
    #  ζ = -1/3*np.log(1-3*ζ_G)
    #  return ζ


# probably not needed
#  def get_P_from_G(ζ_G, Σ):
    #  """
    #  return PDF of USR δN from ζ_G
    #  """
    #  return 1/np.sqrt(2*π*Σ)*np.exp(ζ_G/(6*Σ)) * (1-3*ζ_G)


def get_J_1(x):
    #  return get_J_1(get_P_G(x, Σ), get_P_δN_from_G(x, Σ))
    return 1/(1-3*x)
