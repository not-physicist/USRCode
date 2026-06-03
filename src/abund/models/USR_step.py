"""
    This file contains functions for USR-upward step-SR transitions.
"""


import numpy as np
from scipy.special import erf
import warnings

from .gaussian import get_P_G

π = np.pi
# warning as error
#  warnings.filterwarnings("error")


#  def get_ζ_from_ζ_G(ζ_G, h):
    #  '''
    #  local transformation for upward step from USR to SR
    #  '''
    #  if 1-np.abs(h)*ζ_G >= 0:
        #  ζ = - 2 / np.abs(h) * (np.sqrt(1-np.abs(h)*ζ_G) - 1)
    #  else:
        #  ζ = 0
    #  return ζ


#  def get_P_from_G(ζ_G, Σ, h):
    #  return get_P(get_ζ_from_ζ_G(ζ_G, h), Σ, h)

##############################################################################
# step is dominant
##############################################################################
def get_P(ζ, Σ, h):
    '''
    PDF of upward USR-SR
    Assume ζ, thus P, is an (numpy) array
    '''
    h_abs = np.abs(h)
    #  change the dimnesion of output array, not good for check_PDF()
    #  upper_bound = 2 / h_abs
    #  ζ = ζ[ζ < upper_bound]

    Ω = np.sqrt(2*π*Σ) * (1 + erf(1/(h_abs*np.sqrt(2*Σ))))
    P = (2-h_abs*ζ)/Ω * np.exp(-ζ**2*(4-h_abs*ζ)**2/(32*Σ))

    # get rid of small negative P
    P = np.where(P >= 0, P, np.zeros(P.shape))
    return P


def get_J_1(x, Σ, h):
    # x is ζ_G

    # catch RuntimeWarning from sqrt of neg numbers
    # will be caught in the get_P_integrand() in P_C_l
    with warnings.catch_warnings(category=RuntimeWarning):
        warnings.simplefilter("ignore")
        return 1/np.sqrt(1-np.abs(h)*x)
