"""
    This file contains Gaussian PDF.
"""

import numpy as np

π = np.pi

def get_P_G(ζ, Σ):
    """
    return Gaussian PDF of ζ
    """
    return 1/np.sqrt(2*π*Σ)*np.exp(-ζ**2/(2*Σ))
