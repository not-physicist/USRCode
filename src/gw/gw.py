import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.interpolate import CubicSpline

from tqdm import tqdm
import pickle

from pathos.multiprocessing import ProcessingPool as Pool

from collections.abc import Callable

from ..constants import NUM_THREADS
from ..common import CubicSplineNoExtra

# use the formula in Domenech 2021 to calculate GW spectrum
# from scalar power spectrum
# focus on radiation domination (w=c_s^2=1/3)

###############################################################################


# EOS and sound speed
w = 1 / 3
cs2 = 1 / 3
b = 0

# assume SM degrees of freedom
c_g = 0.4
Ωr0 = 1.62e-5
h = 0.67

'''
def _get_y(u, v):
    return 1 - (1 - cs2 * (u - v)**2) / (2 * cs2 * u * v)


def _get_Transfer_rd(u, v):
    """
    Transfer function in radiation domination
    Assume u and v are scalars
    """
    y = _get_y(u, v)

    # first factor
    A = y**2 / (3 * cs2**2)
    # second factor
    B = ((4 * v**2 - (1 - u**2 + v**2)**2) / (4 * u**2 * v**2))**2
    # third factor
    C1 = np.pi**2 / 4 * y**2 * np.heaviside(np.sqrt(cs2) * (u + v) - 1, 1)
    C2 = (1 - y / 2 * np.log(np.fabs((1 + y) / (1 - y))))**2
    return A * B * (C1 + C2)


def _get_Ω_rh(k, _get_P_R):
    """
    GW energy parameter at early Universe
    """
    def __integrand(u, v, k):
        return _get_Transfer_rd(u, v) * _get_P_R(k * u) * _get_P_R(k * v)

    def __upper_b(v):
        return 1 + v

    def __lower_b(v):
        return np.fabs(1 - v)

    res = dblquad(__integrand, 0, np.inf, __lower_b, __upper_b, args=(k,),
                  epsrel=1e-12)
    print(res)
    return res[0]


def _get_Ω0(k, _get_P_R):
    """
    Today's GW energy parameter
    assume standard values for radiation energy, effective degrees of freedom
    """
    return Ωr0 * _get_Ω_rh(k, _get_P_R)
'''

###############################################################################
# Espinosa et al formula
###############################################################################
'''
def _get_I_c(d, s):
    r = -36 * np.pi * (s**2 + d**2 - 2)**2 / (s**2 - d**2)**3
    r *= np.heaviside(s - 1, 1)
    return r


def _get_I_s(d, s):
    r = -36 * (s**2 + d**2 - 2) / (s**2 - d**2)**2
    r *= ((s**2 + d**2 - 2) / (s**2 - d**2)
          * np.log((1 - d**2) / np.fabs(s**2 - 1)) + 2)
    return r


def get_Ω0_Esp(k, get_P_R):
    Omega = c_g * Ωr0 / 36

    def __integrand(s, d, k):
        i = ((d**2 - 1 / 3) * (s**2 - 1 / 3) / (s**2 - d**2))**2
        i *= get_P_R(np.sqrt(3) / 2 * k * (s + d))
        i *= get_P_R(np.sqrt(3) / 2 * k * (s - d))
        i *= (_get_I_c(d, s)**2 + _get_I_s(d, s)**2)
        return i

    def _lower_b(x):
        return 1 / np.sqrt(3)

    def _upper_b(x):
        return np.inf

    #  res = dblquad(__integrand, 0, 1 / np.sqrt(3), _lower_b, _upper_b,
                  #  args=(k,), epsrel=1e-15)
    #  print(res)
    #  Omega *= res[0]

    def __first_int(d, k):
        i = quad(lambda x: __integrand(x, d, k), 1 / np.sqrt(3), np.inf, epsabs=1e-20, limit=200)
        return i[0]

    res = quad(lambda x: __first_int(x, k), 0, 1 / np.sqrt(3), epsabs=1e-20, limit=200)
    print(res)
    Omega *= res[0]
    return Omega
'''

# Transfer function in radiation domination;
# Splitting the integral for better convergence speed;
# Write the Transfer function in two domains improve speed a little bit

def _get_T_rd1(d, s):
    pref = 12*(d**2-1)**2 * (s**2-1)**2 * (d**2+s**2-6)**4 / (s**2 - d**2)**8
    term1 = ( np.log((3-d**2)/(-s**2+3)) + 2*(s**2-d**2)/(d**2 + s**2 - 6) )**2
    return pref*term1

def _get_T_rd2(d, s):
    pref = 12*(d**2-1)**2 * (s**2-1)**2 * (d**2+s**2-6)**4 / (s**2 - d**2)**8
    term1 = ( np.log((3-d**2)/(s**2-3)) + 2*(s**2-d**2)/(d**2 + s**2 - 6) )**2
    term2 = np.pi**2
    return pref*(term1+term2)


def _get_Ω_rh(k: float, P: Callable) -> float:
    """
    gw energy parameter at production
    expect k to be a scalar
    """
    def integrand1(s: float, d: float) -> float:
        return _get_T_rd1(d, s) * P(k*(s+d)/2) * P(k*(s-d)/2)

    def integrand2(s: float, d: float) -> float:
        return _get_T_rd2(d, s) * P(k*(s+d)/2) * P(k*(s-d)/2)

    Ω1 = dblquad(integrand1, 0, 1, 1, np.sqrt(3))[0]
    Ω2 = dblquad(integrand2, 0, 1, np.sqrt(3), np.inf)[0]

    return Ω1 + Ω2

def get_Ω(k, P):
    """
    current day GW energy parameter 
    use multiprocessing to compute various k's
    """
    # k, P = pickle.load(open(dn + "PS.dat", "rb"))

    if k.shape != P.shape:
        raise IndexError("Inputs (k and P) are of different dimension!")
    
    # Default extrapolation behaviour: extrapolate, can go the negative P 
    # here: outof range k <=> no contribution, set to 0.0
    get_P = CubicSplineNoExtra(k, P, v=False)

    def _get_Ω(x):
        return c_g * Ωr0 *_get_Ω_rh(x, get_P) * h**2

    with Pool(processes=NUM_THREADS) as pool:
        Ω = np.array(list(tqdm(pool.imap(_get_Ω, k), total=len(k))))
    
    return Ω
