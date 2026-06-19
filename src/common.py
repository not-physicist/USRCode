"""
    Some common functions/variables to share between submodules.
"""

import numpy as np 
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from time import perf_counter
# import re


Ωcdm = 0.265

def timer(func):
    """
    This function shows the execution time of
    the function object passed
    """
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def check_PDF(x, y):
    """
    check if the probability is conserved
    """
    s = simpson(y, x)
    if np.abs(s-1) < 1e-3:
        print(f"PDF OK: sum={s}")
    else:
        print(f"PDF not OK: sum={s}")


def v_check_PDF(x, Y):
    for y in Y:
        check_PDF(x, y)


def get_zero(x):
    return 0


def get_one(x):
    return 1

# nice_colors = ["firebrick", "darkorange", "seagreen", "cyan", "blueviolet", "orange"]

def convert_sci_not(x, sd=0):
    """
    convert float number to scientific notation in latex 
    if start with 1, omit
    """
    s = f"{x:.{sd}e}"
    s = s.split("e")
    if float(s[0]) == 1.0:
        # double curl-brackets to escape
        return f"10^{{ {int(s[1])} }}"
    else:
        return rf"{s[0]} \cdot 10^{{ {int(s[1])} }}"


def add_fn_suffix(fn, suffix):
    """
    add suffix to filename, just before the .ext
    """
    ss = fn.rsplit(".", 1)

    return ss[0] + "-" + suffix + "." + ss[-1]

def CubicSplineNoExtra(x, y, extra=0.0, v=True):
    """
    wrapper around CubicSpline 
    so that out of range == 0.0
    """
    cs = CubicSpline(x, y)
    if v:
        cs_new = lambda X: np.array([cs(Xi) if Xi >= x[0] and Xi <= x[-1] else extra for Xi in X])
    else:
        def cs_new(i):
            if i >= x[0] and i <= x[-1]:
                return cs(i)
            else:
                return 0.0
    return cs_new


#################################################################################
# mathematical formulas
#################################################################################
def get_M_H(V, N):
    """
    Horizon mass from number of efolds and V_0. Take g_* = 100
    in terms of solar mass
    """
    #  print(V, N)
    # V, N = model.V0, model.NToGo
    H = np.sqrt(V/3)
    # return 4 * np.pi * 2.2e-39 / H * np.exp(2*N)
    return 2.51e-38 / H * np.exp(2*N)

def get_f_gw(V, N):
    """
    GW frequency from efolds and H_inf V0. Take maximal SM g_*
    in hertz
    """
    H = np.sqrt(V/3)
    return H**(1/2) * np.exp(-N) * 3.62e10

def get_Z(V, N):
    M_H = get_M_H(V, N)
    M_H_eq = 2.5e17 
    return 2 / Ωcdm * (M_H / M_H_eq)**(-1/2)
