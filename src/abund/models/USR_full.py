import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from . import gaussian as g
from . import USR_smooth

"""
USR-SR exact full
"""


def _get_s(η):
    return np.sqrt(9-12*η)


def _get_α(η, h):
    s = _get_s(η)
    α = 1/3*(-2*η/(s-3)*6/(2*η+h) - 1)
    return α


def _get_β(η, h):
    β = 2*η/(2*η+h)
    return β


def _print_αβ(η, h):
    print(_get_α(η, h), _get_β(η, h))


def _print_domain(η, h):
    """
    return domain of ζ(ζ_G)
    """
    r1 = - _get_α(η, h)/_get_β(η, h)
    r2 = - _get_α(η, h)
    print(r1, r2)


def get_ζ(ζ_G, η, h):
    s = _get_s(η)
    α = _get_α(η, h)
    β = _get_β(η, h)

    return -2/(s-3)*np.log(1+β*ζ_G/α) - 1/3*np.log(1+ζ_G/α)


def _invert_ζ(ζ_G, ζ, η, h):
    """
    try to find ζ_G satisfies this
    """
    return get_ζ(ζ_G, η, h) - ζ


def get_ζ_G(ζ, η, h):
    # ζ should be a vector
    r = np.array([root_scalar(_invert_ζ, args=(x, η, h), x0=-1, x1=0).root for x in ζ])
    return r


def get_dd(ζ, η, h):
    # get the derivative for the chain rule
    # very rudimentary, works probably fine
    eps = 1e-5
    deriv = (get_ζ_G(ζ+eps, η, h) - get_ζ_G(ζ-eps, η, h))/(2*eps)
    return deriv


def get_P(ζ, Σ, η, h):
    """
    deprecated: slow and not in the full range
    """
    P = g.get_P_G(get_ζ_G(ζ, η, h), Σ) * np.abs(get_dd(ζ, η, h))
    print(ζ, P)
    return P


def get_P_interp(ζ, Σ, η, h):
    """
    get the PDF with interpolation
    ζ(ζ_G) is well-behaved enough (monotonically increasing)
    """
    ζ_G_sample = np.linspace(-20, 20, 10000)
    ζ_sample = get_ζ(ζ_G_sample, η, h)

    # add one element to ζ
    ζ = np.append(ζ, ζ[-1]+(ζ[1]-ζ[0]))
    ζ_G = np.interp(ζ, ζ_sample, ζ_G_sample)
    dd = (ζ_G[:-1] - ζ_G[1:])/(ζ[:-1]-ζ[1:])
    P = g.get_P_G(ζ_G[:-1], Σ) * np.abs(dd)

    # get rid of nan, need to be careful!!!
    P = np.where(np.isnan(P), np.zeros(P.shape), P)
    return P


def get_J_1(ζ_G, η, h):
    s = _get_s(η)
    α = _get_α(η, h)
    β = _get_β(η, h)
    return -2/(s-3) * β/(α+β*ζ_G) - 1/3 * 1/(α + ζ_G)


def test_inverse():
    # test ζ_G(ζ) in the case of h=0
    η = -0.2
    h = 0

    X = np.linspace(-5, 5, 1000)
    Y1 = get_ζ_G(X, η, h)
    Y2 = USR_smooth.get_ζ_G_from_ζ(X, η)
    # count how many entries exceed 1e-5 difference
    s = sum(np.abs(x) > 1e-5 for x in (Y1-Y2))
    '''
    if s == 0:
        return None
    else:
        print("Something went wrong")
        return Exception("Something went wrong")
    '''

    # test in case of h=-10
    η = -0.1
    h = -10
    _print_αβ(η, h)
    X1 = np.linspace(0, 5, 10000)
    Y1 = get_ζ_G(X1, η, h)

    Y2 = np.linspace(0, 0.5, 10000)
    X2 = get_ζ(Y2, η, h)

    plt.plot(X1, Y1)
    plt.plot(X2, Y2, ls="--")

    plt.xlabel(r"$\zeta$")
    plt.ylabel(r"$\zeta_G$")
    plt.show()
