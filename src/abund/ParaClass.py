"""
    Classes for computing the variances and storing other parameters
"""

import numpy as np
from scipy.integrate import quad
from scipy.integrate import simpson
# from scipy.integrate import trapezoid
#  import sympy as sp
import matplotlib.pyplot as plt
import pickle
from os.path import join
from scipy.interpolate import CubicSpline
from pathlib import Path

from ..constants import POWER_SPEC_FN
from ..common import timer

π = np.pi
# break points for the spehrical bessel functions; 
# seems that the first few are enough
# j0_zeros = [3.141593, 6.283185, 9.424778, 12.566371, 15.707963]
# j1_zeros = [4.493409, 7.725252, 10.904122, 14.066194, 17.220755]
# TODO: get the zeros up to 1000

j0 = lambda x: np.sin(x) / x
dj0_dz = lambda x: np.cos(x) / x - np.sin(x) / x**2

class ParaClass:
    '''
    Class for computing the variances and storing other parameters.
    '''

    def _get_power_G(self, k):
        """
        return the log-normal power spectrum (lognormal)
        k in unit of k_⋆
        """
        return self.A / (np.sqrt(2 * π) * self.Δ)\
            * np.exp(-np.log(k)**2 / (2 * self.Δ**2))

    def _get_Σ_integrand(self, k, which):
        """
        k is normally a scalar
        """
        dΣ = 1 / k * self._get_power_G(k)
        if which == 1:  # XX
            dΣ *= (k * self.k_star_r * dj0_dz(k * self.k_star_r))**2
        elif which == 2:  # XY
            dΣ *= k * self.k_star_r * dj0_dz(k * self.k_star_r) \
                * j0(k * self.k_star_r)
        elif which == 3:  # YY
            dΣ *= j0(k * self.k_star_r)**2
        else:
            return ValueError("which Σ do you mean?")
        return dΣ
    
    # @timer
    def _calc_Σ(self, which):
        """
        return the variance depends on A and which variance
        naively integrating using discrete method seems very bad
        here use quadrature with the power spectrum interpolated 
        since the bessel functions are (highly) oscillatory
        need to use breakpoint corresponding to the zeros
        """
        f = lambda x: self._get_Σ_integrand(x, which)
        if which == 1:
            p = self.j1z
        elif which == 2:
            p = self.j0z + self.j1z
        else:
            p = self.j0z

        # upper limit of integral = larged zero saved (around 1000)
        ul = np.amax(p)
        # limit for subinteval: must be (much) larger than number of breakt points
        Σ = quad(f, 0, ul, limit=p.shape[0]*20, points=p, epsrel=1e-5)
        print(Σ)
        return Σ[0]
        # TODO: potential optimization (currect ~1s per variance)
        # a wrapper function to breakt down the entire integral according to break points
        # then use multiprocessing to caclulate the different intevals

    def get_m_from_C_l(self, C_l):
        """
        return m in unit of M_H from C_l
        """
        m = self.K * (C_l - 3 / 8 * C_l**2 - self.C_c)**self.γ
        return m

    def find_m_range(self):
        '''
        finding max and min values for C_l and m
        '''
        coeff = [-3 / 8, +1, -self.C_c]
        roots = np.roots(coeff)  # find which C_l <=> m = 0
        C_l_min = roots[1]  # this is the 'smaller' root
        # in this range, f(m) is bijective
        print("C_l ∈", "(", C_l_min, self.C_l_max, ")")
        m_min = self.get_m_from_C_l(C_l_min)
        m_max = self.get_m_from_C_l(self.C_l_max)
        print("m ∈", "(", m_min, m_max, ") M_H")
        return m_min, m_max

    # @timer
    def read_jnz(self):
        """
        read out the pre-saved zeros of bessel functions
        by find_bessel_zeros.py
        """
        fn = "data/jnz.npz"
        if Path(fn).exists():
            data = np.load(fn)
            # print(data)
            return data["j0z"], data["j1z"]
        else:
            raise Exception("jnz.npz not found! You might want to run find_bessel_zeros.py first!")

    def __init__(self, A, Δ, k_star, k_star_r, K, C_c, Z):
        """
        Computing the variance during instantiation
        k_star: k_*
        k_star_r: k_* r
        """
        self.A = A
        self.Δ = Δ
        self.k_star = k_star
        self.k_star_r = k_star_r
        self.K = K
        self.C_c = C_c
        self.Z = Z
        self.γ = 0.36  # this should be universal
        self.C_l_max = 4 / 3

        '''
        # Using SymPy to compute derivatives
        z = sp.symbols("z")
        j0 = sp.sin(z) / z
        self.get_j0 = sp.lambdify(z, j0, "numpy")
        dj0_dz = j0.diff(z)
        self.get_dj0_dz = sp.lambdify(z, dj0_dz, "numpy")
        #  print("Derivative of ", sp.latex(j0), " is ", sp.latex(dj0_dz))
        '''

        # Sigmas
        self.j0z, self.j1z = self.read_jnz()
        self.XX = self._calc_Σ(1)
        self.XY = self._calc_Σ(2)
        self.YY = self._calc_Σ(3)
        Σ = [[self.XX, self.XY], [self.XY, self.YY]]
        self.det = np.linalg.det(Σ)
        self.inv = np.linalg.inv(Σ)
        self.m_min, self.m_max = self.find_m_range()

        # Since we want to remove some attribute in the child class,
        # only print out attributes in the parent class
        if self.__class__.__name__ == 'ParaClass':
            print("Class instantiated.")
            print(vars(self))

    def get_Σ(self, which):
        if which == 1:  # XX
            return self.XX
        elif which == 2:  # XY
            return self.XY
        elif which == 3:  # YY
            return self.YY
        else:
            return ValueError("which Σ do you mean?")

    def set_Σ_XY(self, x):
        self.XY = x


def _rm_invalid(a, b):
    """
    remove NaN entries according to a's value
    """
    bool_array = np.invert(np.isnan(a))
    a = a[bool_array]
    b = b[bool_array]
    # need to change the length of a in the end
    return a, b


class ParaClassMod(ParaClass):
    '''
    Class for computing the variances and storing other parameters
    using numerical power spectrum
    k_star_r
    '''
    def __init__(self, Z, dn):
        """
        the same constructor as the parent, except the additional function
        to store the fn of power spectrum from MS equation
        """
        fn = join(dn, POWER_SPEC_FN)
        k, P = pickle.load(open(fn, "rb"))
        k, P = _rm_invalid(k, P)
        self._P = P
        self._k = k

        # fine peak position
        P_max = np.amax(P)
        index_max = np.array([i for i, j in enumerate(P) if j == P_max])
        #  print(P_max, index_max)
        if index_max.shape[0] == 1:
            k_max = k[index_max[0]]
            # self.N_max = N[index_max[0]]
        else:
            print("Something went wrong with finding maximum!")
            return None
        self._k /= k_max
        # print(f"k_max={k_max}")
        # print(self._k, self._P)

        k_star = 1.56e13  # Mpc-1
        k_star_r = 2.74
        K = 1
        C_c = 0.587
        # run the same code as the parent class constructor
        super().__init__(None, None, k_star, k_star_r, K, C_c, Z)

        # delete attributes
        delattr(self, "A")
        delattr(self, "Δ")

        print("Class instantiated.")
        print(f"k_*={k_star}, k_*_r={k_star_r}, K={K}, C_c={C_c}, Z={Z}, XX={self.XX}, XY={self.XY}, YY={self.YY}")

    def _get_power_G(self, x):
        """
        return the power spectrum from MS equation:
            if in range of computed PS: use interpolation
            if not, return 0
        k in unit of k_s.
        """
        # return np.interp(x, self._k, self._P, left=0.0, right=0.0)
        # return np.exp(np.interp(np.log(x), np.log(self._k), np.log(self._P), left=-np.inf, right=-np.inf))
        cs = CubicSpline(self._k, self._P)
        if x >= np.amin(self._k) and x <= np.amax(self._k):
            return cs(x)
        else:
            return 0.0

    def plot_Σ_int(self, dn):
        fig, ax = plt.subplots()
        
        # X = np.logspace(np.log(np.amin(self._k)), np.log(np.amax(self._k)), 1000)
        X = np.logspace(-2, 3, 1000, base=10)
        ax.plot(self._k, self._P, color="k")
        ax.plot(X, X*self._get_Σ_integrand(X, 1), label="XX")
        ax.plot(X, X*np.abs(self._get_Σ_integrand(X, 2)), label="XY")
        ax.plot(X, X*self._get_Σ_integrand(X, 3), label="YY")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim((1e-6, 5e-2))
        ax.legend()

        plt.savefig(join(dn, "Sigma_int.pdf"), bbox_inches="tight")
        plt.close(fig)
