import numpy as np
from scipy import optimize, integrate
from scipy.integrate import solve_ivp
from time import perf_counter
from pathlib import Path
import pickle

# from . import EOM
# from .EOM import _get_f
from .EOM import _get_f
from ..constants import STEP_SIZE_MODELCLASS, SOLVER_MODELCLASS, MODEL_CACHE_FN, EOM_PHI_E_RTOL
from ..common import timer

ns = 0.974 
AR = 2.1e-9
N_tot = 55

#########################################################################################
# first SR phase

def get_V1(x, ϵ, η, V0):
    """
    x = ϕ - ϕᵢ
    """
    return V0 * (1 + np.sqrt(2*ϵ)*x + η/2*x**2)

def get_dV1(x, ϵ, η, V0):
    return V0 * (np.sqrt(2*ϵ) + η*x)

# slow-roll parameters at cmb scale (not to be confused with the ϵ and η parameters in the potential)
def get_ϵV1(x, ϵ, η):
    return 1/2 * ((np.sqrt(2*ϵ) + η*x)/(1 + np.sqrt(2*ϵ)*x + η/2*x**2))**2

def get_ηV1(x, ϵ, η):
    return η/(1 + np.sqrt(2*ϵ)*x + η/2*x**2)

def get_ns(x, ϵ, η):
    """
    tensor to scalar ratio at CMB scale (x = (ϕ_cmb - ϕ_i) / mpl)
    """
    ns = 1 + 2 * get_ηV1(x, ϵ, η) - 6 * get_ϵV1(x, ϵ, η)
    # print("n_s= ", ns)
    return ns

def get_Np(x, ϵ, η):
    """
    using simple integral to get N_+, assume slow-roll (probably fine)
    """
    integrand = lambda y: (get_V1(y, ϵ, η, 1.0)/get_dV1(y, ϵ, η, 1.0))
    
    # print(x, ϵ, η)
    res = integrate.quad(integrand, 0, x)
    Np = res[0]
    # print(Np)
    return Np


def get_ϕcmb_η(ϵ, Np):
    """
    use root finding alg. to find (ϕ_cmb - ϕ_i)/mpl and η at the same time 
    ns is fixed, inputs are ϵ and N_+

    Note that the ϕcmb and η from this are not very accurate.
    It could cause problems for the full solution
    """

    def get_f(x):
        res = [get_ns(x[0], ϵ, x[1]) - ns, get_Np(x[0], ϵ, x[1]) - Np]
        # print(x, res)
        return res

    root = optimize.root(get_f, [1, -0.01], method="lm")
    # root = optimize.root(get_f, [5, 0.01])
    if root.success:
        # print("Found!", root.x)
        # print(root)
        # print(get_ns(root.x[0], ϵ, root.x[1]), get_Np(root.x[0], ϵ, root.x[1]))
        return root.x
    else:
        print("Failure! The desired ϕ_cmb, η is not found!")
        print(root)
        raise ValueError

def get_ϵ_cmb(x, ϵ, η):
    """
    first SR parameter as ϕ_cmb
    """
    y = 1/2 * (np.sqrt(2*ϵ) + η * x)**2
    # print("ϵ at cmb scale: ", y)
    return y

def get_V0(x, ϵ, η): 
    """
    once ϵ, η are determined, can determine the normalization of the potential
    x = (ϕ_cmb - ϕ_i)
    """
    return AR * 24 * np.pi**2 * get_ϵ_cmb(x, ϵ, η)

# @timer
def solve_SR1(ϵ, η, V0, ϕ_cmb_i, step):
    """
    solve the first SR stage: 
        t = N - N_cmb; y = (ϕ - ϕᵢ, π, a)
    """
    _V = lambda x: get_V1(x, ϵ, η, V0)
    _dV = lambda x: get_dV1(x, ϵ, η, V0)
    tspan = [0, 100]
    def get_ϕi(t, y): return y[0]
    get_ϕi.terminal = True
    π_cmb = -np.sqrt(2*get_ϵV1(ϕ_cmb_i, ϵ, η))
    inits = [ϕ_cmb_i, π_cmb, 1.0]
    # print("Initial conditions SR1 ModelClass: ", inits)

    sol = solve_ivp(lambda t, y: _get_f(t, y, _V, _dV), tspan, inits,
                    events=get_ϕi, method=SOLVER_MODELCLASS, max_step=step)    

    if sol.status == 1:
        return sol
    else:
        print(sol.t, sol.y[0])
        raise ValueError("Failure in solve_SR1()!")


####################################################################################### 
# USR phase 

# @timer
def solve_USR(V0, ΔN, πᵢ, step):
    """
    numerically solve USR to get Δphi from V0 and ΔN
    Note Δϕ is defined to be positive
        t = N - N_i; y = (ϕ - ϕᵢ, π, a)
    """
    tspan = [0, 10]  # > 3 would be problematic normally
    initials = [0.0, πᵢ, 1.0]

    def get_term(t, y): return t - ΔN
    get_term.terminal = True
    _V = lambda x: V0 
    _dV = lambda x: 0.0
    sol = solve_ivp(lambda t, y: _get_f(t, y, _V, _dV), tspan, initials, events=get_term, method=SOLVER_MODELCLASS, max_step=step)

    if sol.status == 1:
        # check analytical results
        # print("Analytical: ", sol.y[1, -1]/3*(1-np.exp(3*ΔN)))
        # print("Numerical: ", -sol.y[0, -1])

        # return -sol.y[0, -1], sol.y[1, -1]
        return sol
    else:
        # print(sol.t)
        # print(sol.y[1])
        raise RuntimeError("Failure in solving USR!")

####################################################################################### 
# second SR phase

# potential and its derivative at ϕ < ϕₑ

def get_V2(x, β, phi_e, V0):
        V = 3/(1+2*β) * (x/phi_e)**2 * (1 - 2/3 * (x/phi_e) * (1-β))
        return V * V0

def get_dV2(x, β, phi_e, V0):
        dV = 6/(1+2*β) * x/phi_e**2 * (1 - (1-β)*x/phi_e)
        return dV * V0

# @timer
def solve_SR2(β, ϕₑ, πₑ, V0, step):
    """
    By solving the EOM of inflaton to determine the number of e-fold in the 2nd SR phase
        t = N - N_e; y = (ϕ - ϕₑ, π, a)
    """
    # print(β, ϕₑ, πₑ, V0)

    trange = [0, 50]
    initials = [ϕₑ, πₑ, 1.0]
    
    def get_e1(t, y): return np.abs(y[1]**2/2) - 1
    get_e1.terminal = True
    _V = lambda x: get_V2(x, β, ϕₑ, V0)
    _dV = lambda x: get_dV2(x, β, ϕₑ, V0)
    sol = solve_ivp(lambda t, y: _get_f(t, y, _V, _dV), 
                            trange, initials, events=get_e1, max_step=step, method=SOLVER_MODELCLASS)
    # Since this part (within the root finding) can be quite time consuming,
    # make the step size larger, PS seems fine 

    if sol.status == 1:
        return sol
    else:
        # print(sol.message)
        # raise RuntimeError("solve_SR2() reached the end!")

        # this should not throw an error 
        # if it reached the end, it simply means that the ϕₑ is not correct
        return np.nan

# @timer
def get_ϕₑ(β, πₑ, V0, N2, step):
    """
    vary ϕₑ to get the desired N2
    """
    # print("N2 =", get_N2(β, 10.0, πₑ, self.V0))
    def _f(x):
        sol = solve_SR2(β, x, πₑ, V0, step)
        if sol is np.nan:
            return np.inf
        else:
            return sol.t[-1] - N2

    sol = optimize.root_scalar(_f, bracket=[1, 10], method='brentq', rtol=EOM_PHI_E_RTOL)

    if sol.converged:
        return sol.root 
    else:
        print(sol)
        raise RuntimeError("Finding ϕₑ not successful!")

class USRManuel():
    """
    class for SR-USR-SR potential, modified from the orignal suggested by Manuel
    Δϕ computed from ΔN is not good, need to numerically compute this
    """
    def __init__(self, ϵ, β, N2, ΔN, dn, silent=False, fast=False):
        """
        ϵ, β, N2, ΔN: parameters for the potential 
        silent: if put useful diagnostics to stdout 
        fast: fast excution, the sol_full should not be trusted
        """

        if not silent:
            print(150*"=")

        t0 = perf_counter()

        if ϵ > 1 or ϵ < 0 or β > 1 or β < 0 or N2 < 0 or ΔN < 0 or ΔN > 20:
            raise ValueError("Inputs are bad!")
        
        if dn is not None:
            Path(dn).mkdir(parents=True, exist_ok=True)
         
        if fast:
            stepsize = STEP_SIZE_MODELCLASS * 10
        else:
            stepsize = STEP_SIZE_MODELCLASS

        self.ϵ = ϵ 
        self.β = β
        # N_tot is really the N_tot of SR inflation
        # self.N1 is not the "accurate" N1 !
        self.N1 = N_tot - N2
        self.N2 = N2
        self.ΔN = ΔN
        self.dn = dn

        # first compute the parameter in the 1st SR phase
        self.ϕ_cmb_i, self.η = get_ϕcmb_η(self.ϵ, self.N1)
        # print(self.ϕ_cmb_i, self.η)
        self.V0 = get_V0(self.ϕ_cmb_i, self.ϵ, self.η)
        # the SR approximation at cmb scale
        ϵ_CMB = get_ϵV1(self.ϕ_cmb_i, self.ϵ, self.η)
        # η_CMB = get_ηV1(self.ϕ_cmb_i, self.ϵ, self.η)
        # print("SR parameters at CMB scale: ", ϵ_CMB, η_CMB)
        
        # check the tensor-to-scalar ratio at CMB scale 
        r = 16 * ϵ_CMB 
        # print("Tensor-to-scalar ratio at CMB = ", r)
        if r > 0.056:
            raise ValueError("CMB tensor-to-scalar ratio bound breached: r =", r)

        self.π_cmb = -np.sqrt(2*ϵ_CMB)
        sol1 = solve_SR1(self.ϵ, self.η, self.V0, self.ϕ_cmb_i, stepsize*10)
        self.πᵢ = sol1.y[1][-1]
        # get_ϕcmb_η() is not very accurate (use SR approximation) 
        # the exact N1 is not self.N1
        self.N1_exact = sol1.t[-1]
        # print("N1 from ModelClass.py", sol1.t[-1])
        # print("ϕ_cmb_i from solve_SR1(): ", sol1.y[0][0])

        # USR phase 
        # self.Δϕ, self.πₑ = get_Δϕ_πe(self.V0, self.ΔN, self.πᵢ)
        sol2 = solve_USR(self.V0, self.ΔN, self.πᵢ, stepsize/10)
        self.Δϕ = -sol2.y[0, -1]
        self.πₑ = sol2.y[1, -1]
        # print(self.πᵢ, self.πₑ)
        # print("ΔN from ModelClass.py solver: ", sol2.t[-1])

        # 2nd SR phase
        self.ϕₑ = get_ϕₑ(self.β, self.πₑ, self.V0, self.N2, stepsize*10)
        self.ϕᵢ = self.ϕₑ + self.Δϕ 
        self.ϕ_cmb = self.ϕ_cmb_i + self.ϕᵢ

        # auxiliary quantities 
        # self.h = - 18*np.sqrt(2)/np.sqrt(self.ϵ)*np.exp(3*ΔN)*self.β/self.ϕₑ 
        self.ϵ2 = 18 / self.ϕₑ**2 * (self.β / (1+2*self.β))**2 
        self.η2 = 6 / self.ϕₑ**2 * (2*self.β - 1) / (1 + 2*self.β)
        self.h = 6 * np.sqrt(2*self.ϵ2) / self.πₑ
        
        # stitched up solutions to get the "full solution"
        sol3 = solve_SR2(self.β, self.ϕₑ, self.πₑ, self.V0, stepsize*10)
        # print("N2 from ModelClass.py: ", sol3.t[-1])
        self.sol_full = self.get_sol_full(sol1, sol2, sol3)

        t1 = perf_counter()
        if not silent:
            print("Class initiated:", vars(self))
            print("Initiation took %.2f s" % (t1 - t0))

        # f_NL = 5*self.h*(self.h-self.η2)/(2*(self.h-6)**2)
        # print("Potential SR parameter after USR: ", self.ϵ2, self.η2)
        # print("Smoothness and f_NL: ", self.h,  f_NL)

    def get_V(self, x):
        V = np.where(x > self.ϕₑ, 
                        np.where(x > self.ϕᵢ, 
                                 get_V1(x-self.ϕᵢ, self.ϵ, self.η, 1.0), 
                                 1.0), 
                        get_V2(x, self.β, self.ϕₑ, 1.0))
        return V * self.V0

    def get_dV(self, x):
        dV = np.where(x > self.ϕₑ, 
                        np.where(x > self.ϕᵢ, 
                                 get_dV1(x-self.ϕᵢ, self.ϵ, self.η, 1.0), 
                                 0.0), 
                        get_dV2(x, self.β, self.ϕₑ, 1.0))
        return dV * self.V0
    
    def change_dn(self, new_dn):
        self.dn = new_dn
        Path(self.dn).mkdir(parents=True, exist_ok=True)

    def get_sol_full(self, sol1, sol2, sol3):
        """
        to construct the full sol object from three sol's
        this seems very bad, (the power spectrum from this is not good)
        """
        N = np.append(sol1.t[:-1] - sol1.t[-1] - sol2.t[-1], np.append(sol2.t[:-1] - sol2.t[-1], sol3.t))
        ϕ = np.append(sol1.y[0][:-1] + self.Δϕ + self.ϕₑ, np.append(sol2.y[0][:-1] + self.ϕₑ + self.Δϕ, sol3.y[0]))
        π = np.append(sol1.y[1][:-1], np.append(sol2.y[1][:-1], sol3.y[1]))
        return N, ϕ, π
