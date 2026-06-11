from scipy.optimize import toms748
import numpy as np
from scipy.integrate import simpson

from src.abund.ParaClass import ParaClassMod
from . import ms
from .ms import USRManuel
from .common import get_Z, get_M_H, get_zero, timer
from .constants import PEAK_K
from . import abund
from .abund import USR_full


def get_f(model, k, P):
    Z = get_Z(model.V0, model.N2)
    para = ParaClassMod(k, P, Z, model.dn)

    # m in M_H
    m = np.logspace(-3, np.log10(para.m_max), 1000, base=10)[:-1]  # last element gives NaN C_l
    M_H = get_M_H(model.V0, model.N2)  # in M_\odot
    # print("M_H = ", M_H)
    f = abund.get_f(m, para, lambda x: USR_full.get_J_1(x, model.η2, model.h), get_zero)

    return m*M_H, f

def get_f_tot(model, k, P):
    m, f = get_f(model, k, P)
    return simpson(f, np.log(m))


def compute_single(model, GetP=False, GetF=False):
    """
    for a given model object, compute all the "observables"
    GetPmax and GetF are two options, if true tehy will terminate the evaluation
    """
    k, P = ms.save_solve_all(model, False, N_ms=250, log_k_min=-1, log_k_max=3, force_new=True, tosave=False)
    Pmax = np.amax(P)
    print(f"P_max = {Pmax}")

    if GetP:
        return k, P

    m, f = get_f(model, k, P) 
    
    if GetF:
        return m, f

    # TODO: compute the GW singals


@timer
def fix_h(ϵ, N2, ΔN, h_desired):
    """
    get the appropriate β to achieve desired h
    return the model object
    """

    # default βs (initial guess)
    βi = 1e-4
    model = USRManuel(ϵ, βi, N2, ΔN, None, silent=True)
    # use the scaling to get a better guess 
    βi2 = βi * h_desired/model.h
    
    sol, fo = toms748(lambda x: USRManuel(ϵ, x, N2, ΔN, None, silent=True).h - h_desired, 
                      βi2*0.5, βi2*1.5, rtol=1e-5,
                      full_output=True, disp=True)
    # print(fo)
    print(f"Found the right β for h = {h_desired}: β = {sol}")
    return USRManuel(ϵ, sol, N2, ΔN, None, silent=True)

if __name__ == "__main__":
    print(fix_h(1e-4, 25, 2.8, -20))
