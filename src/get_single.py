from scipy.optimize import toms748
import numpy as np
from scipy.integrate import simpson
from os.path import join
import pickle
from scipy.interpolate import CubicSpline
from pathos.multiprocessing import ProcessingPool as Pool

from src.abund.ParaClass import ParaClassMod
from . import ms
from .ms import USRManuel
from .common import get_Z, get_M_H, get_zero, timer, get_f_gw, add_fn_suffix
from .constants import PEAK_K, MODEL_CACHE_FN, POWER_SPEC_FN, NUM_THREADS
from . import abund
from .abund import USR_full
from . import gw


def get_f(model, k, P):
    Z = get_Z(model.V0, model.N2)
    para = ParaClassMod(k, P, Z, model.dn)

    # m in M_H
    m = np.logspace(-3, np.log10(para.m_max), 1000, base=10)[:-1]  # last element gives NaN C_l
    M_H = get_M_H(model.V0, model.N2+model.ΔN)  # M_H(k_i) in M_\odot
    M_H *= PEAK_K**(-2)
    # print("M_H = ", M_H)
    f = abund.get_f(m, para, lambda x: USR_full.get_J_1(x, model.η2, model.h), get_zero)

    return m*M_H, f

def get_f_tot(model, k, P):
    m, f = get_f(model, k, P)
    return simpson(f, np.log(m))


def compute_single(model, GetP=False, GetF=False, N_ms=250, log_k_min=-1, log_k_max=3):
    """
    for a given model object, compute all the "observables"
    GetPmax and GetF are two options, if true tehy will terminate the evaluation
    """
    k, P = ms.save_solve_all(model, False, N_ms=N_ms, log_k_min=log_k_min, log_k_max=log_k_max, force_new=True, tosave=False)
    # Pmax = np.amax(P)
    # print(f"P_max = {Pmax}")

    if GetP:
        return k, P

    m, f = get_f(model, k, P) 
    
    if GetF:
        return m, f

    # TODO: compute the GW singals


# @timer
def fix_h(ϵ, N2, ΔN, h_desired):
    """
    get the appropriate β to achieve desired h
    return the model object
    """

    # default βs (initial guess)
    βi = 1e-4
    model = USRManuel(ϵ, βi, N2, ΔN, None, silent=True, fast=True)
    # use the scaling to get a better guess 
    βi2 = βi * h_desired/model.h
    
    sol, fo = toms748(lambda x: USRManuel(ϵ, x, N2, ΔN, None, silent=True, fast=True).h - h_desired, 
                      βi2*0.5, βi2*1.5, rtol=1e-5,
                      full_output=True, disp=True)
    # print(fo)
    print(f"Found the right β for h = {h_desired}: β = {sol}")
    return USRManuel(ϵ, sol, N2, ΔN, None, silent=True)


def get_Pmax_DeltaN(ϵ, N2, h_desired):
    """
    return a function which interpolate P(k) against ΔN
    with fixed ϵ, N2, h
    """
    ΔN_all = np.linspace(1.8, 2.4, NUM_THREADS)
    Pmax_all = np.zeros(ΔN_all.shape)
    
    ######################################################
    # first get an interpolator for P_max(ΔN)
    ######################################################
    def _g(x):
        model = fix_h(ϵ, N2, x, h_desired)
        return ms.get_P_max(model)

    with Pool(processes=NUM_THREADS) as pool:
        Pmax_all = np.array(list(pool.imap(_g, ΔN_all)))
    
    print(ΔN_all, Pmax_all)

    # actually P_max(ΔN) follows veyr good power law. But spline seems very simple
    cs = CubicSpline(ΔN_all, Pmax_all, extrapolate=False)

    return cs

def get_P_DeltaN(ϵ, N2, h_desired, cs_Pmax):
    ######################################################
    # compute the full spectrum using ΔN_all[0]
    ######################################################
    # cs = get_Pmax_DeltaN(ϵ, N2, h_desired)

    model = fix_h(ϵ, N2, 1.8, h_desired)
    k, P = compute_single(model, GetP=True, log_k_min=-2, log_k_max=3)

    # it seems that changing N2 changes PEAK_K also.
    # Using the same PEAK_K should be OK
    P_max_full = CubicSpline(k, P)(PEAK_K)
    
    def _get_P_all(x):
        """
        need to get the full power specturm from a given ΔN (from interpolation)
        """
        P_max_desired = cs_Pmax(x)
        P_new = P / P_max_full * P_max_desired
        # print(P_max_desired, np.amax(P), np.amax(P_new))
        return k, P_new
    
    return model, _get_P_all
