import numpy as np 
from os.path import join
import matplotlib.pyplot as plt
import pickle
from pathlib import Path 
from scipy.integrate import simpson
from scipy.optimize import toms748
from scipy.interpolate import CubicSpline
from pathos.multiprocessing import ProcessingPool as Pool

from src.abund.ParaClass import ParaClassMod
from . import ms
from .ms import USRManuel, get_P_max
from .common import get_Z, get_M_H, get_zero
from .abund import USR_full, get_f
from. import get_single
from .constants import NUM_THREADS, POWER_SPEC_FN, BH_F_FN, PEAK_K, MODEL_CACHE_FN

import scienceplots
plt.style.use("science")

PBH_CONST_DN = "data/PBHbounds/bounds"
PBH_CONST_LIST = ["Evaporation.txt", "EROS.txt", "HSC.txt", "K.txt", "M.txt", "CMB.txt"]
PBH_CONST_COORD = [[2e-18, 1e-1], [2e-4, 2.2e-2], [3e-10, 3e-2], [2e-10, 5e-1], [2e-2, 5e-1], [1e2, 1e-3]]
PBH_CONST_ROT = [-90, 0, -90, 0, 0, -90]
PBH_CONST_ENVL_FN = "data/PBHbounds_all.pkl"

DELTA_N_RTOL = 1e-7
N2_RTOL = 1e-5

FIND_N2_DN = "find_N2/"

def comp_envelope_all():
    """
    compute the envelope of all (relevant) PBH constraints
    """
    log_m = np.linspace(-19, 4, 10000)
    log_f = np.ones(log_m.shape) * 10
    for fn in PBH_CONST_LIST:
        mi, fi = np.loadtxt(join(PBH_CONST_DN, fn)).T

        log_f_intp = np.interp(log_m, np.log10(mi), np.log10(fi), left=10, right=10)
        log_f = np.where(log_f_intp < log_f, log_f_intp, log_f)

    return log_m, log_f

def save_envelope_all():
    """
    save the envelope of all (relevant) PBH constraints
    """
    log_m, log_f = comp_envelope_all()
    with open(PBH_CONST_ENVL_FN, "wb") as f:
        pickle.dump((log_m, log_f), f)

def get_envelope_all():
    """
    read the envelope of all (relevant) PBH constraints
    """
    with open(PBH_CONST_ENVL_FN, "rb") as f:
        log_m, log_f = pickle.load(f)

    return log_m, log_f

def check_constraint(m, f):
    r"""
    check if the input m (in M_\odot), f
    is consistent with the existing constraints
    """
    # note: envelope returns quantities in log10 !
    log_m_constr, log_f_constr = get_envelope_all()
    f_max_itp = 10**np.interp(np.log10(m), log_m_constr, log_f_constr)

    integral = simpson(f/f_max_itp/m, m)
    return integral

def plot_constraints(ax):
    """
    plot the __individual__ constraint lines on the ax
    """
    for i in range(0, len(PBH_CONST_LIST)):
        fn = join(PBH_CONST_DN, PBH_CONST_LIST[i])
        mi, fi = np.loadtxt(fn).T
        ax.plot(mi, fi, c="k")
        ax.text(*PBH_CONST_COORD[i], PBH_CONST_LIST[i].replace(".txt", ""), va="center_baseline", rotation=PBH_CONST_ROT[i])

    ax.fill_between([1e-20, 1e5], 1, 1000, color="gray", alpha=0.3, linewidth=0.0)


def find_entire_DM(ϵ, N2, h_desired, dn=None):
    """
    find the set of parameter to have f_tot = 1
    """
    ΔN_all = np.linspace(2.2, 2.5, NUM_THREADS)
    Pmax_all = np.zeros(ΔN_all.shape)
    
    ######################################################
    # first get an interpolator for P_max(ΔN)
    ######################################################
    def _g(x):
        model = get_single.fix_h(ϵ, N2, x, h_desired)
        return ms.get_P_max(model)

    with Pool(processes=NUM_THREADS) as pool:
        Pmax_all = np.array(list(pool.imap(_g, ΔN_all)))
    
    print(ΔN_all, Pmax_all)

    # actually P_max(ΔN) follows veyr good power law. But spline seems very simple
    cs = CubicSpline(ΔN_all, Pmax_all, extrapolate=False)
    
    ######################################################
    # compute the full spectrum using ΔN_all[0]
    ######################################################
    model = get_single.fix_h(ϵ, N2, ΔN_all[0], h_desired)
    k, P = get_single.compute_single(model, GetP=True)

    # it seems that changing N2 changes PEAK_K also.
    # Using the same PEAK_K should be OK
    P_max_full = CubicSpline(k, P)(PEAK_K)
    
    ######################################################
    # use root finding to find the right ΔN to give f_tot = 1
    ######################################################
    def _get_P_all(x):
        """
        need to get the full power specturm from a given ΔN (from interpolation)
        """
        P_max_desired = cs(x)
        P_new = P / P_max_full * P_max_desired
        # print(P_max_desired, np.amax(P), np.amax(P_new))
        return k, P_new

    def _f(x):
        """
        get ftot from x=ΔN
        """
        k, P = _get_P_all(x)

        # note that changing ΔN doesn't affect the scale of USR,
        # hence Z should be the same for all ΔN 
        # and no need to get new model object everytime (or run fix_h() )
        # model = get_single.fix_h(ϵ, N2, x, h_desired)
        
        ftot = get_single.get_f_tot(model, k, P)
        # print(f"ftot = {ftot}")

        # ftot exponential sensitive in ΔN; use log for faster convergence
        return np.log(ftot)

    sol, fo = toms748(_f, ΔN_all[0], ΔN_all[-1], rtol=DELTA_N_RTOL, full_output=True, disp=True)
    model = get_single.fix_h(ϵ, N2, sol, h_desired)

    ######################################################
    # now the "correct" ΔN has been found 
    ######################################################
    k, P = _get_P_all(sol)
    # use the "universal" model object, just like in _f()
    # model = get_single.fix_h(ϵ, N2, sol, h_desired)
    m, f = get_single.get_f(model, k, P)
    
    ftot = simpson(f, np.log(m))
    print(f"Found the correct ΔN with ftot={ftot}")
    print(fo)

    log_int = np.log(check_constraint(m, f))
    print(f"N2 = {N2}, log(integral) = {log_int}")
    
    # to save stuff into cache
    if dn is not None:
        # if need β and other stuff, run get_single.fix_h()
        # paras = [ϵ, N2, sol]
        
        Path(dn).mkdir(parents=True, exist_ok=True)

        np.savez(join(dn, POWER_SPEC_FN), k=k, P=P)
        np.savez(join(dn, BH_F_FN), m=m, f=f)

        with open(join(dn, MODEL_CACHE_FN), "wb") as f:
            pickle.dump(model, f)

    return m, f


def find_N2_window(ϵ, h_desired, bracket, dn):
    """
    find the range for N_- 
    so that the mass function sit inside the asteroid window
    """
    def _f(x):
        print("="*100)
        print(f"Trying with N2 = {x}")
        m, f = find_entire_DM(ϵ, x, h_desired)
        log_int = np.log(check_constraint(m, f))
        # print(f"N2 = {x}, log(integral) = {log_int}")
        # ftot exponential sensitive in ΔN; use log for faster convergence
        return log_int
    
    fn_ps = join(dn, POWER_SPEC_FN)
    fn_bh = join(dn, BH_F_FN)
    if Path(fn_ps).exists() and Path(fn_bh).exists():
        print("Found cache files. SKIPPING...")
        data = np.load(fn_bh)
        m = data["m"]
        f = data["f"]
    else:
        sol, fo = toms748(_f, *bracket, full_output=True, disp=True, rtol=N2_RTOL)
        print("Found the right N2!")
        print(fo)

        # re-compute the mass function 
        m, f = find_entire_DM(ϵ, sol, h_desired, dn)
        # log_int = np.log(check_constraint(m, f))
        # print(f"log(integral) = {log_int}")

    return m, f


def find_N2_window_all(ϵ, ax=None):
    h_all = [-20]
    approx_N2 = [[18, 23]]

    for i in range(0, len(h_all)):
        bracket1 = [approx_N2[i][0] - 1, approx_N2[i][0] + 1]
        dn1 = join(FIND_N2_DN, f"h={h_all[i]:.1f}-left")
        m1, f1 = find_N2_window(ϵ, h_all[i], bracket1, dn1)

        bracket2 = [approx_N2[i][1] - 1, approx_N2[i][1] + 1]
        dn2 = join(FIND_N2_DN, f"h={h_all[i]:.1f}-right")
        m2, f2 = find_N2_window(ϵ, h_all[i], bracket2, dn2)

        if ax is not None: 
            ax.plot(m1, f1)
            ax.plot(m2, f2)


def plot_mass_func():
    fig, ax = plt.subplots()

    plot_constraints(ax)
    find_N2_window_all(1e-4, ax=ax)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M/M_\odot$")
    ax.set_ylabel(r"$f_\text{PBH}$")
    ax.set_xlim(1e-18, 1e4)
    ax.set_ylim((1e-4, 3))

    fn = join(FIND_N2_DN, "f_PBH.pdf") 
    plt.savefig(fn, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    # model = USRManuel(1e-4, 3.351912804219496e-05 , 19.961953829161633, 2.2, 'tmp/')
    # ms.save_solve_all(model, toplot=True)
    
    # find_N2_window_all(1e-4)

    # h = -20: [19.959870481923993, 24.74341529302672]
    plot_mass_func()
