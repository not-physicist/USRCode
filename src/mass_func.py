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
from .common import add_fn_suffix
from .abund import USR_full, get_f
from. import get_single
from .constants import NUM_THREADS, POWER_SPEC_FN, BH_F_FN, PEAK_K, MODEL_CACHE_FN
from .gw_signal import plot_Omega

import scienceplots
plt.style.use("science")

# PBH_CONST_DN = "data/PBHbounds/bounds"
# PBH_CONST_LIST = ["Evaporation.txt", "EROS.txt", "HSC.txt", "K.txt", "M.txt", "CMB.txt", "511keV.txt"]
# PBH_CONST_COORD = [[2e-18, 1e-1], [2e-4, 2.2e-2], [3e-10, 3e-2], [2e-10, 5e-1], [2e-2, 5e-1], [1e2, 1e-3],  [1e-25, 1e-1]]
# PBH_CONST_ROT = [-90, 0, -90, 0, 0, -90, 0]

PBH_CONST_DN = "data/PBHconstraints/"
PBH_CONST_LIST = ["CMB", "EGR", "Voyager", "SUB", "EROS", "acc"]
PBH_CONST_LABELS = ["Evap", None, None, "HSC", "EROS", "Accret"]
PBH_CONST_COORD = [[5e-19, 1e-1], None, None, [2e-10, 5e-2], [2e-5, 2e-1], [1e2, 1e-3]]
PBH_CONST_ROT = [-90, None, None, -80, -45, -90]

PBH_CONST_ENVL_FN = "data/PBHbounds_all.pkl"


DELTA_N_RTOL = 1e-7
N2_RTOL = 1e-5

FIND_N2_DN = "find_N2/"

def comp_envelope_all():
    """
    compute the envelope of all (relevant) PBH constraints
    """
    log_m = np.linspace(-22, 5, 10000)
    log_f = np.ones(log_m.shape) * 1e5
    for fn in PBH_CONST_LIST:
        # log_mi, log_fi = np.loadtxt(join(PBH_CONST_DN, fn)).T
        # log_f_intp = np.interp(log_m, np.log10(mi), np.log10(fi), left=10, right=10)
        # log_f = np.where(log_f_intp < log_f, log_f_intp, log_f)

        log_mi, log_fi = np.loadtxt(join(PBH_CONST_DN, fn+".dat")).T
        log_f_intp = np.interp(log_m, log_mi, log_fi, left=5, right=5)
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
    # print(m, log_m_constr, log_f_constr)
    f_max_itp = 10**np.interp(np.log10(m), log_m_constr, log_f_constr)
    # print(f_max_itp)

    integral = simpson(f/f_max_itp/m, m)
    return integral

def plot_constraints(ax):
    """
    plot the __individual__ constraint lines on the ax
    """
    for i in range(0, len(PBH_CONST_LIST)):
        fn = join(PBH_CONST_DN, PBH_CONST_LIST[i] + ".dat")
        mi, fi = np.loadtxt(fn).T
        ax.plot(10**mi, 10**fi, c="k")
        if PBH_CONST_COORD[i] is not None:
            ax.text(*PBH_CONST_COORD[i], PBH_CONST_LABELS[i], va="center_baseline", rotation=PBH_CONST_ROT[i])

    ax.fill_between([1e-20, 1e5], 1, 1000, color="gray", alpha=0.3, linewidth=0.0)
    
    # to test of get_envelope_all() is working
    # log_m, log_f = get_envelope_all()
    # ax.plot(10**log_m, 10**log_f, color="red")


def find_entire_DM(ϵ, N2, h_desired, dn=None, suffix=None):
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
    if dn is not None and suffix is not None:
        # if need β and other stuff, run get_single.fix_h()
        # paras = [ϵ, N2, sol]
        
        fn_ps = add_fn_suffix(join(dn, POWER_SPEC_FN), suffix)
        np.savez(fn_ps, k=k, P=P)

        fn_bh = add_fn_suffix(join(dn, BH_F_FN), suffix)
        np.savez(fn_bh , m=m, f=f)

        fn_model = add_fn_suffix(join(dn, MODEL_CACHE_FN), suffix)
        with open(fn_model, "wb") as f:
            pickle.dump(model, f)

    return m, f


def find_N2_window(ϵ, h_desired, bracket, dn, suffix):
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
    
    # fn_ps = add_fn_suffix(join(dn, POWER_SPEC_FN), suffix)
    fn_bh = add_fn_suffix(join(dn, BH_F_FN), suffix)

    if Path(fn_bh).exists():
        print("Found cache files. SKIPPING...")
        with np.load(fn_bh) as data:
            m = data["m"]
            f = data["f"]
    else:
        sol, fo = toms748(_f, *bracket, full_output=True, disp=True, rtol=N2_RTOL)
        print("Found the right N2!")
        print(fo)

        # re-compute the mass function 
        m, f = find_entire_DM(ϵ, sol, h_desired, dn, suffix)
        # log_int = np.log(check_constraint(m, f))
        # print(f"log(integral) = {log_int}")

    return m, f


def find_N2_window_all(ϵ, plot=False):
    h_all = [-20, -10, -1]
    approx_N2 = [[18, 24], [18, 24], [18, 24]]

    for i in range(0, len(h_all)):
        dn = join(FIND_N2_DN, f"h={h_all[i]:.1f}")
        Path(dn).mkdir(parents=True, exist_ok=True)

        # left == lightest
        bracket1 = [approx_N2[i][0] - 1, approx_N2[i][0] + 1]
        # dn1 = join(FIND_N2_DN, f"h={h_all[i]:.1f}-left")
        m1, f1 = find_N2_window(ϵ, h_all[i], bracket1, dn, "left")
        
        # right = heaviest
        bracket2 = [approx_N2[i][1] - 1, approx_N2[i][1] + 1]
        # dn2 = join(FIND_N2_DN, f"h={h_all[i]:.1f}-right")
        m2, f2 = find_N2_window(ϵ, h_all[i], bracket2, dn, "right")

        if plot: 
            fig, ax = plt.subplots()

            plot_constraints(ax)
            ax.plot(m1, f1)
            ax.plot(m2, f2)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$M/M_\odot$")
            ax.set_ylabel(r"$f_\text{PBH}$")
            ax.set_xlim(1e-19, 1e4)
            ax.set_ylim((1e-4, 2))

            fn = join(dn, "f_PBH.pdf") 
            plt.savefig(fn, bbox_inches="tight")
            plt.close(fig)

            # plot_Omega(dn)


if __name__ == "__main__":
    # save_envelope_all()

    # model = USRManuel(1e-4, 3.351912804219496e-05 , 19.961953829161633, 2.2, 'tmp/')
    # ms.save_solve_all(model, toplot=True)
    
    find_N2_window_all(1e-4, True)

    # h = -20: [18.114994614678395, 24.79124705334623]
    # plot_mass_func()
