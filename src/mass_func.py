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
from .constants import NUM_THREADS

import scienceplots
plt.style.use("science")

PBH_CONST_DN = "data/PBHbounds/bounds"
PBH_CONST_LIST = ["Evaporation.txt", "EROS.txt", "HSC.txt", "K.txt", "M.txt", "CMB.txt"]
# PBH_CONST_LIST = ["Evaporation.txt", "Microlensing.txt", "CMB.txt"]
PBH_CONST_COORD = [[1e-18, 1e-4], [1e-4, 2e-2], [1e-10, 3e-3], [2e-10, 5e-1], [1e-2, 5e-1], [1e2, 1e-3]]
PBH_CONST_ENVL_FN = "data/PBHbounds_all.pkl"


def comp_envelope_all():
    log_m = np.linspace(-19, 4, 10000)
    log_f = np.ones(log_m.shape) * 10
    for fn in PBH_CONST_LIST:
        mi, fi = np.loadtxt(join(PBH_CONST_DN, fn)).T

        log_f_intp = np.interp(log_m, np.log10(mi), np.log10(fi), left=10, right=10)
        log_f = np.where(log_f_intp < log_f, log_f_intp, log_f)

    return log_m, log_f

def save_envelope_all():
    log_m, log_f = comp_envelope_all()
    with open(PBH_CONST_ENVL_FN, "wb") as f:
        pickle.dump((log_m, log_f), f)

def get_envelope_all():
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
        ax.text(*PBH_CONST_COORD[i], PBH_CONST_LIST[i].replace(".txt", ""), va="center_baseline")

    ax.fill_between(ax.get_xlim(), 1, 1000, color="gray", alpha=0.3, linewidth=0.0)

def find_entire_DM(ϵ, N2, h_desired):
    """
    find the set of parameter to have f_tot = 1
    """
    ΔN_all = np.linspace(3.5, 3.7, NUM_THREADS)
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
    cs = CubicSpline(ΔN_all, Pmax_all)
    # print(cs(ΔN_all[0]))
    
    ######################################################
    # compute the full spectrum using ΔN_all[0]
    ######################################################
    model = get_single.fix_h(ϵ, N2, ΔN_all[0], h_desired)
    k, P = get_single.compute_single(model, GetP=True)
    
    ######################################################
    # use root finding to find the right ΔN to give f_tot = 1
    ######################################################
    def _get_P_all(x):
        """
        need to get the full power specturm from a given ΔN (from interpolation)
        """
        P_max_desired = cs(x)
        P_new = P / np.amax(P) * P_max_desired
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
        print(f"ftot = {ftot}")
        return ftot - 1

    sol, fo = toms748(_f, ΔN_all[0], ΔN_all[-1], rtol=1e-5, full_output=True, disp=True)
    print(fo)

    ######################################################
    # now the "correct" ΔN has been found 
    ######################################################
    k, P = _get_P_all(sol)
    # use the "universal" model object, just like in _f()
    # model = get_single.fix_h(ϵ, N2, sol, h_desired)
    m, f = get_single.get_f(model, k, P)
    return m, f


def test():
    # DeltaN = 2.25
    N2 = 25
    ϵ = 1e-4 
    h = -20
    dir_n = "./find_entire_DM/"
    Path(dir_n).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    plot_constraints(ax)

    m, f = find_entire_DM(ϵ, N2, h)
    # print(m, f)
    ax.plot(m, f)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M/M_\odot$")
    ax.set_ylabel(r"$f_\text{PBH}$")
    ax.set_ylim((1e-4, 3))

    fn = join(dir_n, "f_PBH.pdf") 
    print(fn)
    plt.savefig(fn, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    test()

    # model = USRManuel(1e-4, 1e-5, 23, 2.6, None)
    # print(ms.get_P_max(model))
    
    # h = -20
    # N2 = 20 around the lower end of the astroid window 
    # ΔN ~ 2.25
    # N2 = 25: ΔN ~ 3.3
    # BUG: for N2=25, the power spectrum seems very low (for ΔN>3)
