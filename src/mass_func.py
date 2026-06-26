import numpy as np 
from os.path import join
import matplotlib.pyplot as plt
import pickle
from pathlib import Path 
from scipy.integrate import simpson
from scipy.optimize import toms748, root_scalar
from scipy.interpolate import CubicSpline
from pathos.multiprocessing import ProcessingPool as Pool

from src.abund.ParaClass import ParaClassMod
from . import ms
from .ms import USRManuel, get_P_max
from .common import add_fn_suffix, get_M_H, get_f_gw
from .abund import USR_full, get_f
from. import get_single
from .constants import NUM_THREADS, POWER_SPEC_FN, BH_F_FN, PEAK_K, MODEL_CACHE_FN
from .gw_signal import plot_Omega, get_SNR, get_Ω_itp, get_Ω_itp_full
from .get_single import get_P_DeltaN, get_Pmax_DeltaN, get_f_tot
from .gw import get_Ω

import scienceplots
plt.style.use("science")

# PBH_CONST_DN = "data/PBHbounds/bounds"
# PBH_CONST_LIST = ["Evaporation.txt", "EROS.txt", "HSC.txt", "K.txt", "M.txt", "CMB.txt", "511keV.txt"]
# PBH_CONST_COORD = [[2e-18, 1e-1], [2e-4, 2.2e-2], [3e-10, 3e-2], [2e-10, 5e-1], [2e-2, 5e-1], [1e2, 1e-3],  [1e-25, 1e-1]]
# PBH_CONST_ROT = [-90, 0, -90, 0, 0, -90, 0]

PBH_CONST_DN = "data/PBHconstraints/"
PBH_CONST_LIST = ["CMB", "EGR", "Voyager", "SUB", "EROS", "acc"]
PBH_CONST_LABELS = ["Evap", None, None, "HSC", "EROS", "Accret"]
PBH_CONST_COORD = [[5e-19, 1e-1], None, None, [2e-10, 5e-2], [2e-5, 2e-1], [1e2, 1e-2]]
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
    cs_Pmax = get_Pmax_DeltaN(ϵ, N2, h_desired)

    # NOTE: get_P_DeltaN scales the powe spectrum __globally__ 
    # this is not very accurate in log scale: the first SR is wrong! 
    # but since we care about the abundance here, it is not important!
    model, _get_P_all = get_single.get_P_DeltaN(ϵ, N2, h_desired, cs_Pmax)

    ######################################################
    # use root finding to find the right ΔN to give f_tot = 1
    ######################################################

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
        with open(fn_model, "wb") as file:
            print("Dumping the model: ", vars(model))
            pickle.dump(model, file)

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

    if not Path(fn_bh).exists():
        sol, fo = toms748(_f, *bracket, full_output=True, disp=True, rtol=N2_RTOL)
        print("Found the ccorrect N2!")
        print(fo)

        # re-compute the mass function 
        _, _ = find_entire_DM(ϵ, sol, h_desired, dn, suffix)
        # log_int = np.log(check_constraint(m, f))
        # print(f"log(integral) = {log_int}")
    else:
        print("Found cache files. SKIPPING...")

    with np.load(fn_bh) as data:
        m = data["m"]
        f = data["f"]

    return m, f


def find_N2_window_all(ϵ, plot=False):
    # h_all = [-20, -10, -5, -1]
    # approx_N2 = [[18, 24], [18, 24], [18, 25], [18, 25]]
    h_all = [-20, -10, -5, -1]
    approx_N2 = [[18, 25], [18, 25], [18, 26], [19, 26]]

    for i in range(0, len(h_all)):
        dn = join(FIND_N2_DN, f"h={h_all[i]:.1f}")
        print(dn)
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
            ax.set_ylim((1e-5, 2))

            fn = join(dn, "f_PBH.pdf") 
            plt.savefig(fn, bbox_inches="tight")
            plt.close(fig)

            plot_Omega(dn)


def plot_SNR_all():
    """
    plot SNR against N2 for the selected h
    """
    fig, ax = plt.subplots()

    h_all = [-20, -10, -5, -1]
    for i in range(0, len(h_all)):
        dn = join(FIND_N2_DN, f"h={h_all[i]:.1f}")

        N2_bracket, itp_f, itp_Ω = get_Ω_itp(dn)

        N2_all = np.linspace(*N2_bracket, 100)
        SNR_all = np.zeros(N2_all.shape)
        for j in range(0, len(N2_all)):
            f = itp_f(N2_all[j])
            Ω = itp_Ω(N2_all[j])
            SNR_all[j] = get_SNR(f, Ω)
            # ax.plot(f, Ω)

        ax.plot(N2_all, SNR_all, label=f"$h={h_all[i]:.0f}$")
    
    ax.set_xlabel("$N_-$")
    ax.set_ylabel("SNR")

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    ax.legend()
    plt.savefig(join(FIND_N2_DN, "SNR_N2.pdf"), bbox_inches="tight")
    plt.close(fig)


###########################################################################
# project LISA sensitivity back to f(m)
###########################################################################

def plot_LISA_f():
    """
    expect that the left and right scenarios have been found already

    basic idea: 
        since the f_tot = 1 has been found, we just need to vary N2 to move it left and right
    """
    # h_all = [-20, -10, -5, -1]
    h_all = [-20, -1]
    M_H_CACHE = "M_H.npz"

    fig, ax = plt.subplots()
    plot_constraints(ax)

    for i in range(0, len(h_all)):
        dn = join(FIND_N2_DN, f"h={h_all[i]:.1f}")
        # suffixes = ["left", "right"] 
        
        fn_M_H = join(dn, M_H_CACHE)
        if Path(fn_M_H).exists():
            with np.load(fn_M_H) as data:
                ftot = data["ftot"]
                M_H = data["M_H"]
        else:
            ftot = np.logspace(-8, 0, num=5)
            M_H= np.array([_get_ftot_small(dn, x) for x in ftot])
            np.savez(fn_M_H, ftot=ftot, M_H=M_H)

        # print(ftot_desired_all, M_H_all)

        # ax.plot(M_H[:, 0], ftot)
        # ax.plot(M_H[:, 1], ftot)
        ax.fill_betweenx(ftot, M_H[:, 0], M_H[:, 1], alpha=0.4, label=rf"$h={h_all[i]:.1f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-19, 1e4)
    ax.set_ylim((1e-5, 2))
    ax.set_xlabel(r"$M/M_\odot$")
    ax.set_ylabel(r"$f_\text{PBH}$")

    ax.legend(frameon=True)
    plt.savefig(join(FIND_N2_DN, "LISA_to_f.pdf"), bbox_inches="tight")
    plt.close(fig)



def _get_ftot_small(dn, ftot_desired):
    """
    get the f_tot(M_H) with SNR == 1

    all the model parameters are taken from model files saved in dn
    """
    suffixes = ["left", "right"]
    
    fs = [[], []]
    Ωs = [[], []]
    N2s = np.zeros(2)
    for i in [0, 1]:
        ###################################################
        # first get ftot == ftot_desired at N2s[0] and N2s[1]
        ###################################################
        fn_model = add_fn_suffix(join(dn, MODEL_CACHE_FN), suffixes[i])
        model_bad = pickle.load(open(fn_model, "rb"))
        
        fn_model = add_fn_suffix(join(dn, POWER_SPEC_FN), suffixes[i])
        with np.load(fn_model) as data:
            k = data["k"]
            P = data["P"]

        _get_ftot = lambda x: np.log10(get_f_tot(model_bad, k, P*x)/ftot_desired)
        sol = root_scalar(_get_ftot, bracket=[1e-2, 1.1], rtol=DELTA_N_RTOL)
        # print(sol)
        if not sol.converged:
            raise RuntimeError("Solver not converged!")

        Ω = get_Ω(k, P*sol.root)
        f = k * get_f_gw(model_bad.V0, model_bad.N2+model_bad.ΔN)

        SNR = get_SNR(f, Ω)
        # print(f"SNR = {SNR}")

        fs[i] = f 
        Ωs[i] = Ω 
        N2s[i] = model_bad.N2
   
    ###################################################
    # get the interpolators
    ###################################################
    _, itp_f, itp_Ω = get_Ω_itp_full(fs[0], Ωs[0], N2s[0], fs[1], Ωs[1], N2s[1])
    brackets = [[10, N2s[0]], [N2s[1], 30]]
    ###################################################
    # compute N2
    ###################################################
    M_Hs = np.zeros(2)
    for i in [0, 1]:
        sol = root_scalar(lambda x: np.log10(get_SNR(itp_f(x), itp_Ω(x))), 
                          bracket=brackets[i], rtol=1e-5, method="brentq")
        if not sol.converged:
            raise RuntimeError("Solver not converged!")

        fn = add_fn_suffix(join(dn, MODEL_CACHE_FN), suffixes[i])
        model_bad = pickle.load(open(fn, "rb"))
        # need to regenerate model object, since V0 is different for different N2
        model = USRManuel(model_bad.ϵ, model_bad.β, sol.root, model_bad.ΔN, 
                          None, fast=True, silent=True)
        M_Hs[i] = get_M_H(model.V0, sol.root)
    
    return M_Hs

if __name__ == "__main__":
    # save_envelope_all()

    # model = USRManuel(1e-4, 5.243775946638871e-06, 20, 1.8 , 'tmp/')
    # model = get_single.fix_h(1e-3, 25, 1.8, -1)
    # ms.save_solve_all(model, toplot=True)
    
    # find_N2_window_all(1e-3, True)

    # plot_SNR_all()

    plot_LISA_f()
