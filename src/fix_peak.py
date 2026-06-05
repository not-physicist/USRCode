from pathlib import Path
# from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
# import matplotlib as mpl
# from os.path import exists
#  from scipy.special import erf
# from scipy.special import erfc
# from scipy.optimize import root_scalar
# from scipy.optimize import brentq
# from scipy import optimize
from scipy.optimize import toms748
# from os import listdir 
from os.path import isdir, isfile, join
import pickle
from scipy.interpolate import CubicSpline

import scienceplots
plt.style.use("science")

from . import common 
from .common import get_zero
from .common import get_one

from .import ms
from .ms import USRManuel
from .ms import EOM
# from .ms.main import find_β

from .abund import ParaClassMod
# from .abund import ParaClass
from .abund import get_f_total
# from .abund import plot_integrand_2D

from .abund import USR_full
from .abund import USR_only
from .abund import USR_smooth
# from .abund import gaussian

# from .gw import get_Ω

from .constants import POWER_SPEC_FN, POWER_SPEC_PLOT_FN, MODEL_CACHE_FN
from .common import timer 

# relative tolerance for root finding
DELTAN_RTOL = 1e-5 
BETA_RTOL = 1e-4

# @common.timer
def test_single():
    """
    test only one set of parameters
    """
    DeltaN = 2.40
    N2 = 20
    ϵ = 1e-4 
    β = 2e-5
    dir_n = "./test_single/"
    Path(dir_n).mkdir(parents=True, exist_ok=True)

    # model = USRManuel(ϵ, β, N2, DeltaN, dir_n)
    # ms.save_solve_all(model, True, N_ms=200, log_k_min=-2, log_k_max=3, force_new=True)
    
    # PBH formation
    # Z = common.get_Z(model.V0, model.N2)
    # print(f"Z = {Z:.2e}")
    # para = ParaClassMod(Z, model.dn)
    # para.plot_Σ_int()
    # f_tot = get_f_total(para, lambda x: USR_full.get_J_1(x, model.η2, model.h), get_zero, dir_n)
    # print(f"f_tot = {f_tot:.2e}")

    # SIGW
    # Ω = get_Ω(dir_n)
    # print(Ω)

# @timer
def compute_single(model, only_P_max=False, N_ms=200, force_new=False, log_k_min=-3, log_k_max=4, silent=False):
    """
    Compute the background and ms solution first;
    Then calculate the abundance depending on the statistics
    """

    # model = USRManuel(ϵ, β, N2, DeltaN, dir_n)
    ms.save_solve_all(model, False, N_ms=N_ms, force_new=force_new, log_k_min=log_k_min, log_k_max=log_k_max, silent=silent)
    fn = join(model.dn, POWER_SPEC_FN)
    P_max = np.amax(pickle.load(open(fn, "rb"))[1])
    # print("P_max = ", P_max)

    if only_P_max:
        return P_max
    else:
        Z = common.get_Z(model.V0, model.N2)
        para = ParaClassMod(Z, model.dn)
        para.plot_Σ_int(model.dn)

        res = get_fs(model, para, True)
        return [P_max, *res]


def get_fs(model, para, get_Σ=False):
    """
    Calculate the abundances using four different statistics
    once the power spectrum is saved under fn_ms
    """
    f_1 = get_f_total(para, lambda x: USR_full.get_J_1(x, model.η2, model.h), get_zero)
    f_2 = get_f_total(para, lambda x: USR_smooth.get_J_1(x, model.η2), get_zero)
    f_3 = get_f_total(para, lambda x: USR_only.get_J_1(x), get_zero)
    f_4 = get_f_total(para, get_one, get_zero)

    print("Abundances are: ", f_1, f_2, f_3, f_4)

    if get_Σ:
        return para.XX, para.XY, para.YY, f_1, f_2, f_3, f_4
    else:
        return f_1, f_2, f_3, f_4


def get_model_fixed_height(N2, ϵ, h_desired, P_desired, dn):
    """
    find the appropriate β and ΔN to get the desired h and P_max
    returns and saves the correspinding model files
    """
    # P_desired = 2e-3
    # interpolator for approximate roots
    # itp = CubicSpline([-20, -10, -5, -1], [8.46198627e-05, 5.62065147e-05, 3.78923316e-05, 1.51516104e-05])
    itp = CubicSpline([-20, -10, -5, -1], [9.131304878971815e-05, 5.799549787583399e-05, 3.905100757912693e-05, 1.5805344608102757e-05])

    def _get_h_P_max(β, ΔN):
        """
        auxiliary function to get h and P_max from β and ΔN
        """
        model = USRManuel(ϵ, β, N2, ΔN, dn, silent=True)
        P_max = compute_single(model, only_P_max=True, 
                               force_new=True, N_ms=50, log_k_min=0, log_k_max=1, silent=True)
        print(f"h ={model.h}, max(P)={P_max}")
        # return [model.h - h_desired, P_max - P_desired]
        return [model.h, P_max]

    # want to use the fact that h \propto exp(3*ΔN), P_max \propto exp(6*ΔN)
    # initial guess of ΔN; used **always** at first !
    ΔN0 = 2.0  
    def _get_f_only_β(x):
        """
        auxiliary function
        find the appropriate ΔN to get the right P_max
        """
        # first use the _default_ value of ΔN
        res = _get_h_P_max(x, ΔN0)
        # print(res)
        
        # use the scaling to get the approximate position of ΔN
        ΔN1 = ΔN0 + np.log(P_desired/res[1])/3
        # res2 = _get_h_P_max(x, ΔN1)
        # print(res2)

        # scaling not good enough to find the right ΔN 
        # (but good enough as estimate for it)
        # use root finding again
        sol, fo = toms748(lambda y: _get_h_P_max(x, y)[1] - P_desired, 
                          ΔN1*0.9, ΔN1*1.1, rtol=DELTAN_RTOL, full_output=True, disp=True)

        print("Found the right ΔN!")
        print(fo)

        model = USRManuel(ϵ, x, N2 , sol, dn, silent=True)

        return model
    
    model_fn = join(dn, MODEL_CACHE_FN) 
    if Path(model_fn).exists():
        print(f"Model cache {model_fn} exists. Skipping...")
        model = pickle.load(open(join(dn, MODEL_CACHE_FN), "rb"))
    else:
        print(f"Model cahce {model_fn} not found. Finding the root...")
        root_approx = itp(h_desired)
        bracket = [root_approx*0.9, root_approx*1.1]

        sol, fo = toms748(lambda x: _get_f_only_β(x).h - h_desired, *bracket, 
                          rtol=BETA_RTOL, maxiter=20, full_output=True, disp=True)

        print("Found the right β!")
        print(fo)
        
        print(f"Recomputing ΔN...")
        model = _get_f_only_β(sol)

        with open(model_fn, "wb") as f:
            pickle.dump(model, f)

    return model


def plot_fixed_height():
    """
    get the models with appropriate h and P_desire
    plot the sigma's, power spectra, and the abundances
    """
    P_desired = 2e-3
    N2 = 20
    ϵ = 1e-3
    # h_desired_all = [-10]
    # h_desired_all = [-20, -10, -5, -1]
    # h_desired_all = np.linspace(-20, -1, 16)
    h_desired_all = np.arange(-20, -1, 2)
    dn = "fixed_height/"
    cmap = colormaps["viridis_r"]
    N_ms = 500

    # print(roots_all)
    
    # plot power spectra together
    fig, ax = plt.subplots()
    
    # Now compute Σ and f_tot more carefully (more data points)
    res = np.zeros((len(h_desired_all), 8))
    for i in range(0, len(h_desired_all)):
        print("="*100)
        dn_new = join(dn, f"h={h_desired_all[i]:.1f}/")
        model = get_model_fixed_height(N2, ϵ, h_desired_all[i], P_desired, dn_new)

        with open(join(model.dn + POWER_SPEC_FN), "rb") as f:
            k, P = pickle.load(f)

        if len(k) == N_ms:
            print("The complete power spectrum has been computed. Skipping...")
            res[i] = compute_single(model, only_P_max=False, N_ms=N_ms, log_k_min=-2, log_k_max=4, force_new=False)
        else:
            res[i] = compute_single(model, only_P_max=False, N_ms=N_ms, log_k_min=-2, log_k_max=4, force_new=True)

        print("P_max = ", res[i, 0])
        
        c = cmap(i/(len(h_desired_all)-1))
        if i in range(0, len(h_desired_all))[::2]:
        # skipping some h, so that the PS doesn't look so dense
            with open(join(model.dn + POWER_SPEC_FN), "rb") as f:
                k, P = pickle.load(f)
                ax.plot(k, P, c=c)
    
    map = cm.ScalarMappable(norm=plt.Normalize(vmin=np.amin(h_desired_all), vmax=np.amax(h_desired_all)), cmap=cmap)
    fig.colorbar(map, ax=ax, label=r"$h$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k/k_i$")
    ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}$")
    plt.savefig(join(dn, POWER_SPEC_PLOT_FN), bbox_inches="tight")
    plt.close(fig)

    # plot the Σ's and f_tot's
    fig, ax = plt.subplots()
    ax.plot(h_desired_all, res[:, 1], label=r"$\Sigma_{XX}$")
    ax.plot(h_desired_all, res[:, 2], label=r"$\Sigma_{XY}$")
    ax.plot(h_desired_all, res[:, 3], label=r"$\Sigma_{YY}$")
    ax.set_xlabel("$h$")
    ax.legend()
    plt.savefig(join(dn, "Sigmas.pdf"), bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots()
    ax.plot(h_desired_all, res[:, 4], label="full")
    ax.plot(h_desired_all, res[:, 5], label="smooth")
    ax.plot(h_desired_all, res[:, 6], label="perfect USR")
    ax.plot(h_desired_all, res[:, 7], label="Gaussian")
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$f_\text{tot}$")
    ax.set_yscale("log")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.3), fancybox=True)
    plt.savefig(join(dn, "f_tots.pdf"), bbox_inches="tight")
    plt.close(fig)


def check_Sigma_int():
    """
    it seems that with fixed height, h doesn't affect Σ's at all
    checking this here
    """
    dn = "fixed_height/Sigma_check"
    model = pickle.load(open(join(dn, MODEL_CACHE_FN), "rb"))
    model.change_dn(dn)
    _ = compute_single(model, N_ms=100, force_new=True, log_k_min=-2, log_k_max=1, silent=False, only_P_max=False)

    _ = compute_single(model, N_ms=100, force_new=True, log_k_min=-2, log_k_max=3, silent=False, only_P_max=False)


if __name__ == "__main__":
    # test_single()
    # _get_root_DeltaN(20, 1e-3, 1e-4, './test_single', 2e-3, [2.3, 2.4])
    # plot_DeltaN_Pmax()
    # find_β(1e-3, 20, 2.0, -10, [1e-5, 1e-3])
    # get_roots_fixed_height(20, 1e-3, [-10], "fixed_height/", 2e-3)
    # check_Sigma_int()
    get_model_fixed_height(20, 1e-3, -16, 2e-3, "fixed_height/h=-16.0/")
    # plot_fixed_height()

