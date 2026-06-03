from pathlib import Path
# from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
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

    model = USRManuel(ϵ, β, N2, DeltaN, dir_n)
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

    if only_P_max:
        return P_max
    else:
        # TODO: cumbersome to compute Z everytime. Maybe put this in f_m.py?
        Z = common.get_Z(model.V0, model.N2)
        para = ParaClassMod(Z, model.dn)

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

'''
def _get_root_DeltaN(N2, ϵ, β, dir_ni, P_desired, bracket):
    """
    find desired ΔN to get P = P_desired
    problem: different h ! 
    not useful
    """
    print("Find root at ", dir_ni)

    def _root_aux_f(x, return_model=False):
        """
        """
        model = USRManuel(ϵ, β, N2, x, dir_ni)
        # to get the maximum of PS, only need to compute PS around the peak (duh!)
        f = compute_single(model, only_P_max=True, force_new=True, N_ms=50, log_k_min=0, log_k_max=2) - P_desired
        print(f"DeltaN = {x} with P - P_desired = {f}, max(P) = {f+P_desired}")
        if return_model:
            return model
        else:
            return f
    
    # Note that the tolerances are for the root position, NOT for the P - P_desired!
    # Note that need to compute PS accurately enough (more data points), so that  
    # it is "smooth" enough for root finding

    sol, fo = toms748(_root_aux_f, bracket[0], bracket[1], 
                      rtol=1e-4, maxiter=20, full_output=True)
    print(fo)
    
    if fo.converged:
        # sometimes the root finding algorithm likes to go "one step further"
        # need to regenerate the data for the root
        # model = _root_aux_f(sol, True)
        model = USRManuel(ϵ, β, N2, sol, dir_ni)
        print(compute_single(model, force_new=True, log_k_min=-2, log_k_max=3, N_ms=200))
    
        return sol 
    else:
        raise Exception("Root finding in _get_root_DeltaN not successful!")
'''

def get_roots_fixed_height(N2, ϵ, h_desired_all, dn, P_desired):
    """
    find the appropriate β and ΔN to get the desired h and P_max
    returns and saves the correspinding model files
    """
    # P_desired = 2e-3
    # interpolator for approximate roots
    itp = CubicSpline([-20, -10, -5, -1], [8.46198627e-05, 5.62065147e-05, 3.78923316e-05, 1.51516104e-05])

    def _get_h_P_max(β, ΔN):
        """
        auxiliary function to get h and P_max from β and ΔN
        """
        model = USRManuel(ϵ, β, N2, ΔN, dn, silent=True)
        P_max = compute_single(model, only_P_max=True, 
                               force_new=True, N_ms=50, log_k_min=0, log_k_max=2, silent=True)
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
                          ΔN1*0.9, ΔN1*1.1, rtol=1e-3, full_output=True)
        if fo.converged:
            print("Found the right ΔN!")
            print(fo)

            model = USRManuel(ϵ, x, N2 , sol, dn, silent=True)

            with open(join(model.dn, MODEL_CACHE_FN), "wb") as f:
                pickle.dump(model, f)

            return model
        else:
            raise Exception("Finding the right ΔN failed!")
    
    # Find the roots (appropriate β and ΔN)
    fn = join(dn, "roots.pkl")
    if Path.exists(fn):
        print("Found cache for roots. Skipping...")
        with open(fn, "rb") as f:
            models_all = pickle.load(f)
    else:
        # models_all = np.zeros((len(h_desired_all), 2))
        models_all = []

        for i in range(0, len(h_desired_all)):
            root_approx = itp(h_desired_all[i])
            bracket = [root_approx*0.9, root_approx*1.1]

            sol, fo = toms748(lambda x: _get_f_only_β(x).h - h_desired_all[i], *bracket, 
                              rtol=1e-2, maxiter=20, full_output=True)

            if fo.converged:
                print("Found the right β!")
                print(fo)
                
                with open(join(dn, MODEL_CACHE_FN), "rb") as f:
                    model = pickle.load(f)
                
                # only to regenerate the model object,
                # if the root finding algo "took a step beyond"
                if model.β != sol:
                    model = _get_f_only_β(sol)

                dn_new = join(dn, f"h={h_desired_all[i]:.1f}/")
                model.change_dn(dn_new)
                models_all.append(model)
            else:
                raise Exception("Finding the right β failed!")
        
        with open(fn, "wb") as f:
            pickle.dump(models_all, f)

    return models_all


def plot_fixed_height():
    """
    get the models with appropriate h and P_desire
    plot the sigma's, power spectra, and the abundances
    """
    P_desired = 2e-3
    N2 = 20
    ϵ = 1e-3
    h_desired_all = [-10]
    # h_desired_all = [-20, -10, -5, -1]
    # h_desired_all = np.linspace(-20, -1, 16)
    dn = "fixed_height/"

    # get the desired β and ΔN
    models_all = get_roots_fixed_height(N2, ϵ, h_desired_all, dn, P_desired)
    # print(roots_all)
    
    # plot power spectra together
    fig, ax = plt.subplots()
    
    res = np.zeros((len(models_all), 8))
    # Now compute Σ and f_tot more carefully (more data points)
    for i in range(0, len(models_all)):
        model = models_all[i]
        # dni = join(dn, f"h={h_desired_all[i]:.1f}/")

        res[i] = compute_single(model, only_P_max=False, N_ms=500, log_k_min=-2, log_k_max=4, force_new=True)
        print("P_max = ", res[i, 0])

        with open(join(model.dn + POWER_SPEC_FN), "rb") as f:
            k, P = pickle.load(f)
            ax.plot(k, P, label=f"$h={model.h:.1f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k/k_i$")
    ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}$")
    ax.legend()
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


'''
def test_h2():
    """
    test power spec and so on with fixed ΔN various h
    with the same epsilonV and epsilonVTilde
    """

    # test power spec and so on with fixed ΔN and ϵTildeV and various h
    DeltaN = 2.0
    h = [-1, -5, -10, -15, -20]
    dir_n = "./test_h2/same_eps_tilde/"
    epsilonVTilde = 0.01
    epsilonV = [_get_epsilonV(x, DeltaN, epsilonVTilde) for x in h]

    dir_ni = [dir_n + f"{x:.2f}/" for x in h]
    for x, y, z in zip(dir_ni, h, epsilonV):
        compute_single(x, DeltaN, y, f=False, epsilonV=z)

    plot_multiple_power(dir_ni, dir_n)


def plot_Gow():
    A = 4.05e-3  # to get f_PBH ~ 1
    Δ = 0.3
    k_star = 1.56e13  # Mpc-1
    k_star_r = 2.74
    K = 1
    C_c = 0.587
    Z = 6.88e16
    para = ParaClass(A, Δ, k_star, k_star_r, K, C_c, Z)


def test_C_l():
    h = -5.0
    NToGo = 10.0
    epsilonV = 1e-7
    etaV = -0.01

    dir_n = "./results_mine3/1.9778/"
    fn_ms = dir_n + "P_k.npz"
    k_star = 1.56e13  # Mpc-1
    k_star_r = 2.74
    K = 1
    C_c = 0.587
    Z = 6.88e16
    para = ParaClassMod(k_star, k_star_r, K, C_c, Z, fn_ms)
    plot_integrand_2D(para, dir_n)
    f_1 = get_f_total(para, lambda x: USR_full.get_J_1(x, etaV, h), get_zero, dir_n)


def test_DeltaN():
    """
    test power spectra with different h and DeltaN
    so that the peak height is the same
    Want to have fixed epsilonVTilde
    """    

    dir_n = "./test_DeltaN/"
    #  h = -np.logspace(0, np.log10(20), 10)
    h = [-1, -5, -10, -15, -20]
    epsilonVTilde = 0.01
    etaV = -0.01
    _approx_h = -np.logspace(0, np.log10(20), 10)
    _approx_roots = [1.5120330876047157, 1.6023424686467176, 1.689378756122145, 1.7684004085675151, 1.8399999999999994, 1.9, 1.9585027091691636, 2.0, 2.0399999999999987, 2.0699999999999994]
    # h is always negativ anyway
    approx_roots = np.interp(np.abs(h), np.abs(_approx_h), _approx_roots)
    brackets = [(x-0.1, x+0.1) for x in approx_roots]
    dir_ni = [dir_n + f"h={x:.2f}/" for x in h]
    P_desired = 2e-3
    
    # npz file to store root finding results
    cache_fn = dir_n + "N_h.npz"
    if not exists(cache_fn):
        # create power spectra in each sub-directory
        results = np.array([_get_root_DeltaN(x, y, P_desired, z) for x, y, z in zip(h, dir_ni, brackets)])
        DeltaN = results[:, 0]
        aux_results = results[:, 1]
        root_results = results[:, 2]
        print(DeltaN, root_results, aux_results)
        np.savez(cache_fn, DeltaN=DeltaN, h=h)
    else:
        data = np.load(cache_fn, allow_pickle=True)
        DeltaN = data["DeltaN"]
        h = data["h"]

    # plot spectra together
    plot_multiple_power(dir_ni, dir_n)

    # plot fs together
    fs_fn = dir_n +"fs.npz"
    Σs_fn = dir_n +"Σs.npz"
    if not exists(fs_fn) or not exists(Σs_fn):
        data = np.array([get_fs(etaV, x, y + "P_k.npz", y, get_Σ=True) for x, y in zip(h, dir_ni)])
        #  print(data)
        #  print(data.shape)
        fs = data[:, 3:]
        Σs = data[:, :3]
        #  print(fs, Σs)
        np.savez(fs_fn, fs=fs)
        np.savez(Σs_fn, Σs=Σs)
    else:
        data = np.load(fs_fn)
        fs = data["fs"]
        data = np.load(Σs_fn)
        Σs = data["Σs"]
        #  print(fs, Σs)

    fig, ax = plt.subplots()
    ax.plot(h, fs[:, 1], label="smooth")
    ax.plot(h, fs[:, 0], label="full")
    ax.plot(h, fs[:, 2], label="perfect USR")
    ax.plot(h, fs[:, 3], label="gaussian", ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$f_{\rm PBH}$")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.3), fancybox=True)
    plt.savefig(dir_n + "f.pdf", bbox_inches="tight")
    plt.close()
    
    fig, ax = plt.subplots()
    ax.plot(h, Σs[:, 0], label=r"$\Sigma_{XX}$")
    ax.plot(h, Σs[:, 1], label=r"$\Sigma_{XY}$")
    ax.plot(h, Σs[:, 2], label=r"$\Sigma_{YY}$")
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$\Sigma_i$")
    plt.legend()
    plt.savefig(dir_n + "Sigmas.pdf", bbox_inches="tight")
    plt.close()

def _get_h(dn):
    return float(dn.replace("h=", ""))


def _get_h_dir(dn):
    # read 
    dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
    hs = [_get_h(x) for x in dirs]
    hs, dirs = zip(*sorted(zip(hs, dirs), reverse=True))
    #  print(dirs, fs)
    return hs, dirs


def plot_OmegaGW(dn):
    """
    plot Omega_GW's on top of each other
    """
    hs, dirs = _get_h_dir(dn)

    fig, ax = plt.subplots()
    cmap = mpl.colormaps['viridis'].reversed()

    for (h, dn_i) in zip(hs, dirs):
        fn = join(dn, dn_i, "P_k.npz")
        data = np.load(fn)
        k = data["k"]
        P = data["P"]
        try:
            Omega = data["Omega"]
        except KeyError:
            Omega = SIGWfast.get_Ω(fn, w=1/3, ProgBar=True)
        #  print(Omega[0:10], len(Omega))
        color = cmap(h/np.amin(hs))
        ax.plot(k, Omega, c=color, label=f"$h={h}$")
    
    ax.set_xlabel("$k/k_s$")
    ax.set_ylabel(r"$\Omega_{GW} h^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim((1e-2, 1e3))
    ax.set_ylim((1e-16, 1e-10))
    ax.legend()

    plt.savefig(dn+"Omega.pdf", bbox_inches="tight")
    plt.close()
'''


if __name__ == "__main__":
    # test_single()
    # _get_root_DeltaN(20, 1e-3, 1e-4, './test_single', 2e-3, [2.3, 2.4])
    # plot_DeltaN_Pmax()
    # find_β(1e-3, 20, 2.0, -10, [1e-5, 1e-3])
    # get_roots_fixed_height(20, 1e-3, [-10], "fixed_height/", 2e-3)
    plot_fixed_height()
