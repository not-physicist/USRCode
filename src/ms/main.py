from . import EOM 
from .ModelClass import USRManuel 
from . import ms
from ..common import convert_sci_not

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from os.path import join
from scipy.optimize import toms748

COMPARE_DN = "compare_ps/"

def _get_P_all(fn, models, num, log_k_min, log_k_max):
    """
    use the models to compute the power spectrum 
    fn: filename for the cache file
    """
    if Path(fn).exists():
        print("Cache found.")
        with open(fn, "rb") as f:
            k_all, P_all = pickle.load(f)
    else:
        k_all = np.zeros((len(models), num))
        P_all = np.zeros((len(models), num))

        for i in range(0, len(models)):
            sol = EOM.solve_eom(models[i], False)

            backgrounds = ms.get_ms_quant(models[i], sol)
            k_all[i], P_all[i] = ms.solve_power_spec(*backgrounds, models[i].ϕᵢ, plot_fn=None, num=num, log_k_min=log_k_min, log_k_max=log_k_max)
        
        with open(fn, "wb") as f:
            pickle.dump((k_all, P_all), f)

    return k_all, P_all


def compare_DeltaN():
    N2 = 20
    ϵ = 1e-3 
    β = 1e-4

    num = 200
    log_k_min = -2
    log_k_max = 3
    fn = "compare_DeltaN"
    cache_fn = join(COMPARE_DN, fn+".pkl")
    fig_fn = join(COMPARE_DN, fn+".pdf")

    DeltaN_all = np.linspace(2.2, 1.8, 3)
    # technically, can make the following loop run in parallel
    models = [USRManuel(ϵ, β, N2, x, COMPARE_DN) for x in DeltaN_all]

    k_all, P_all = _get_P_all(cache_fn, models, num, log_k_min, log_k_max)

    fig, ax = plt.subplots()
    ax.set_title(rf"$N_- = {N2}, \tilde{{\epsilon}}_V = {convert_sci_not(ϵ)}, \beta = {convert_sci_not(β)}$")

    for i in range(0, len(models)):
        label = rf"$\Delta N={DeltaN_all[i]:.2f}$"
        ax.plot(k_all[i], P_all[i], label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k/k_i$")
    ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}$")
    ax.set_ylim((1e-12, 2e-2))
    ax.legend(loc="lower right")
    
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig)

def compare_beta():
    ϵ = 1e-3
    N2 = 22 
    ΔN= 2.0
    num = 400
    β_all = [1e-3, 1e-4, 1e-5]

    fn = "compare_beta"
    cache_fn = join(COMPARE_DN, fn+".pkl")
    fig_fn = join(COMPARE_DN, fn+".pdf")

    models = [USRManuel(ϵ, x, N2, ΔN, COMPARE_DN) for x in β_all]

    k_all, P_all = _get_P_all(cache_fn, models, num, -2, 3)

    fig, ax = plt.subplots()
    ax.set_title(rf"$\tilde{{\epsilon}}_V = {convert_sci_not(ϵ)}, N_- = {N2}, \Delta N = {ΔN}$")

    for i in range(0, len(β_all)):
        label = rf"$\beta={convert_sci_not(β_all[i])}$"
        ax.plot(k_all[i], P_all[i], label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k/k_i$")
    ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}$")
    ax.set_ylim((1e-12, 3e-2))
    ax.legend(loc="upper left")
    
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig)

def compare_eps():
    β = 1e-4
    N2 = 22 
    ΔN = 2.0
    ϵ_all = [1e-3, 3e-4, 1e-4]

    fn = "compare_eps"
    cache_fn = join(COMPARE_DN, fn+".pkl")
    fig_fn = join(COMPARE_DN, fn+".pdf")

    num = 200
    log_k_min = -2 
    log_k_max = 3
    models = [USRManuel(x, β, N2, ΔN, COMPARE_DN) for x in ϵ_all]
    
    k_all, P_all = _get_P_all(cache_fn, models, num, log_k_min, log_k_max)

    fig, ax = plt.subplots()
    ax.set_title(rf"$\beta = {convert_sci_not(β)}, N_- = {N2}, \Delta N = {ΔN}$")

    for i in range(0, len(ϵ_all)):
        label = rf"$\tilde{{\epsilon}}_V={convert_sci_not(ϵ_all[i])}$"
        ax.plot(k_all[i], P_all[i], label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k/k_i$")
    ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}$")
    ax.set_ylim((1e-12, 1e-2))
    ax.legend(loc="upper left")
    
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig)


def find_β(ϵ, N2, ΔN, h_desired, bracket):
    """
    find β to get h = h_desired
    """
    def _get_f(x):
        model = USRManuel(ϵ, x, N2, ΔN, None, False)
        return model.h - h_desired
    
    sol, fo = toms748(_get_f, *bracket, rtol=1e-5, full_output=True)
    if fo.converged:
        print(sol, fo)
        return USRManuel(ϵ, sol, N2, ΔN, None, True)
    else:
        raise Exception("Finding β not successful!")


def compare_h():
    N2 = 20
    ΔN = 2.0
    # ϵ_all = [1e-3, 2e-4, 1e-4]
    ϵ = 1e-3
    β_bracket = [1e-5, 1e-3]
    # β_brackets = [[1e-4, 1.2e-4], [4.5e-5, 4.7e-5], [3e-5, 3.3e-4]]
    # h = -10
    h_all = [-5, -10, -30]

    fn = "compare_h"
    root_fn = join(COMPARE_DN, fn+"_root.pkl")
    cache_fn = join(COMPARE_DN, fn+".pkl")
    fig_fn = join(COMPARE_DN, fn+".pdf")

    num = 200
    log_k_min = -2 
    log_k_max = 3
    
    if Path.exists(root_fn):
        with open(root_fn, "rb") as f:
            models = pickle.load(f)
    else:
        models = [find_β(ϵ, N2, ΔN, h_all[i], β_bracket) for i in [0, 1, 2]]
        with open(root_fn, "wb") as f:
            pickle.dump(models, f)
    
    k_all, P_all = _get_P_all(cache_fn, models, num, log_k_min, log_k_max)

    fig, ax = plt.subplots()
    ax.set_title(rf"$N_- = {N2}, \Delta N = {ΔN}, \tilde{{\epsilon}}_V = {ϵ}$")

    for i in range(0, len(h_all)):
        label = rf"$h={h_all[i]:.1f}$"
        ax.plot(k_all[i], P_all[i], label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k/k_i$")
    ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}$")
    ax.set_ylim((1e-12, 1e-2))
    ax.legend(loc="lower right")

    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    compare_DeltaN()
    # compare_beta()
    # compare_eps()

    # find_β(1e-3, 22, 2.0, -20, [1e-5, 1e-3])
    # compare_h()

    # model = USRManuel(0.001, 1e-3, 22, 2.0, "new_pot/")
    # sol = EOM.solve_eom(model, True)

    # ms.plot_z(model, sol)
    # ms.plot_power_evolution(model, sol)
