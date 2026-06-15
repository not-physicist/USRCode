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
from scipy.integrate import solve_ivp

from ..constants import EOM_PLOT_FN
from .EOM import _get_f
from .ModelClass import solve_USR, solve_SR1

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


def compare_EOM(model, sol):
    """
    compare the EOM from model object and from EOM.py
    """
    # compare the background ϕ(t)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    
    N1 = sol[0] - sol[0][0]
    ϕ1 = sol[1]
    π1 = sol[2]

    N2 = model.sol_full[0] - model.sol_full[0][0]
    ϕ2 = model.sol_full[1]
    π2 = model.sol_full[2]

    ax1.plot(N1, ϕ1, c="k")
    ax1.plot(N2, ϕ2, ls="--", label="ModelClass.py")
    ax1.set_xlabel(r"$N$")
    ax1.set_ylabel(r"$\phi$")
    ax1.set_xlim((30, 33))
    ax1.set_ylim((3.3, 3.4))
    ax1.legend()

    itp = lambda x: np.interp(x, N2, ϕ2)
    ax2.plot(N1, np.abs((ϕ1 - itp(N1))/ϕ1), c='k')
    ax2.set_xlabel("$N$")
    ax2.set_ylabel(r"$\Delta_\text{rel} \phi$")
    ax2.set_yscale("log")
    
    # print(model.sol_full[0][0])
    # print("ϕᵢ: ", np.interp(model.N1_exact, N1, ϕ1), np.interp(model.N1_exact, N2, ϕ2))
    # print("ϕₑ: ", np.interp(model.N1_exact + model.ΔN, N1, ϕ1), np.interp(model.N1_exact+model.ΔN, N2, ϕ2))
    # print("πᵢ: ", np.interp(model.N1_exact, N1, π1), np.interp(model.N1_exact, N2, π2))
    itp = lambda x: np.interp(x, ϕ1[::-1], N1[::-1])
    print("ΔN from EOM.py: ", itp(model.ϕₑ) - itp(model.ϕᵢ))

    itp = lambda x: np.interp(x, ϕ1[::-1], π1[::-1])
    print("πᵢ from EOM.py : ", itp(model.ϕᵢ))
    itp2 = lambda x: np.interp(x, ϕ2[::-1], π2[::-1])
    print("πᵢ from ModelClass.py : ", itp2(model.ϕᵢ))
    # BUG: STEP_SIZE_EOM has to be at least 1e-4 to get the (more or less)
    # correct ΔN
    # possibily due to the fact that the solver can step over the start 
    # of the USR phase
  
    fn = join(model.dn, EOM_PLOT_FN)
    plt.savefig(fn)
    plt.close(fig)
   
    ms.plot_z(model, sol)
    ms.plot_z(model, model.sol_full, "z_full.pdf")

    # compare the power spectrum
    # backgrounds = ms.get_ms_quant(model, sol)
    # k, P = ms.solve_power_spec(*backgrounds, model.ϕᵢ, plot_fn=None, num=250, log_k_min=-2, log_k_max=3)
    backgrounds_model = ms.get_ms_quant(model, model.sol_full)
    k_model, P_model = ms.solve_power_spec(*backgrounds_model, model.ϕᵢ, plot_fn=None, num=250, log_k_min=-2, log_k_max=3)
    
    fig, ax = plt.subplots()
    # ax.plot(k, P, c="k")
    ax.plot(k_model, P_model, ls="--")
    # ax.plot(sol[0], backgrounds[3], c="k")
    # ax.plot(model.sol_full[0], backgrounds_model[3], ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    plt.savefig(join(model.dn, "PS_comp.pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_USR():
    """
    """
    model = USRManuel(1e-4, 1e-5, 25, 2.4, "tmp/")

    # sol1 = solve_USR(model.V0, model.ΔN, model.πᵢ)
    sol1 = solve_SR1(model.ϵ, model.η, model.V0, model.ϕ_cmb_i)

    u0 = [model.ϕ_cmb, model.π_cmb, 1.0]
    _V = lambda x: model.get_V(x)
    _dV = lambda x: model.get_dV(x)
    # tspan = [0, model.N1_exact]
    tspan = [0, 50]
    sol2 = solve_ivp(lambda t, y: EOM._get_f(t, y, _V, _dV), tspan, u0, method="DOP853", max_step=1e-4)

    fig, ax = plt.subplots()

    ax.plot(sol1.t, sol1.y[0] + model.ϕᵢ,)
    ax.plot(sol2.t, sol2.y[0], ls="--")
    # ax.plot([sol2.t[0], sol2.t[-1]], [model.ϕᵢ, model.ϕᵢ], c="k")
    # ax.plot([sol2.t[0], sol2.t[-1]], [model.ϕₑ, model.ϕₑ], c="k")

    itp = lambda x: np.interp(x, sol2.y[0][::-1], sol2.t[::-1])
    print("ΔN from EOM.py: ", itp(model.ϕₑ) - itp(model.ϕᵢ))

    print(sol1.y[0][-1]+model.ϕᵢ, sol2.y[0][-1])
    print(sol1.y[1][-1], sol2.y[1][-1])

    fn = join(model.dn, "SR1.pdf")
    plt.savefig(fn)
    plt.close(fig)


if __name__ == "__main__":
    # compare_DeltaN()
    # compare_beta()
    # compare_eps()

    # find_β(1e-3, 22, 2.0, -20, [1e-5, 1e-3])
    # compare_h()

    # ms.plot_z(model, sol)
    # ms.plot_power_evolution(model, sol)

    model = USRManuel(1e-4, 1e-5, 25, 1.5, "tmp/")
    sol = EOM.solve_eom(model, True)
    compare_EOM(model, sol)
    # ms.plot_power_evolution(model, model.sol_full)

    # plot_USR()
