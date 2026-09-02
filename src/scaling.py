"""
try to find the scaling relation between f_{PBH} and Ω_{SIGW}
"""
import numpy as np 
import matplotlib.pyplot as plt
from os.path import join
from scipy.integrate import simpson
from scipy.optimize import toms748, curve_fit
import pickle
from pathlib import Path 

from . import get_single, gw_signal


SCALING_DN = "scaling/"

# probably ok not to change it
ϵ = 1e-4 
N2 = 20  # has peak ~1e-16 M_odot, maybe need refinement
F_PBH_RANGE = [1e-4, 1e4]

DELTA_N_RTOL = 1e-7
NDeltaN = 10

def _get_f_Ω_single(x, model, _get_P, ax1, ax2):
    """
    x = ΔN; output: ftot and Ωtot
    """
    k, P = _get_P(x)
    # ftot = get_single.get_f_tot(model, k, P)
    # Ωtot = gw_signal._get_Ω_tot(k, P, model)

    m, f_PBH = get_single.get_f(model, k, P)
    f, Ω = gw_signal._get_Ω(k, P, model)
    # print(m, f_PBH)
    # print(f, Ω)
    
    if ax1 is not None:
        ax1.plot(m, f_PBH)
    if ax2 is not None:
        ax2.plot(f, Ω)

    return [simpson(f_PBH, np.log(m)), simpson(Ω, np.log(f))]

def _find_DeltaN(x, f_desired, model, _get_P):
    k, P = _get_P(x)
    m, f = get_single.get_f(model, k, P) 
    return np.log(simpson(f, np.log(m))/f_desired)

def compute_f_Ω(h_all):
    ftot_all = np.zeros((len(h_all), NDeltaN))
    Ωtot_all = np.zeros((len(h_all), NDeltaN))
   
    for i in range(0, len(h_all)):
        h = h_all[i] 

        cs = get_single.get_Pmax_DeltaN(ϵ, N2, h)
        model, _get_P = get_single.get_P_DeltaN(ϵ, N2, h, cs)
        suffix = f"_h={h:.1f}"        

        fig2, (ax1, ax2) = plt.subplots(ncols=2)
        
        DeltaN_low, fo = toms748(_find_DeltaN, 1.8, 2.4, rtol=DELTA_N_RTOL, full_output=True, disp=True, args=(F_PBH_RANGE[0], model, _get_P))
        print(fo)

        DeltaN_high, fo = toms748(_find_DeltaN, 1.8, 2.4, rtol=DELTA_N_RTOL, full_output=True, disp=True, args=(F_PBH_RANGE[1], model, _get_P))
        print(fo)

        # try linearly evenly spaced ΔN first 
        ΔN = np.linspace(DeltaN_low, DeltaN_high, NDeltaN) 
        ftot_all[i], Ωtot_all[i] = np.array([_get_f_Ω_single(x, model, _get_P, ax1, ax2) for x in ΔN]).T

        # ax.plot(ftot, Ωtot*1e10, label=f"$h={h:.1f}$")
        # ax.plot(np.log10(ftot), np.log10(Ωtot*1e10), label=f"$h={h:.1f}$")

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"$M/M_\odot$")
        ax1.set_ylabel(r"$f_\text{PBH}$")

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$f/\text{Hz}$")
        ax2.set_ylabel(r"$\Omega_{\text{GW}, 0} h^2$")

        fn = join(SCALING_DN, f"f_Omega{suffix}.pdf")
        fig2.savefig(fn, bbox_inches="tight")
        plt.close(fig2)
    
    return ftot_all, Ωtot_all 

def plot_scaling():
    # h_all = [-20] 
    h_all = [-1, -5, -10, -20]
    
    cache_fn = join(SCALING_DN, "cache.pkl")

    if Path(cache_fn).exists():
        with open(cache_fn, "rb") as f:
            h_all, ftot, Ωtot = pickle.load(f)
    else:
        ftot, Ωtot = compute_f_Ω(h_all)
        with open(cache_fn, "wb") as f:
            pickle.dump((h_all, ftot, Ωtot), f)

    fig, ax = plt.subplots()

    for i in range(0, len(h_all))[::-1]:
        X = np.log10(ftot[i])
        Y = np.log10(Ωtot[i]*1e11)
        ax.plot(X, Y, label=f"$h={h_all[i]:.1f}$")

        # curve fit
        def _f(x, a, b):
            return a*x + b
        
        popt, pcov = curve_fit(_f, X, Y)
        print(popt, pcov)

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_xlabel(r"$f_\text{PBH}$")
    # ax.set_ylabel(r"$\Omega_{\text{GW}, 0} h^2 \cdot 10^{10}$")
    ax.set_xlabel(r"$\log_{10}(f_\text{PBH})$")
    ax.set_ylabel(r"$\log_{10}(\Omega_{\text{GW}, 0} h^2 \cdot 10^{11})$")
    ax.legend(loc="upper left")

    fn = join(SCALING_DN, "scaling.pdf")
    plt.savefig(fn, bbox_inches="tight") 
    plt.close(fig)


######################################################################################
# f_PBH against ΔN for several h's
######################################################################################

def compute_NG(h_all, ΔN):
    ftot_all = np.zeros((len(h_all), NDeltaN))
    Ωtot_all = np.zeros((len(h_all), NDeltaN))
    
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    for i in range(0, len(h_all)):
        h = h_all[i]
        cs = get_single.get_Pmax_DeltaN(ϵ, N2, h)
        model, _get_P = get_single.get_P_DeltaN(ϵ, N2, h, cs)

        ftot_all[i], Ωtot_all[i] = np.array([_get_f_Ω_single(x, model, _get_P, ax1, ax2) for x in ΔN]).T
        print(ftot_all)
        print(Ωtot_all)

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"$M/M_\odot$")
        ax1.set_ylabel(r"$f_\text{PBH}$")

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$f/\text{Hz}$")
        ax2.set_ylabel(r"$\Omega_{\text{GW}, 0} h^2$")

        suffix = f"_h={h:.1f}"        
        fn = join(SCALING_DN, f"f_Omega_NG{suffix}.pdf")
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
   
    return ftot_all, Ωtot_all

def plot_NG():
    # h_all = [-1, -5, -10, -20]
    h_all = [-1, -20]
    # ΔN = np.linspace(2.1542458130071527, 2.247559828766694, NDeltaN)
    ΔN = np.linspace(1.9, 2.0, NDeltaN)

    cache_fn = join(SCALING_DN, "cache_NG.pkl")

    if Path(cache_fn).exists():
        with open(cache_fn, "rb") as f:
            h_all, ftot, Ωtot = pickle.load(f)
    else:
        ftot, Ωtot = compute_NG(h_all, ΔN)
        with open(cache_fn, "wb") as f:
            pickle.dump((h_all, ftot, Ωtot), f)
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for i in range(0, len(h_all))[::-1]:
        ax1.plot(ΔN, np.log10(ftot[i]), label=f"$h={h_all[i]:.1f}$")
        ax2.plot(ΔN, np.log10(Ωtot[i]*1e11), label=f"$h={h_all[i]:.1f}$")

    # ax1.set_yscale("log")
    ax1.set_xlabel(r"$\Delta N$")
    ax1.set_ylabel(r"$\log_{10}(f_\text{PBH})$")

    fn = join(SCALING_DN, "scaling_NG_f.pdf")
    ax1.legend(loc="lower right")
    fig1.savefig(fn, bbox_inches="tight") 
    plt.close(fig1)

    # ax2.set_yscale("log")
    ax2.set_xlabel(r"$\Delta N$")
    ax2.set_ylabel(r"$\log_{10}(\Omega_{\text{GW}, 0}h^2 \cdot 10^{11})$")

    fn = join(SCALING_DN, "scaling_NG_Omega.pdf")
    ax2.legend(loc="lower right")
    fig2.savefig(fn, bbox_inches="tight") 
    plt.close(fig2)



if __name__ == "__main__":
    # model = get_single.fix_h(ϵ, N2, 2.0, -1)
    # m, f = get_single.compute_single(model, GetF=True)
    # print(m)

    # plot_scaling()
    plot_NG()
