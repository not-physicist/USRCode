import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pickle
from scipy.integrate import simpson
from pathlib import Path

# from scipy.interpolate import CubicSpline

from .constants import GW_PLOT_FN, MODEL_CACHE_FN, POWER_SPEC_FN, GW_SPEC_FN
from . import gw 
from .common import get_f_gw, add_fn_suffix, CubicSplineNoExtra


GW_SENS_DN = "data/PLIS/"
GW_SENS_FN = ["plis_BBO.dat", "plis_LISA.dat", "plis_ET.dat"]
GW_EXP_ANNO_COORD = [(1e0, 5e-17), (2.2e-5, 1e-10), (2e0, 1e-9)]
GW_EXP_ANNO_ROT = [55, -52, -90]


def _draw_gw_bounds(ax):
    """
    draw a GW bound curve defined in bound_fn
    """
    # for (i, j) in zip(GW_SENS_FN, GW_EXP_ANNO_COORD):
    for i in range(0, len(GW_SENS_FN)):
        fn = join(GW_SENS_DN, GW_SENS_FN[i])
        sens = np.genfromtxt(fn, comments="#").T
        f = 10**sens[0]
        omegah2 = 10**sens[1]
        ax.plot(f, omegah2, color="k")

        anno = GW_SENS_FN[i].replace("plis_", "").replace(".dat", "")
        ax.text(*GW_EXP_ANNO_COORD[i], anno, rotation=GW_EXP_ANNO_ROT[i])

    # for BBN bound
    h = 0.67
    ax.fill_between([1e-6, 1e6], 2e-6*h**2, 1, color="grey", alpha=0.5, linewidth=0.0)
    ax.annotate("BBN", [1e-3, 2e-6])


def plot_Omega(dn):
    fig, ax = plt.subplots()

    _draw_gw_bounds(ax)

    f_Omega_left = get_Ω(dn, "left")
    ax.plot(*f_Omega_left)
    # get_SNR(dn, "left")

    f_Omega_right = get_Ω(dn, "right")
    ax.plot(*f_Omega_right)
    # get_SNR(dn, "right")
    
    ax.set_xlim((1e-5, 1e1))
    ax.set_ylim((1e-18, 2e-5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f/\text{Hz}$")
    ax.set_ylabel(r"$\Omega_{\text{GW}, 0}h^2$")
    
    fn_Omega = join(dn, GW_PLOT_FN)
    plt.savefig(fn_Omega, bbox_inches="tight")
    plt.close(fig)


def _get_Ω(k, P, model):
    Ω = gw.get_Ω(k, P)
    f_i = get_f_gw(model.V0, model.N2+model.ΔN)
    f = f_i * k
    return f, Ω

def _get_Ω_tot(k, P, model):
    f, Ω = _get_Ω(k, P, model)
    return simpson(Ω, np.log(f))

def get_Ω(dn, suffix=""):
    """
    to compute and save the GW signals Ωh^2
    if cache file exists, then read out
    """
    fn_GW = add_fn_suffix(join(dn, GW_SPEC_FN), suffix) 
    if Path(fn_GW).exists():
        print("Found cache files. SKIPPING...")
        with np.load(fn_GW) as data:
            f = data["f"]
            Ω = data["Omega"]

        return f, Ω

    fn_model = add_fn_suffix(join(dn, MODEL_CACHE_FN), suffix)
    fn_ps = add_fn_suffix(join(dn, POWER_SPEC_FN), suffix)

    with open(fn_model, "rb") as f:
        model = pickle.load(f)

    with np.load(fn_ps) as data:
        k = data["k"]
        P = data["P"]

    f, Ω = _get_Ω(k, P, model)

    np.savez(fn_GW, f=f, Omega=Ω)
    return f, Ω

def _comp_SNR(f, Omega, log_f_det, log_Omega_det):
    """
    compute the SNR
    """
    n = 1 # LISA: auto-correlation
    year = 3.15e7  # s
    
    # print(f, Omega)
    # print(10**log_f_det, 10**log_Omega_det)
    cs = CubicSplineNoExtra(np.log10(f), np.log10(Omega), -np.inf)
    SNR2 = n * year * simpson(( 10**(cs(log_f_det)) / 10**(log_Omega_det) )**2, 
                              10**log_f_det)
    return SNR2
    # BUG: this reults seems wrong!


def save_strain_data(exp="LISA"):
    """
    read Ωh^2(f) from file and save to npz for faster read-out time
    """
    fn = f"data/GWstrain/strain_{exp}.dat"
    data = np.loadtxt(fn, comments="#").T
    log_f = data[0] 
    log_Omega = data[1]
    np.savez(f"data/{exp}_strain.npz", log_f=log_f, log_Omega=log_Omega)


def get_SNR(f, Ω, exp="LISA"):
    """
    read the Omega h^2 from file and calculate SNR of LISA
    """
    # f, Ω = get_Ω(dn, suffix)
    
    # reading the npz file is ~ 5ms, very fast!
    with np.load(f"data/{exp}_strain.npz") as data:
        log_f = data["log_f"]
        log_Omega = data["log_Omega"]

    SNR2 = _comp_SNR(f, Ω, log_f, log_Omega)
    # print(f"SNR^2 = {SNR2}")
    
    return np.sqrt(SNR2)


def get_Ω_itp_full(f1, Ω1, N2_1, f2, Ω2, N2_2):
    # f1 and f2 should only differ by a (universal) constant across all entries
    # TODO: add a single check?

    # frequency is exponential in N_-, take logarithms
    Δlog_f = np.log(f1/f2)[0]
    itp_f = lambda x: np.exp(-(x-N2_1) / (N2_2-N2_1) * Δlog_f) * f1

    # first order approximation: Ω1 = Ω2 (not true!)
    # TODO: do a linear interpolation
    itp_Ω = lambda x: Ω1

    return [N2_1, N2_2], itp_f, itp_Ω

def get_Ω_itp(dn):
    """
    get interpolated GW signals between left and right
    """
    f1, Ω1 = get_Ω(dn, "left")
    N2_1 = pickle.load(open(add_fn_suffix(join(dn, MODEL_CACHE_FN), "left"), "rb")).N2

    f2, Ω2 = get_Ω(dn, "right")
    N2_2 = pickle.load(open(add_fn_suffix(join(dn, MODEL_CACHE_FN), "right"), "rb")).N2
    
    return get_Ω_itp_full(f1, Ω1, N2_1, f2, Ω2, N2_2)


if __name__ == "__main__":
    save_strain_data("BBO")
