import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import CubicSpline 
import tqdm
from os.path import join

# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
# with the std multiprocessing: cannot pickle the functions

import scienceplots
plt.style.use("science")

from ..constants import NUM_THREADS, STEP_SIZE_MS, SOLVER_MS

########################################################################
# solve MS equation

def get_ms_quant(model, sol):
    """
    calculate the quantities for the background for solving the ms equation
    """
    # print(sol)
    N, ϕ, π = sol
    # print(N)
    
    # Check dimension of three arrays 
    # print(N.shape, ϕ.shape, π.shape)
    # Check duplicate in the sol
    _, c = np.unique(N, return_counts=True)
    if (c[c>1]).shape[0] > 0:
        print("WARNING: duplcate element in EOM solution!")

    a = np.exp(N - N[0])
    H = np.sqrt(model.get_V(ϕ) / (3 - π**2/2))
    dϕdt = π * H
    t = cumulative_trapezoid(1/H, N - N[0])
    # τ = cumulative_trapezoid(1/(a*H), N)
    z = a * dϕdt / H
    dzdt = np.diff(z[:-1]) / np.diff(t)
    dzdt_z = dzdt / z[:-2]
    # return N, ϕ, dϕdN, a, H, dϕdt, t, z, dzdt_z
    return t, a, H, z, dzdt_z, ϕ

def plot_z(model, sol, fn="z.pdf"):
    N, ϕ, π = sol
    t, a, H, z, dzdt_z, ϕ = get_ms_quant(model, sol)

    fig, ax = plt.subplots()
    dz = np.diff(z[:-1]) / np.diff(t) * a[:-2]
    ddz = np.diff(dz) / np.diff(t[:-1]) * a[:-3]
    ddz_z = ddz / z[:-3]

    # ax.plot(N, np.abs(z))
    ax.plot(N[:-3], np.abs(ddz_z)/(a*H)[:-3]**2, c="k")

    ax.set_yscale("log")
    
    fn_full = join(model.dn, fn)
    plt.savefig(fn_full, bbox_inches="tight")
    plt.close(fig)

def find_time(t, k_aH, X):
    """
    find the time where k/aH = X
    so that deep in horizon to initialize the fluctuations
    """
    # TODO: add error handling?

    # k_aH = k / (a * H)
    t_i = np.interp(X, (k_aH[:-1])[::-1], t[::-1])
    return t_i

def init_fluc(k, t, a, H, z, dzdt_z):
    """
    initialize the fluctuations using the BD vacuum
    """
    tᵢ = find_time(t, k/(a*H), 100) 
    zᵢ = np.interp(tᵢ, t, z[:-1])
    dzdt_zᵢ = np.interp(tᵢ, t[:-1], dzdt_z)
    gᵢ = 1 / (np.sqrt(2*k) * np.abs(zᵢ)) + 0.0j 
    dgᵢ = -dzdt_zᵢ * gᵢ + 0.0j 
    return tᵢ, [gᵢ, dgᵢ]

def get_diff_eq_ms(t, y, k, _dzdt_z, _a, _H):
    """
    defines the (transformed) MS equation
    """
    g, dg = y
    
    dzdt_z = _dzdt_z(t)
    a = _a(t)
    H = _H(t)

    dydt = [dg, -(H + 2*dzdt_z - 2.0j * k/a)*dg + 2.0j * k/a * dzdt_z * g]
    return dydt

def solve_ms_single(k, t, a, H, z, dzdt_z, aH_USR_i, evo=False):
    """
    solve MS equation with a single k
    """
    
    tᵢ, inits = init_fluc(k, t, a, H, z, dzdt_z)

    if k / aH_USR_i < np.exp(2.5):
        # if horizon-exit takes place before the end of USR (2.5 is the typical/maximal duration)
        # take the termination time to be ~5 e-fold after USR
        t_f = find_time(t, aH_USR_i/(a*H), 0.01)
    else:
        t_f = find_time(t, k/(a*H), 0.01)
    # print(k/aH_USR_i, t_f)
    tspan = [tᵢ, t_f]

    # tspan = [tᵢ, t[-1]]

    # print(a.shape, H.shape, t.shape, dzdt_z.shape)
    _a = lambda x: np.interp(x, t, a[:-1])
    _H = lambda x: np.interp(x, t, H[:-1]) 
    _dzdt_z = lambda x: np.interp(x, t[:-1], dzdt_z)
    
    # using CubicSpline; quite slow, not significant improvement
    # _a = CubicSpline(t, a[:-1], extrapolate=False)
    # _H = CubicSpline(t, H[:-1], extrapolate=False)
    # _dzdt_z = CubicSpline(t[:-1], dzdt_z, extrapolate=False)

    sol = solve_ivp(get_diff_eq_ms, tspan, inits, args=(k, _dzdt_z, _a, _H), 
                     rtol=1e-12, atol=1e-12, method=SOLVER_MS, max_step=STEP_SIZE_MS/H[0])

    # power spectrum
    if evo:
        P = np.abs(sol.y[0])**2 * k**3 / (2*np.pi**2)
        return sol.t, P
    else:
        P = np.abs(sol.y[0, -1])**2 * k**3 / (2*np.pi**2)
        return P

def solve_power_spec(t, a, H, z, dzdt_z, ϕ, ϕᵢ, num=100, plot_fn=None, log_k_min=-3, log_k_max=4, silent=False):
    """
    solve MS equations for all k 
    using pathos multiprocessing
    """
    # for the start of USR phase
    aH_USR_i = np.interp(ϕᵢ, ϕ[::-1], (a*H)[::-1])
    # print((a*H)[0], (a*H)[-1], aH_USR_i)
    k = np.logspace(log_k_min, log_k_max, num) 
    
    def get_P(x):
        return solve_ms_single(x, t, a, H, z, dzdt_z, aH_USR_i)
    
    if silent:
        with Pool(processes=NUM_THREADS) as pool:
            P = np.array(list(pool.imap(get_P, k* aH_USR_i)))
    else:
        with Pool(processes=NUM_THREADS) as pool:
            P = np.array(list(tqdm.tqdm(pool.imap(get_P, k* aH_USR_i), total=len(k))))

    # print(np.amax(P))
    
    if plot_fn is not None:
        fig, ax = plt.subplots()
        ax.plot(k, P, c="k")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k/k_s$")
        ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}(k)$")
        plt.savefig(plot_fn, bbox_inches="tight")
        plt.close(fig)

    return k, P

def plot_power_evolution(model, sol):
    """
    plot the evolution of the "power spectrum" 
    in order to show the horzion-crossing effects 
    """

    N, ϕ, _ = sol
    t, a, H, z, dzdt_z, ϕ = get_ms_quant(model, sol)

    # for the start of USR phase
    aH_i = np.interp(model.ϕᵢ, ϕ[::-1], (a*H)[::-1])
    # aH_e = np.interp(model.ϕₑ, ϕ[::-1], (a*H)[::-1])
    # print((a*H)[0], (a*H)[-1], aH_USR_i)
    k = np.logspace(-2, 3, 6) 
    # print(k)
    
    fig, ax = plt.subplots()
    
    cmap = mpl.colormaps["viridis"]

    for i, ki in enumerate(k):
        # print(f"k/aH_i = {ki}, k/aH_e = {ki*aH_i/aH_e}")
        sol_t, P = solve_ms_single(ki*aH_i, t, a, H, z, dzdt_z, aH_i, evo=True)
        # print(sol_t, P)
        sol_N = np.interp(sol_t, t, N[:-1])
        c = cmap(i/(len(k)-1))
        # need to -1 to match the color of colorbar (don't know why...)
        ax.plot(sol_N, P, color=c)
        
        # draw the horizon crossing line
        k_aH = ki * aH_i / (a*H)
        hor_cross = np.interp(1.0, k_aH[::-1], N[::-1])
        # print(hor_cross)
        # ax.plot([hor_cross, hor_cross], [1e-10, 1], color=c, linestyle="dotted")
        ax.scatter(hor_cross, np.interp(hor_cross, sol_N, P), marker="*", color=c)

    # draw a band 
    ymin, ymax = ax.get_ylim()
    ax.fill_between([-model.ΔN, 0], ymin, ymax, color="grey", alpha=0.5, edgecolor=None)

    ax.set_xlabel("$N-N_e$")
    ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}(k, t)$")
    ax.set_yscale("log")
    ax.set_xlim((-12, 8))
    
    # plt.legend()
    # print(np.log10(np.amin(k)), np.log10(np.amax(k)))
    map = cm.ScalarMappable(norm=plt.Normalize(vmin=np.log10(np.amin(k)), vmax=np.log10(np.amax(k))), cmap=cmap)
    fig.colorbar(map, ax=ax, label=r"$\log_{10}(k/(aH)_i)$")
    plt.savefig(model.dn + "P_evo.pdf", bbox_inches="tight")
    plt.close()
