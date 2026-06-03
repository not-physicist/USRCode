import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.integrate import solve_ivp

import scienceplots
plt.style.use("science")

from ..constants import STEP_SIZE_EOM, SOLVER_EOM
from ..common import timer

def _get_H2(ϕ, dϕ, _V):
    return _V(ϕ) / (3 - dϕ**2/2)

def _get_f(t, y, _V, _dV):
    """
    defines the inflaton EOM
    number of e-fold as time variable
    """
    # print(y)
    phi = y[0]
    phi_d = y[1]
    a = y[2]
    
    H2 = _get_H2(phi, phi_d, _V)
    if H2 <= 0:
        return [0, 0, 0]

    dydt = [phi_d, phi_d**3/2 - 3*phi_d - _dV(phi)/H2, a]

    return dydt

def get_Δϵ(model, sol):
    """
    by solving the EOM using the taylor expanded EOM, 
    check if the new potential with β is OK.
    (If the solutions are similar for the beginning)
    """
    N, ϕ, π = sol
    # print(N)
    # N = sol.t 
    # ϕ = sol.y[0]
    # π = sol.y[1]

    _V = lambda x: model.V0*(1 + np.sqrt(2*model.ϵ2)*(x-model.ϕₑ) + model.η2/2*(x-model.ϕₑ)**2)
    _dV = lambda x: model.V0*(np.sqrt(2*model.ϵ2) + model.η2*(x-model.ϕₑ))
    tspan = [0, 100]
    def get_e1(t, y): return np.abs(y[1]**2/2) - 1
    get_e1.terminal = True
    initials = [model.ϕₑ, np.interp(0.0, N, π), 1.0]
    # print(model.πₑ, np.interp(0.0, N, π))

    sol2 = solve_ivp(lambda t, y: _get_f(t, y, _V, _dV), tspan, initials, events=get_e1, method='DOP853', max_step=0.01)

    N_new = sol2.t
    ϕ_new = sol2.y[0]
    π_new = sol2.y[1]
    # print(model.πₑ, π_new[0])
    
    π_interp = np.interp(N_new, N, π)
    Δϵ = np.abs((π_interp**2 - π_new**2) / π_interp**2)
    # print(π_interp[0:10], π_new[0:10], Δϵ)

    ϕ_interp = np.interp(N_new, N, ϕ)
    Δϕ = np.abs((ϕ_interp**2 - ϕ_new**2) / ϕ_interp**2)

    return N_new, Δϵ, Δϕ


def plot_eom(model, sol):
    """
    plot the numerical solution to inflaton EOM
    """
    N, ϕ, π = sol
    # N = sol.t 
    # ϕ = sol.y[0]
    # π = sol.y[1]
    # N2, ϕ2, π2 = model.sol

    fig, (ax1, ax2) = plt.subplots(figsize=(4, 3), ncols=2)
    ax1.plot(N, ϕ/model.ϕ_cmb, label=r"$\phi/\phi_\text{cmb}$")
    # ax1.plot(N2, ϕ2/model.ϕ_cmb, label=r"$\phi_2/\phi_\text{cmb}$")
    ax1.set_xlabel(r"$N$")
    # ax1.set_yscale("log")
    ax1.legend()

    ax2.plot(N, np.log10(π**2/2), label=r"$\log(\phi'^2/2)$")
    ax2.plot([N[0], N[-1]], np.ones(2)*np.log10(model.ϵ2), label=r"$\log(\epsilon_{V, 2})$")
    
    N_new, Δϵ, Δϕ = get_Δϵ(model, sol)
    ax2.plot(N_new, np.log10(Δϵ), label=r"$\log (\Delta_\text{rel} \epsilon)$")
    ax2.plot(N_new, np.log10(Δϕ), label=r"$\log (\Delta_\text{rel} \phi)$")

    # ax.set_yscale("log")
    ax2.set_xlabel(r"$N - N_e$")
    ax2.legend(loc="upper left")


    plt.savefig(model.dn + "eom.pdf", bbox_inches="tight")
    plt.close(fig)


def get_eternal(ϕ, dϕ, _V):
    """
    check if we have entered eternal inflation
    """
    H2 = _get_H2(ϕ, dϕ, _V) 
    x = np.amin(np.abs(2*np.pi*dϕ / np.sqrt(H2)))
    # print("dϕ/H = ", x)
    if x <= 1:
        raise Exception("ETERNAL INFLATION!!!!!")

# @timer
def solve_eom(model, toplot=False):
    """
    solve the complete EOM of inflaton field
    """
    t0 = perf_counter()

    _V = lambda x: model.get_V(x)
    _dV = lambda x: model.get_dV(x)
    tspan = [0, 100]
    def get_e1(t, y): return np.abs(y[1]**2/2) - 1
    get_e1.terminal = True
    # π_cmb = -np.sqrt(2*get_ϵV1(model.ϕ_cmb_i, model.ϵ, model.η))
    initials = [model.ϕ_cmb, model.π_cmb, 1.0]

    # sol = solve_ivp(lambda t, y: _get_f(t, y, _V, _dV), tspan, initials, events=get_e1, rtol=1e-12)
    sol = solve_ivp(lambda t, y: _get_f(t, y, _V, _dV), tspan, initials, events=get_e1, method=SOLVER_EOM, max_step=STEP_SIZE_EOM)
    t1 = perf_counter()

    if sol.status == 1:
        N_e = np.interp(model.ϕₑ, sol.y[0][::-1], sol.t[::-1])

        get_eternal(sol.y[0], sol.y[1], _V)
       
        if toplot:
            print("Solving EOM took %.2f s" % (t1 - t0))
            plot_eom(model, (sol.t - N_e, sol.y[0], sol.y[1]))

        return sol.t-N_e, sol.y[0], sol.y[1]
    else:
        print("Failure in solve_eom()")
        print(sol.t, sol.y[0])
        print("Initial velocity at beginning of USR: ", -np.sqrt(2*model.ϵ), 
              np.interp(model.ϕᵢ, sol.y[0][::-1], sol.y[1][::-1]))
        raise ValueError
