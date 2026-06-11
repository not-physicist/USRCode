from . import EOM 
from .ModelClass import USRManuel
from . import ms

from ..constants import POWER_SPEC_FN, EOM_SOL_FN, POWER_SPEC_PLOT_FN
from ..common import timer

import pickle
from pathlib import Path
from os.path import join
import numpy as np

from ..constants import PEAK_K

def save_solve_all(model, toplot=False, N_ms=200, force_new=False, log_k_min=-3, log_k_max=4, silent=False, tosave=False):
    """
    solve the background and compute the power spectrum
    """
    # fn_eom = join(model.dn, EOM_SOL_FN)
    if model.dn is None and tosave:
        raise RuntimeError("model.dn is set to None but tosave is True!")

    if model.dn is not None:
        fn_ps = join(model.dn, POWER_SPEC_FN)

        if Path(fn_ps).exists() and not force_new:
            print("File cache exists. Skipping...")
            return pickle.load(open(fn_ps, "rb"))

    sol = EOM.solve_eom(model, toplot)
    backgrounds = ms.get_ms_quant(model, sol)
    # itp = lambda x: np.interp(x, sol[1][::-1], sol[0][::-1])
    # print("ΔN from EOM.py: ", itp(model.ϕᵢ) - itp(model.ϕₑ))

    if toplot:
        fn = join(model.dn, POWER_SPEC_PLOT_FN)
    else:
        fn = None

    k, P = ms.solve_power_spec(*backgrounds, model.ϕᵢ, plot_fn=fn, num=N_ms, log_k_min=log_k_min, log_k_max=log_k_max, silent=silent)
    
    if tosave:
        # with open(fn_eom, "wb") as f:
        #     pickle.dump(sol, f)

        with open(fn_ps, "wb") as f:
            pickle.dump((k, P), f)

    return k, P

def get_P_max(model):
    """
    only get the maximum of the power spectrum. use PEAK_K as the location
    """
    sol = EOM.solve_eom(model, False)
    t, a, H, z, dzdt_z, ϕ = ms.get_ms_quant(model, sol)
    aH_USR_i = np.interp(model.ϕᵢ, ϕ[::-1], (a*H)[::-1])
    
    Pmax = ms.solve_ms_single(PEAK_K*aH_USR_i, t, a, H, z, dzdt_z, aH_USR_i, False)
    return Pmax
