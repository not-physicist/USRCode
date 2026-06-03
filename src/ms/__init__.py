from . import EOM 
from .ModelClass import USRManuel
from . import ms

from ..constants import POWER_SPEC_FN, EOM_SOL_FN, POWER_SPEC_PLOT_FN
from ..common import timer

import pickle
from pathlib import Path
from os.path import join

def save_solve_all(model, toplot=False, N_ms=200, force_new=False, log_k_min=-3, log_k_max=4, silent=False):
    """
    solve the background and compute the power spectrum
    """
    fn_eom = join(model.dn, EOM_SOL_FN)
    fn_ps = join(model.dn, POWER_SPEC_FN)

    if Path(fn_ps).exists() and not force_new:
        print("File cache exists. Skipping...")
        return None

    sol = EOM.solve_eom(model, toplot)
    backgrounds = ms.get_ms_quant(model, sol)

    if toplot:
        fn = join(model.dn, POWER_SPEC_PLOT_FN)
    else:
        fn = None

    k, P = ms.solve_power_spec(*backgrounds, model.ϕᵢ, plot_fn=fn, num=N_ms, log_k_min=log_k_min, log_k_max=log_k_max, silent=silent)
    
    with open(fn_eom, "wb") as f:
        pickle.dump(sol, f)

    with open(fn_ps, "wb") as f:
        pickle.dump((k, P), f)
