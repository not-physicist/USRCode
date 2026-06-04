"""
more like config file
"""
import psutil 

# default: use all the cpu threads 
NUM_THREADS = psutil.cpu_count()

# step size for ODE solvers; in H^-1 or num of efolds
STEP_SIZE_MODELCLASS = 1e-2
# STEP_SIZE_MODELCLASS = 2e-3
SOLVER_MODELCLASS = 'DOP853'

# solving inflaton EOM requires such small step size, for the USR phase likely
STEP_SIZE_EOM = 2e-3
SOLVER_EOM = "DOP853" 

# solving MS equations: 1e-2 seems a bit too large, need smaller for nice plots
STEP_SIZE_MS = 5e-3
SOLVER_MS = 'DOP853'

MODEL_CACHE_FN = "model.pkl"

POWER_SPEC_FN = "PS.pkl"
POWER_SPEC_PLOT_FN = "PS.pdf"

EOM_SOL_FN = "EOM_sol.pkl"
EOM_PLOT_FN = "EOM.pdf"

BH_F_FN = "f_m.pkl" 

GW_SPEC_FN = "Omega_f.pkl"
