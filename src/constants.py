"""
more like config file
"""
import psutil 

# default: use all the cpu threads 
NUM_THREADS = psutil.cpu_count()

# BUG: h=-1 requires 1e-4 and 1e-4

# step size for ODE solvers; in H^-1 or num of efolds
STEP_SIZE_MODELCLASS = 1e-4
# STEP_SIZE_MODELCLASS = 2e-3
SOLVER_MODELCLASS = 'DOP853'
# rtol for (root) finding ϕₑ
EOM_PHI_E_RTOL = 1e-3

# not needed now
# solving inflaton EOM requires such small step size, for the USR phase likely
# larger than 1e-3: P_max(ΔN) becomes wiggly
# now 5e-4: P_max and f_tot are very sensitive to ΔN; often need to do ~1e-4 increments in ΔN to find the right P_max and f_tot
STEP_SIZE_EOM = 1e-4
# STEP_SIZE_EOM_USR = 1e-5
SOLVER_EOM = "DOP853" 


# solving MS equations: 1e-2 seems a bit too large, need smaller for nice plots
STEP_SIZE_MS = 1e-3
SOLVER_MS = 'DOP853'

# k/k_i at P = P_max
PEAK_K = 3.29297126
# actually, it probably dones't matter if PEAK_K is accurate 
# as long as it is used consistently for all interpolations

MODEL_CACHE_FN = "model.pkl"

# POWER_SPEC_FN = "PS.pkl"
POWER_SPEC_FN = "PS.npz"
POWER_SPEC_PLOT_FN = "PS.pdf"

EOM_SOL_FN = "EOM_sol.pkl"
EOM_PLOT_FN = "EOM.pdf"

# BH_F_FN = "f_m.pkl" 
BH_F_FN = "f_m.npz" 

GW_PLOT_FN = "GW_Omega.pdf"
GW_SPEC_FN = "Omega_f.npz"
