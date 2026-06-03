from .ParaClass import ParaClassMod
from .ParaClass import ParaClass
from .f_m import get_f_total
from .f_m import get_f

from .models import USR_full
from .models import gaussian
from .models import USR_smooth
from .models import USR_only

# from .common import get_zero
# from .common import get_one

# from ... import common

from .P_C_l import plot_integrand_2D

from .P_zeta import plot_P_ζ_full as plot_P_ζ_full
from .P_C_l import plot_full as plot_P_Cl_full
from .f_m import plot_f_eta

def plot_Gow():
    A = 4.05e-3  # to get f_PBH ~ 1
    #  A = 0.25
    Δ = 0.3
    k_star = 1.56e13  # Mpc-1
    k_star_r = 2.74
    K = 1
    C_c = 0.587
    Z = 6.88e16
    para = ParaClass(A, Δ, k_star, k_star_r, K, C_c, Z)

    plot_P_ζ_full(para)
    # plot_P_Cl_full(para)
    # plot_f_eta(para)
