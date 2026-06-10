# Originally: https://scipy-cookbook.readthedocs.io/items/SphericalBesselZeros.html

### recursive method: computes zeros ranges of Jn(r,n) from zeros of Jn(r,n-1)
### (also for zeros of (rJn(r,n))')
### pros : you are certain to find the right zeros values;
### cons : all zeros of the n-1 previous Jn have to be computed;
### note : Jn(r,0) = sin(r)/r

import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq
import pickle

def Jn_zeros(n,nt):
    zerosj = np.zeros((n+1, nt))
    zerosj[0] = np.arange(1,nt+1)*np.pi
    points = np.arange(1,nt+n+1)*np.pi
    racines = np.zeros(nt+n)
    for i in range(1,n+1):
        for j in range(nt+n-i):
            foo = brentq(lambda x: spherical_jn(i, x), points[j], points[j+1])
            racines[j] = foo
    points = racines
    zerosj[i][:nt] = racines[:nt]
    return (zerosj)

# 300 zeros ~ x=1000: it should be sufficient
n = 1 
nt = 300
jnz = Jn_zeros(n,nt)
# j0z = jnz[0]
# j1z = jnz[1]

fn = "data/jnz.npz"
np.savez(fn, j0z=jnz[0], j1z=jnz[1])
# fn = "data/jnz.pkl"
# pickle.dump(jnz, open(fn, "wb"))
