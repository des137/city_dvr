from contextlib import contextmanager
import time

@contextmanager
def benchmark(name):
    start = time.time()
    yield
    end = time.time()

    print('{} took {:.2f} ms'.format(name, (end-start)*1000.0))


# 1D harmonic-oszillator benchmark for the DVR
# comments:
# (i)  converged(!) results for the SINE DVR are offset
# but the splitting between energy EVs is correct
# (ii) the choice of parameters has significant effect
# on the grid sisze which is necessary for convergence
import numpy as np
import numpy.linalg
from itertools import product

_EPS = np.finfo(float).eps

np.set_printoptions(linewidth=300, suppress=True)

# parameters defining the physical system
# ----------------------------------------
hbarc = 197.327
mn = 938.
# spherically symmetric oszillator strength
K = 12.
#
# lattice set-up
# -----------------------------
# implemented DVR bases: ['SINE','SINEap']
dvrbasis = 'SINE'

# nbr: number of particles
nbr  = 1

# dimc: spatial coordinate dimensions (e.g. cartesian x,y,z)
dimc = 3

# Lr: length of each coordinate axis ("box size")
Lr = 7

# Nr: number of grid points on axis
Nr = 7

# for all box bases with f(a)=f(b)=0:
# endpoints are not elements of the grid!
dv = (Nr - 1)**3
print('dv^2: {0}'.format(dv**2))

# grid spacing
dr = float(Lr) / float(Nr + 1)

# offset, moves coordinate axis from (0, Lr) to (-Lr, Lr)
L0 = -float(Lr) / 2.

# we are going to fill matrices, POT and KIN, each initialized to 0
with benchmark('matrix initialization'):
    POT = np.zeros((dv, dv))
    KIN = np.zeros((dv, dv))

print('shape(POT) = {}, shape(KIN) = {}'.format(np.shape(POT), np.shape(KIN)))

grid       = list(product(np.arange(1, Nr),repeat=dimc*nbr))

#
with benchmark("main calculation (ECCE: parallalize this)"):
    s = 0
    for a in grid:
        r = 0
        for b in grid:
            POT[r, s] = 0.0
            if np.array_equal(a, b):
                POT[r,s] = 0.5*K*sum([(a[n]*dr+L0)**2 for n in range(dimc)])
            for i in range(dimc*nbr):
                if ((a[i]==b[i])&(a[(i+1)%dimc]==b[(i+1)%dimc])&(a[(i+2)%dimc]==b[(i+2)%dimc])):
                    KIN[r,s] += np.pi**2/(2.*Lr**2)*( ((2.*Nr**2+1)/3.)-np.sin(np.pi*a[i]/float(Nr))**(-2))
                if ((a[i]<>b[i])&(a[(i+1)%dimc]==b[(i+1)%dimc])&(a[(i+2)%dimc]==b[(i+2)%dimc])):
                    KIN[r,s] += (-1)**(a[i]-b[i])*np.pi**2/(2.*Lr**2)*(np.sin(np.pi*(a[i]-b[i])/(2.*Nr))**(-2)-np.sin(np.pi*(a[i]+b[i])/(2.*Nr))**(-2))
            r += 1
        s += 1

with benchmark("Diagonalization"):
        KIN *= hbarc**2 / (2 * mn)
        HAM = (KIN + POT)
        EV = np.sort(np.linalg.eigvals(HAM))

nzero = 0
for a in HAM.flatten():
    if abs(a)>_EPS:
        nzero += 1
print 'Hamilton matrix: %d/%d non-zero entries'%(nzero,dv**2)
print 'DVR:'
print np.real(EV)[:6]
#
Eana = np.sort(
    np.array([[[(nx + ny + nz + 1.5) * hbarc * np.sqrt(K / mn)
                for nx in range(20)] for ny in range(20)]
              for nz in range(20)]).flatten())
print 'ANA:'
print Eana[:6]
print '%d/%d non-zero entries in KIN'%(nzero,dv**2)