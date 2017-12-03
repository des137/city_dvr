from contextlib import contextmanager
from itertools import product
import time
import numpy as np
import numpy.linalg

np.set_printoptions(linewidth=300, suppress=True)


@contextmanager
def benchmark(name):
    start = time.time()
    yield
    end = time.time()

    print('{} took {:.2f} ms'.format(name, (end - start) * 1000.0))


# parameters defining the physical system
_EPS = np.finfo(float).eps
hbarc = 197.327
# nucleon mass
mn = 938.
# spherically symmetric oszillator strength
K = 12.

# lattice set-up
# partnbr: number of particles
partnbr = 1
# spacedims: spatial coordinate dimensions (e.g. cartesian x,y,z)
spacedims = 3
# length of a coordinate axis/box
Ltot = 7.
# left (L0) and right (LN1) boundary of a coordinate axis;
# these endpoints are not elements of the grid;
L0 = -Ltot / 2.
LN1 = Ltot / 2.
# number of grid points on axis
N = 7
# grid spacing
dr = float(Ltot) / float(N + 1)

# dimension of the Hilbert space
dv = (N - 1)**(spacedims * partnbr)
# print dimension of Hamilton matrix
print('dv^2: {0}'.format(dv**2))

# initialization of the two components of the Hamiltonian:
# H = mkinetic + mpotential
with benchmark('matrix initialization'):
    mpotential = np.zeros((dv, dv))
    mkinetic = np.zeros((dv, dv))
    print('shape(mpotential) = {}'.format(np.shape(mpotential)))
    print('shape(mkinetic) = {}'.format(np.shape(mkinetic)))


def fill_mkinetic(row=[], col=[]):
    return


grid = list(product(np.arange(1, N), repeat=spacedims * partnbr))

#
with benchmark("main calculation (ECCE: parallalize this)"):
    # column index
    colidx = 0
    for a in grid:
        # row index
        rowidx = 0
        for b in grid:
            mpotential[rowidx, colidx] = 0.0
            if np.array_equal(a, b):
                mpotential[rowidx, colidx] = 0.5 * K * sum([(
                    a[n] * dr + L0)**2 for n in range(spacedims)])
            for i in range(spacedims * partnbr):
                if ((a[i] == b[i]) &
                    (a[(i + 1) % spacedims] == b[(i + 1) % spacedims]) &
                    (a[(i + 2) % spacedims] == b[(i + 2) % spacedims])):
                    mkinetic[rowidx, colidx] += np.pi**2 / (
                        2. * Ltot**2) * (((2. * N**2 + 1) / 3.) -
                                         np.sin(np.pi * a[i] / float(N))**(-2))
                if ((a[i] != b[i]) &
                    (a[(i + 1) % spacedims] == b[(i + 1) % spacedims]) &
                    (a[(i + 2) % spacedims] == b[(i + 2) % spacedims])):
                    mkinetic[rowidx, colidx] += (-1)**(
                        a[i] - b[i]) * np.pi**2 / (2. * Ltot**2) * (
                            np.sin(np.pi * (a[i] - b[i]) / (2. * N))**
                            (-2) - np.sin(np.pi * (a[i] + b[i]) /
                                          (2. * N))**(-2))
            rowidx += 1
        colidx += 1

with benchmark("Diagonalization"):
    mkinetic *= hbarc**2 / (2 * mn)
    HAM = (mkinetic + mpotential)
    EV = np.sort(np.linalg.eigvals(HAM))

nzero = 0
for a in HAM.flatten():
    if abs(a) > _EPS:
        nzero += 1
print('Hamilton matrix: %d/%d non-zero entries' % (nzero, dv**2))
print('DVR:')
print(np.real(EV)[:6])
#
Eana = np.sort(
    np.array([[[(nx + ny + nz + 1.5) * hbarc * np.sqrt(K / mn)
                for nx in range(20)] for ny in range(20)]
              for nz in range(20)]).flatten())
print('ANA:')
print(Eana[:6])
print('%d/%d non-zero entries in mkinetic' % (nzero, dv**2))
