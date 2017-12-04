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
mn = 938
# spherically symmetric oszillator strength
K = 2

# lattice set-up
# partnbr: number of particles
partnbr = 1
# spacedims: spatial coordinate dimensions (e.g. cartesian x,y,z)
spacedims = 3
# length of a coordinate axis/box
Ltot = 12
# left (L0) and right (LN1) boundary of a coordinate axis;
# these endpoints are not elements of the grid;
L0 = -Ltot / 2
LN1 = Ltot / 2

# number of grid points on axis
N = 12
# grid spacing
dr = (LN1 - L0) / N

# dimension of the Hilbert space/grid
dv = (N - 1)**(spacedims * partnbr)

# initialization of the two components of the Hamiltonian:
# H = mkinetic + mpotential
with benchmark('matrix initialization'):
    mpotential = np.zeros((dv, dv))
    mkinetic = np.zeros((dv, dv))
    print('Hamiltonian = MAT{}'.format(np.shape(mpotential)))


# kernel which evaluates kinetic energy at grid points row,col
def fill_mkinetic(row=[], col=[]):
    tmp = 0.0
    # sum the contributions to matrix element from each spatial dimension
    for i in range(len(row)):
        # diagonal elements
        if row == col:
            tmp += ((
                (2. * N**2 + 1) / 3.) - np.sin(np.pi * row[i] / float(N))**
                    (-2))

        # off-diagonal elements in one dimension demand equality of all other
        # coordinates
        if ((row[i] != col[i]) &
            (row[(i + 1) % spacedims] == col[(i + 1) % spacedims]) &
            (row[(i + 2) % spacedims] == col[(i + 2) % spacedims])):
            tmp += (-1)**(row[i] - col[i]) * (
                np.sin(np.pi * (row[i] - col[i]) / (2. * N))**
                (-2) - np.sin(np.pi * (row[i] + col[i]) / (2. * N))**(-2))
    return tmp


# kernel which evaluates potential at grid points row,col
def fill_mpotential(row=[], col=[]):
    if row == col:
        return 0.5 * K * sum([(row[n] * dr + L0)**2 for n in range(spacedims)])
    else:
        return 0.0


# writing of the Hamilton matrix contains two basic loops
# column index b and row index a; e.g.:2 particles: a=(x1,y1,z1,x2,y2,y3) and
# b=(x1',y1',z1',x2',y2',y3')
with benchmark("matrix filling"):
    # column index
    colidx = 0
    # row loop; each grid point specifies <spacedims> coordinates per particle
    for a in product(np.arange(1, N), repeat=spacedims * partnbr):
        # row index
        rowidx = 0
        # column loop;
        for b in product(np.arange(1, N), repeat=spacedims * partnbr):
            mpotential[rowidx, colidx] = fill_mpotential(a, b)
            mkinetic[rowidx, colidx] = fill_mkinetic(a, b)
            rowidx += 1
        colidx += 1

# calculate the eigenvalues of the sum of the kinetic and potential matrix
with benchmark("Diagonalization"):
    mkinetic *= np.pi**2 / (2. * (LN1 - L0)**2) * hbarc**2 / (2 * mn)
    HAM = (mkinetic + mpotential)
    EV = np.sort(np.linalg.eigvals(HAM))

#
nzero = 0
for a in HAM.flatten():
    if abs(a) > _EPS:
        nzero += 1
print('Hamilton matrix: %d/%d non-zero entries' % (nzero, dv**2))
print('DVR:')
print(np.real(EV)[:6])

Eana = np.sort(
    np.array([[[(nx + ny + nz + 1.5) * hbarc * np.sqrt(K / mn)
                for nx in range(20)] for ny in range(20)]
              for nz in range(20)]).flatten())
print('ANA:')
print(Eana[:6])