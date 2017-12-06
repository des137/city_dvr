from contextlib import contextmanager
from itertools import product
from scipy.sparse import *
from scipy.sparse.linalg import eigsh
import time
import numpy as np
import numpy.linalg

np.set_printoptions(linewidth=300, suppress=True, precision=7)

#delta function implementation necessary according to the number of particles
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
K = -150.

# lattice set-up
# partnbr: number of particles
partnbr = 2
# spacedims: spatial coordinate dimensions (e.g. cartesian x,y,z)
spacedims = 3
grdpointdim = partnbr * spacedims
# length of a coordinate axis/box
Ltot = 12
# left (L0) and right (LN1) boundary of a coordinate axis;
# these endpoints are not elements of the grid;
L0 = -Ltot / 2
LN1 = Ltot / 2

# number of grid points on axis
N = 4
c1 = np.pi / (2. * N)  # helper: calculate once instead of in every loop
c2 = ((2. * N**2 + 1) / 3.)
# grid spacing
dr = (LN1 - L0) / N

# dimension of the Hilbert space/grid
dv = (N - 1)**(spacedims * partnbr)

# calculate only the Nev lowest eigenvalues
Nev = 4

# initialization of the two components of the Hamiltonian:
# H = mkinetic + mpotential
with benchmark('matrix initialization'):
    try:
        # use regular matrix data type for low-dimensional problems
        mpotential = np.zeros((dv, dv))
        mkinetic = np.zeros((dv, dv))
    except:
        # for problems that exhaust available memory use sparse type
        # filling time increases by 2 orders of magnitude
        mpotential = dok_matrix((dv, dv))
        mkinetic = dok_matrix((dv, dv))
    print('Hamiltonian = MAT{}'.format(np.shape(mpotential)))


# kernel which evaluates kinetic energy at grid points row,col
def fill_mkinetic(row=[], col=[]):
    tmp = 0.0
    # sum the contributions to matrix element from each spatial dimension
    for i in range(grdpointdim):
        # diagonal elements
        if row == col:
            tmp += (c2 - np.sin(2 * c1 * row[i])**(-2))

        # off-diagonal elements in one dimension demand equality of all other
        # coordinates; all "fancy" version of the if clause, e.g., np.not_equal
        # np.delete, np.prod are an order of magnitude slower;
        if ((row[i] != col[i]) &
            (row[(i + 1) % grdpointdim] == col[(i + 1) % grdpointdim]) &
            (row[(i + 2) % grdpointdim] == col[(i + 2) % grdpointdim]) &
            (row[(i + 3) % grdpointdim] == col[(i + 3) % grdpointdim]) &
            (row[(i + 4) % grdpointdim] == col[(i + 4) % grdpointdim]) &
            (row[(i + 5) % grdpointdim] == col[(i + 5) % grdpointdim])):
            tmp += (-1)**(row[i] - col[i]) * (
                np.sin(c1 * (row[i] - col[i]))**
                (-2) - np.sin(c1 * (row[i] + col[i]))**(-2))
    return tmp


# kernel which evaluates potential at grid points row,col
def fill_mpotential(row=[], col=[]):
    if row == col:
        v1 = 0.5 * K * np.exp(-sum([(row[n] * dr + L0)**2
                                    for n in range(spacedims)]))
        v2 = 0.0
        if partnbr == 2:
            v2 = 0.5 * K * np.exp(-sum([(row[n + spacedims] * dr + L0)**2
                                        for n in range(spacedims)]))
        return v1 + v2
        #return 0.5 * K * sum([(row[n] * dr + L0)**2 for n in range(spacedims)])
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
    mkinetic *= np.pi**2 / (2. * (LN1 - L0)**2) * hbarc**2 / (2 * mn)
    HAM = (mkinetic + mpotential)

# calculate the eigenvalues of the sum of the Hamilton matrix (Hermitian)
with benchmark("Diagonalization -- full matrix structure (DVR)"):
    EV = np.sort(np.linalg.eigvalsh(HAM))
    print('Hamilton matrix: %d/%d non-zero entries\n' % (coo_matrix(HAM).nnz,
                                                         dv**2))
    print('DVR-full:', np.real(EV)[:Nev])

# calculate the lowest N eigensystem of the matrix in sparse format
with benchmark("Diagonalization -- sparse matrix structure (DVR)"):
    try:
        evals_small, evecs_small = eigsh(
            coo_matrix(HAM), Nev, which='SA', maxiter=5000)
        print('DVR-sparse:', evals_small)
    except:
        print('DVR-sparse: diagonalization did not converge/did fail.')