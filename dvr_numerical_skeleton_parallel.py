from contextlib import contextmanager
from itertools import product
import time

import numpy as np
import numpy.linalg
from mpi4py import MPI
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from dvr import calc_mkinetic, calc_mpotential

np.set_printoptions(linewidth=300, suppress=True, precision=7)


@contextmanager
def benchmark(name):
    start = time.time()
    yield
    end = time.time()

    print('{} took {:.2f} ms'.format(name, (end - start) * 1000.0))


# multi-processor bookkeeping
comm = MPI.COMM_WORLD
# who am I?
mpi_rank = comm.Get_rank()
# how many of us are there?
mpi_size = comm.Get_size()

# parameters defining the physical system
_EPS = np.finfo(float).eps
hbarc = 197.327
# nucleon mass
mn = 938
# Gaussian 2-body interaction
lec = -505.1 * 2
beta = 4.

# lattice set-up
PARTNBR = 2  # number of particles
SPACEDIMS = 3  # spatial coordinate dimensions (e.g. cartesian x,y,z)
GRDPOINTDIM = PARTNBR * SPACEDIMS  # length of a coordinate axis/box

Ltot = 6
# left (L0) and right (LN1) boundary of a coordinate axis;
# these endpoints are not elements of the grid;
L0 = -Ltot / 2
LN1 = Ltot / 2

# number grid segments = (grid points - 1)
N = 5
c1 = np.pi / (2. * N)  # helper: calculate once instead of in every loop
c2 = ((2. * N**2 + 1) / 3.)
# grid spacing
dr = (LN1 - L0) / N

# dimension of the Hilbert space/grid
dv = (N - 1)**(SPACEDIMS * PARTNBR)
# number of rows which are filled by a single processor
if mpi_rank == 0:
    mpi_nbrrows = dv
else:
    mpi_nbrrows = int(dv / (mpi_size - 1))
    if mpi_rank == mpi_size - 1:
        mpi_nbrrows += dv % (mpi_size - 1)
    # processor n fills rows [mpi_row_offset , mpi_row_offset + mpi_nbrrows]
    mpi_row_offset = (mpi_rank - 1) * int(dv / (mpi_size - 1))

# calculate only the Nev lowest eigenvalues
Nev = 4

# initialization of the two components of the Hamiltonian:
# H = mkinetic + mpotential
mpotential = np.zeros((dv, dv))
mkinetic = np.zeros((dv, dv))
mhamilton = np.zeros((dv, dv))

# writing of the Hamilton matrix contains two basic loops
# column index b and row index a; e.g.:2 particles: a=(x1,y1,z1,x2,y2,y3) and
# b=(x1',y1',z1',x2',y2',y3')
extime = 0
if mpi_rank != 0:
    start = time.time()
    # column index
    colidx = 0
    # row loop; each grid point specifies <SPACEDIMS> coordinates per particle
    for a in product(np.arange(1, N), repeat=SPACEDIMS * PARTNBR):
        # row index
        rowidx = mpi_row_offset
        # column loop;
        for b in list(
                product(np.arange(1, N), repeat=SPACEDIMS *
                        PARTNBR))[mpi_row_offset:mpi_row_offset + mpi_nbrrows]:
            mpotential[rowidx, colidx] = calc_mpotential(a, b, GRDPOINTDIM)
            mkinetic[rowidx, colidx] = calc_mkinetic(a, b, SPACEDIMS)
            rowidx += 1
        colidx += 1
    mkinetic *= np.pi**2 / (2. * (LN1 - L0)**2) * hbarc**2 / (2 * mn)
    mhamilton = (mkinetic + mpotential)
    end = time.time()
    extime = (end - start)
    print('cpu%d: sub-matrix filled in %4.4fs' % (mpi_rank, extime * 1000))
comm.Barrier()
mhamilton = comm.gather(mhamilton, root=0)
fill_time = []
fill_time = comm.gather(extime, root=0)
if mpi_rank == 0: print('root: total fill time %4.4fs' % sum(fill_time * 1000))

# calculate the eigenvalues of the sum of the Hamilton matrix (Hermitian)
if mpi_rank == 0:
    for n in range(1, mpi_size):
        mhamilton[0] += mhamilton[n]
    with benchmark("Diagonalization -- full matrix structure (DVR)"):
        EV = np.sort(np.linalg.eigvalsh(mhamilton[0]))
        print('Hamilton matrix: %d/%d non-zero entries\n' %
              (coo_matrix(mhamilton[0]).nnz, dv**2))
        print('DVR-full:', np.real(EV)[:Nev])

    # calculate the lowest N eigensystem of the matrix in sparse format
    with benchmark("Diagonalization -- sparse matrix structure (DVR)"):
        try:
            evals_small, evecs_small = eigsh(
                coo_matrix(mhamilton[0]), Nev, which='SA', maxiter=5000)
            print('DVR-sparse:', evals_small)
        except:
            print('DVR-sparse: diagonalization did not converge/did fail.')
