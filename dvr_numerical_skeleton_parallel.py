from itertools import product

import numpy as np
import numpy.linalg
from mpi4py import MPI
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from dvr import calc_mkinetic, calc_mpotential
from physics import NUCLEON_MASS, PLANCS_CONSTANT, C, JOULE_PER_EV
from util import benchmark

np.set_printoptions(linewidth=300, suppress=True, precision=7)


# multi-processor bookkeeping
comm = MPI.COMM_WORLD
# who am I?
mpi_rank = comm.Get_rank()
# how many of us are there?
mpi_size = comm.Get_size()

# parameters defining the physical system
_EPS = np.finfo(float).eps

#       h*c/(2*pi) in MeV*fm
HBARC = PLANCS_CONSTANT * C / (2 * np.pi) * JOULE_PER_EV * 10e12 * 10e-6

# Gaussian 2-body interaction
LEC = -505.1 * 2
BETA = 4.0

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
N = 4
c1 = np.pi / (2. * N)  # helper: calculate once instead of in every loop
c2 = ((2. * N**2 + 1) / 3.)
GRID_SPACING = (LN1 - L0) / N

# dimension of the Hilbert space/grid
dv = (N - 1)**(SPACEDIMS * PARTNBR)
# number of rows which are filled by a single processor
mpi_nbrrows = int(dv / (mpi_size - 1))
if mpi_rank == mpi_size - 1:
    mpi_nbrrows += dv % (mpi_size - 1)
# processor n fills rows [mpi_row_offset , mpi_row_offset + mpi_nbrrows]
mpi_row_offset = mpi_rank * int(dv / (mpi_size - 1))

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
with benchmark("cpu%d: Matrix filling" % mpi_rank):
    # column index
<<<<<<< HEAD
    colidx = mpi_row_offset
    print('cpu%d: rows %d -> %d' % (mpi_rank, mpi_row_offset,
                                    mpi_row_offset + mpi_nbrrows))
    # row loop; each grid point specifies <spacedims> coordinates per particle
    for a in list(
            product(np.arange(1, N), repeat=spacedims *
                    partnbr))[mpi_row_offset:mpi_row_offset + mpi_nbrrows]:
=======
    colidx = 0
    # row loop; each grid point specifies <SPACEDIMS> coordinates per particle
    for a in product(np.arange(1, N), repeat=SPACEDIMS * PARTNBR):
>>>>>>> 925e980509de94dc4a8879b8eefe0800f1d175b7
        # row index
        rowidx = 0
        # column loop;
<<<<<<< HEAD
        for b in product(np.arange(1, N), repeat=spacedims * partnbr):
            mpotential[rowidx, colidx] = fill_mpotential(a, b)
            mkinetic[rowidx, colidx] = fill_mkinetic(a, b)
=======
        for b in list(
                product(np.arange(1, N), repeat=SPACEDIMS *
                        PARTNBR))[mpi_row_offset:mpi_row_offset + mpi_nbrrows]:
            mpotential[rowidx, colidx] = calc_mpotential(a, b, SPACEDIMS, GRID_SPACING, LEC, BETA)
            mkinetic[rowidx, colidx] = calc_mkinetic(a, b, GRDPOINTDIM)
>>>>>>> 925e980509de94dc4a8879b8eefe0800f1d175b7
            rowidx += 1
        colidx += 1
    mkinetic *= np.pi**2 / (2. * (LN1 - L0)**2) * HBARC**2 / (2 * NUCLEON_MASS)
    mhamilton = (mkinetic + mpotential)

comm.Barrier()
mhamilton = comm.gather(mhamilton, root=0)

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
