from contextlib import contextmanager
from itertools import product
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from mpi4py import MPI
import time
import numpy as np
import numpy.linalg

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
# partnbr: number of particles
partnbr = 2
# spacedims: spatial coordinate dimensions (e.g. cartesian x,y,z)
spacedims = 3
grdpointdim = partnbr * spacedims
# length of a coordinate axis/box
Ltot = 6
# left (L0) and right (LN1) boundary of a coordinate axis;
# these endpoints are not elements of the grid;
L0 = -Ltot / 2
LN1 = Ltot / 2

# number grid segments = (grid points - 1)
N = 4
c1 = np.pi / (2. * N)  # helper: calculate once instead of in every loop
c2 = ((2. * N**2 + 1) / 3.)
# grid spacing
dr = (LN1 - L0) / N

# dimension of the Hilbert space/grid
dv = (N - 1)**(spacedims * partnbr)
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
        return lec * np.exp(-beta * sum([(row[n] - row[n + spacedims] * dr)**2
                                         for n in range(spacedims)]))
    else:
        return 0.0


# writing of the Hamilton matrix contains two basic loops
# column index b and row index a; e.g.:2 particles: a=(x1,y1,z1,x2,y2,y3) and
# b=(x1',y1',z1',x2',y2',y3')
with benchmark("cpu%d: Matrix filling" % mpi_rank):
    # column index
    colidx = mpi_row_offset
    print('cpu%d: rows %d -> %d' % (mpi_rank, mpi_row_offset,
                                    mpi_row_offset + mpi_nbrrows))
    # row loop; each grid point specifies <spacedims> coordinates per particle
    for a in list(
            product(np.arange(1, N), repeat=spacedims *
                    partnbr))[mpi_row_offset:mpi_row_offset + mpi_nbrrows]:
        # row index
        rowidx = 0
        # column loop;
        for b in product(np.arange(1, N), repeat=spacedims * partnbr):
            mpotential[rowidx, colidx] = fill_mpotential(a, b)
            mkinetic[rowidx, colidx] = fill_mkinetic(a, b)
            rowidx += 1
        colidx += 1
    mkinetic *= np.pi**2 / (2. * (LN1 - L0)**2) * hbarc**2 / (2 * mn)
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