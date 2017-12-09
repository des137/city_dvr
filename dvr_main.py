from itertools import product

import numpy as np
import numpy.linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from dvr import calc_mkinetic, calc_mpotential
from physics import NUCLEON_MASS, PLANCS_CONSTANT, C, JOULE_PER_EV
from util import benchmark

np.set_printoptions(linewidth=300, suppress=True, precision=7)

#       h*c/(2*pi) in MeV*fm
HBARC = PLANCS_CONSTANT * C / (2 * np.pi) * JOULE_PER_EV * 10e12 * 10e-6

# Gaussian 2-body interaction
LEC = -505.1 * 2
BETA = 4.0

# lattice set-up
PARTNBR = 2  # number of particles
SPACEDIMS = 3  # spatial coordinate dimensions (e.g. cartesian x,y,z)
BOXSIZE = 6  # physical length of one spatial dimension (in Fermi)
N_SEGMENTS = 3  # number of segments in which each spatial axis is discretized

N_EIGENV = 4  # number of eigenvalues to be calculated with <eigsh>


def calc_mhamilton(n_particle, dim_space, l_box, n_seg, lec, beta):
    """ Function returns the Hamilton matrix; 
        it contains two basic loops over the
        column index b and the row index a;
        example: 2 particles: a=(x1,y1,z1,x2,y2,y3) and
                              b=(x1',y1',z1',x2',y2',y3')

    :n_particle: number of particles
    :dim_space: spatial (Cartesian) dimensions
    :l_box: physical extent of one spatial dimension
    :n_seg: number of segments each coordinate is devided into
    :lec,beta: parameters specifying the interaction potential

    :return: full Hamilton matrix 
    """
    dim_grdpoint = n_particle * dim_space  # dimension of a single coordinate point
    l0 = -l_box / 2  # left position of the box
    lnp1 = l_box / 2  # right end of the box ; ln+1-l0 = l_box
    grd_spacing = l_box / n_seg  # width of a segment (in Fermi)
    dim_h = (n_seg - 1)**dim_grdpoint  # dim(Hilbert space/grid)

    mpotential = np.zeros((dim_h, dim_h))
    mkinetic = np.zeros((dim_h, dim_h))
    mhamilton = np.zeros((dim_h, dim_h))

    colidx = 0
    # row loop; each grid point specifies <spacedims> coordinates per particle
    for a in product(np.arange(1, n_seg), repeat=dim_grdpoint):
        rowidx = 0
        # column loop;
        for b in product(np.arange(1, n_seg), repeat=dim_grdpoint):
            mpotential[rowidx, colidx] = calc_mpotential(
                a, b, dim_space, grd_spacing, lec, beta)
            mkinetic[rowidx, colidx] = calc_mkinetic(a, b, dim_grdpoint, n_seg)
            rowidx += 1
        colidx += 1
    mkinetic *= np.pi**2 / (2. * (l_box)**2) * HBARC**2 / (2 * NUCLEON_MASS)
    mhamilton = (mkinetic + mpotential)
    return mhamilton


""" main section of the program
    1. set up the Hamiltonian
    2. full Diagonalization
    3. approximate Diagonalization (extract only the N_EIGENV lowest EV's)
"""

ham = []
with benchmark("Matrix filling"):
    ham = calc_mhamilton(PARTNBR, SPACEDIMS, BOXSIZE, N_SEGMENTS, LEC, BETA)

with benchmark("Diagonalization -- full matrix structure (DVR)"):
    # calculate the eigenvalues of the sum of the Hamilton matrix (Hermitian)
    EV = np.sort(np.linalg.eigvalsh(ham))
    print('Hamilton matrix: %d/%d non-zero entries\n' %
          (coo_matrix(ham).nnz, (N_SEGMENTS - 1)**(PARTNBR * SPACEDIMS)**2))
    print('DVR-full:', np.real(EV)[:N_EIGENV])

with benchmark("Diagonalization -- sparse matrix structure (DVR)"):
    # calculate the lowest N eigensystem of the matrix in sparse format
    try:
        evals_small, evecs_small = eigsh(
            coo_matrix(ham), N_EIGENV, which='SA', maxiter=5000)
        print('DVR-sparse:', evals_small)
    except:
        print('DVR-sparse: diagonalization did not converge/did fail.')