import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from dvr import calc_grid, calc_Ekin, calc_potential
from physics import NUCLEON_MASS, PLANCS_CONSTANT, C, JOULE_PER_EV, HBARC, eigenvalues_harmonic_osci
from util import benchmark

np.set_printoptions(linewidth=300, suppress=True, precision=5)

# single-particle parameters and other physical constants
MASS = NUCLEON_MASS

# Gaussian 2-body interaction
LEC = -505.1
BETA = 4.0
POT_GAUSS = ['GAUSS', BETA, LEC]

# isotropic harmonic oscillator
OMEGA = 1.
X_EQ = 0.
POT_HO = ['HO', X_EQ, MASS / HBARC, OMEGA]

# lattice set-up
PARTNBR = 2  # number of particles
SPACEDIMS = 2  # spatial coordinate dimensions (e.g. Cartesian x,y,z)
BASIS_DIM = 5  # (dim of variational basis) = (nbr of grid points) = (nbr of segments - 1)

# specify the variational basis
BOX_SIZE = 6  # physical length of one spatial dimension (in Fermi); relevant for specific bases, only!
BOX_ORIGIN = -BOX_SIZE / 2.
BASIS_SINE = ['SINE', [SPACEDIMS, BOX_SIZE, BOX_ORIGIN, MASS / HBARC]]
BASIS_HO = ['HO', [SPACEDIMS, X_EQ, (MASS / HBARC), OMEGA]]
# each axis is devided into = (number of grid points) - 1

N_EIGENV = 14  # number of eigenvalues to be calculated with <eigsh>


def calc_mhamilton(n_particle, dim_space, dim_basis, specs_basis,
                   specs_potential):
    """ Function returns the Hamilton matrix; 

        :n_particle: number of particles
        :dim_space: spatial (Cartesian) dimensions
        :dim_basis: variational-basis dim = number of segments each coordinate is divided into
        :specs_potential: parameters specifying the interaction potential
        :specs_basis: parameters specifying the basis

        :return: full Hamilton matrix in D(iscrete) V(ariable) R(epresentation)
    """
    # dimension of a single coordinate point; e.g., 2D 2Part: (x1,y1,x2,y2)
    # D spatial dimensions for each of the N particles;
    dim_grdpoint = n_particle * dim_space
    # each component of a grid point takes dim_basis discrete values
    # e.g. x1 \in {x_1,...,x_dim_basis} where x_1 is an eigenvalue of the position matrix
    dim_h = dim_basis**dim_grdpoint

    # initialize empty matrices (might have to be "sparsed" for larger dim. problems)
    mpotential = np.zeros((dim_h, dim_h))
    mkinetic = np.zeros((dim_h, dim_h))
    mhamilton = np.zeros((dim_h, dim_h))

    # obtain eigensystem of the position operator in the basis of choice
    # eigenvalues '=' grid points ; transformation matrix necessary for Ekin
    # STATUS: one basis for all coordinates (future: xy->HO, z->SINE)
    coordOP_evSYS = calc_grid(dim_basis, specs_basis)

    # calculate potential and kinetic-energy matrices for a chosen basis
    # STATUS: for each additional basis, the matrices need to be specified in this function!
    mkinetic = calc_Ekin(dim_basis, n_particle, specs_basis, coordOP_evSYS)
    # STATUS: the potential matrix is assumed to be diagonal (future: OPE+B => potential has non-zero offdiagonal elements)
    mpotential = np.diag(
        calc_potential(n_particle, dim_space, specs_potential, coordOP_evSYS))

    mhamilton = (mkinetic + mpotential)
    return mhamilton


""" main section of the program
    1. set up the Hamiltonian
    2. full Diagonalization
    3. approximate Diagonalization (extract only the N_EIGENV lowest EV's)
"""

ham = []
with benchmark("Matrix filling"):
    ham = calc_mhamilton(PARTNBR, SPACEDIMS, BASIS_DIM, BASIS_HO, POT_HO)

sparsimonius = False  # False '=' full matrix diagonalization; True '=' approximate determination of the lowest <N_EIGEN> eigenvalues

if sparsimonius:
    with benchmark("Diagonalization -- sparse matrix structure (DVR)"):
        # calculate the lowest N eigensystem of the matrix in sparse format
        evals_small, evecs_small = eigsh(
            coo_matrix(ham), N_EIGENV, which='SA', maxiter=5000)
        print('Hamilton ( %d X %d ) matrix: %d/%d non-zero entries\n' %
              (np.shape(ham)[0], np.shape(ham)[1], coo_matrix(ham).nnz,
               (BASIS_DIM**(SPACEDIMS * PARTNBR))**2))
        print('DVR-sparse:\n', evals_small[:min(N_EIGENV, np.shape(ham)[1])])
else:
    with benchmark("Diagonalization -- full matrix structure (DVR)"):
        # calculate the eigenvalues of the sum of the Hamilton matrix (Hermitian)
        EV = np.sort(np.linalg.eigvalsh(ham))
        print('Hamilton (%dX%d) matrix: %d/%d non-zero entries\n' %
              (np.shape(ham)[0], np.shape(ham)[1], coo_matrix(ham).nnz,
               (BASIS_DIM**(SPACEDIMS * PARTNBR))**2))
        print('DVR-full:\n', np.real(EV)[:N_EIGENV])

with benchmark("Calculate %d-particle %d-dimensional HO ME analytically" %
               (PARTNBR, SPACEDIMS)):
    nmax = 20
    print(eigenvalues_harmonic_osci(POT_HO[3], nmax,
                                    PARTNBR * SPACEDIMS)[:N_EIGENV])