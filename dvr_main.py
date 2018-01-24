import numpy as np
import matplotlib.pyplot as plt
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
POT_HO_INTER = ['HOINT', 100*OMEGA]

# lattice set-up
PARTNBR = 2  # number of particles
SPACEDIMS = 1  # spatial coordinate dimensions (e.g. Cartesian x,y,z)
BASIS_DIM = 10  # (dim of variational basis) = (nbr of grid points) = (nbr of segments - 1)

# specify the variational basis
BOX_SIZE = 8  #BASIS_DIM + 0  # physical length of one spatial dimension (in Fermi); relevant for specific bases, only!
BOX_ORIGIN = -BOX_SIZE / 2.
BASIS_SINE = ['SINE', [SPACEDIMS, BOX_SIZE, BOX_ORIGIN, MASS / HBARC]]
BASIS_SINC = ['SINC', [SPACEDIMS, BOX_SIZE, BOX_ORIGIN, MASS / HBARC]]
BASIS_HO = ['HO', [SPACEDIMS, X_EQ, (MASS / HBARC), OMEGA]]
# each axis is devided into = (number of grid points) - 1

N_EIGENV = 02  # number of eigenvalues to be calculated with <eigsh>


def calc_mhamilton(n_part, dim_space, dim_bas, spec_bas, spec_pot):
    """ Function returns the Hamilton matrix; 

        :n_part: number of particles
        :dim_space: spatial (Cartesian) dimensions
        :dim_bas: variational-basis dim = number of segments each coordinate is divided into
        :spec_pot: parameters specifying the interaction potential
        :spec_bas: parameters specifying the basis

        :return: full Hamilton matrix in D(iscrete) V(ariable) R(epresentation)
    """
    # dimension of a single coordinate point; e.g., 2D 2Part: (x1,y1,x2,y2)
    # D spatial dimensions for each of the N particles;
    dim_grdpoint = n_part * dim_space
    # each component of a grid point takes dim_bas discrete values
    # e.g. x1 \in {x_1,...,x_dim_bas} where x_1 is an eigenvalue of the position matrix
    dim_h = dim_bas**dim_grdpoint

    # initialize empty matrices (might have to be "sparsed" for larger dim. problems)
    mpotential = np.zeros((dim_h, dim_h))
    mkinetic = np.zeros((dim_h, dim_h))
    mhamilton = np.zeros((dim_h, dim_h))

    # obtain eigensystem of the position operator in the basis of choice
    # eigenvalues '=' grid points ; transformation matrix necessary for Ekin
    # STATUS: one basis for all coordinates (future: xy->HO, z->SINE)
    coordOP_evSYS = calc_grid(dim_bas, spec_bas)

    # calculate potential and kinetic-energy matrices for a chosen basis
    # STATUS: for each additional basis, the matrices need to be specified in this function!
    mkinetic = calc_Ekin(dim_bas, n_part, spec_bas, coordOP_evSYS)
    # STATUS: the potential matrix is assumed to be diagonal (future: OPE+B => potential has non-zero offdiagonal elements)
    mpotential = np.diag(
        calc_potential(n_part, dim_space, spec_pot, coordOP_evSYS))

    mhamilton = (mkinetic + mpotential)
    return mhamilton


""" main section of the program
    1. set up the Hamiltonian
    2. full Diagonalization
    3. approximate Diagonalization (extract only the N_EIGENV lowest EV's)
"""

ham = []
with benchmark("Matrix filling"):
    ham = calc_mhamilton(PARTNBR, SPACEDIMS, BASIS_DIM, BASIS_SINC,
                         POT_HO_INTER)

sparsimonius = True  # False '=' full matrix diagonalization; True '=' approximate determination of the lowest <N_EIGEN> eigenvalues

if sparsimonius:
    with benchmark("Diagonalization -- sparse matrix structure (DVR)"):
        # calculate the lowest N eigensystem of the matrix in sparse format
        evals_small, evecs_small = eigsh(
            coo_matrix(ham), N_EIGENV, which='SA', maxiter=5000)
        print(
            'Hamilton ( %d X %d ) matrix: %d/%d = %3.2f%% non-zero entries\n' %
            (np.shape(ham)[0], np.shape(ham)[1], coo_matrix(ham).nnz,
             (BASIS_DIM**
              (SPACEDIMS * PARTNBR))**2, 100. * coo_matrix(ham).nnz / float(
                  (BASIS_DIM**(SPACEDIMS * PARTNBR))**2)))
#        print('DVR-sparse:\n', evals_small[:min(N_EIGENV, np.shape(ham)[1])])
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

#plt.plot(evals_small[:min(N_EIGENV, np.shape(ham)[1])])
plt.plot(eigenvalues_harmonic_osci(POT_HO[3], N_EIGENV,
                                    PARTNBR * SPACEDIMS)[:N_EIGENV])
plt.show()

exit()    