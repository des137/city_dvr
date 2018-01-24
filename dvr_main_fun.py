import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from dvr import calc_grid, calc_Ekin, calc_potential
from physics import NUCLEON_MASS, PLANCS_CONSTANT, C, JOULE_PER_EV, HBARC, eigenvalues_harmonic_osci
from util import benchmark
MASS = NUCLEON_MASS
LEC = -5.1
BETA = 4.0
POT_GAUSS = ['GAUSS', BETA, LEC]
OMEGA = 1.
X_EQ = 0.
POT_HO = ['HO', X_EQ, MASS / HBARC, OMEGA]
POT_HO_INTER = ['HOINT', 100*OMEGA]
PARTNBR = 2
SPACEDIMS = 2
BASIS_DIM = 5
BOX_SIZE = 8
BOX_ORIGIN = -BOX_SIZE / 2.
BASIS_SINE = ['SINE', [SPACEDIMS, BOX_SIZE, BOX_ORIGIN, MASS / HBARC]]
BASIS_HO = ['HO', [SPACEDIMS, X_EQ, (MASS / HBARC), OMEGA]]
N_EIGENV = 400
def calc_mhamilton(n_part, dim_space, dim_bas, spec_bas, spec_pot):
    dim_grdpoint = n_part * dim_space
    dim_h = dim_bas**dim_grdpoint
    mpotential = np.zeros((dim_h, dim_h))
    mkinetic = np.zeros((dim_h, dim_h))
    mhamilton = np.zeros((dim_h, dim_h))
    coordOP_evSYS = calc_grid(dim_bas, spec_bas)
    mkinetic = calc_Ekin(dim_bas, n_part, spec_bas, coordOP_evSYS)
    mpotential = np.diag(calc_potential(n_part, dim_space, spec_pot, coordOP_evSYS))
    mhamilton = (mkinetic + mpotential)
    return mhamilton
ham = []
with benchmark("Matrix filling"):
    ham = calc_mhamilton(PARTNBR, SPACEDIMS, BASIS_DIM, BASIS_HO,POT_HO_INTER)
sparsimonius = True
if sparsimonius:
    with benchmark("Diagonalization -- sparse matrix structure (DVR)"):
        evals_small, evecs_small = eigsh(
            coo_matrix(ham), N_EIGENV, which='SA', maxiter=5000)
        print('Hamilton ( %d X %d ) matrix: %d/%d = %3.2f%% non-zero entries\n' %
            (np.shape(ham)[0], np.shape(ham)[1], coo_matrix(ham).nnz,
             (BASIS_DIM** (SPACEDIMS * PARTNBR))**2, 100. * coo_matrix(ham).nnz / float((BASIS_DIM**(SPACEDIMS * PARTNBR))**2)))
else:
    with benchmark("Diagonalization -- full matrix structure (DVR)"):
        EV = np.sort(np.linalg.eigvalsh(ham))
        print('Hamilton (%dX%d) matrix: %d/%d non-zero entries\n' %
              (np.shape(ham)[0], np.shape(ham)[1], coo_matrix(ham).nnz,
               (BASIS_DIM**(SPACEDIMS * PARTNBR))**2))
        print('DVR-full:\n', np.real(EV)[:N_EIGENV])

o=evals_small[:min(N_EIGENV, np.shape(ham)[1])]
exit()