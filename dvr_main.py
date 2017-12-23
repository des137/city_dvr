import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from dvr import calc_grid, calc_Ekin, calc_potential, calc_mhamilton
from physics import NUCLEON_MASS, PLANCS_CONSTANT, C, JOULE_PER_EV, HBARC, eigenvalues_harmonic_osci
from util import benchmark

np.set_printoptions(linewidth=300, suppress=True, precision=5)

# single-particle parameters and other physical constants
MASS = NUCLEON_MASS

# isotropic harmonic oscillator
OMEGA_BASIS = .2
OMEGA_TRAP = 0.5
LEC_HOINT = 100.
X_EQ = 0.
POT_HO = ['HO', X_EQ, MASS / HBARC, OMEGA_TRAP]
POT_HOINT = ['HOINT', LEC_HOINT]

# lattice set-up
PARTNBR = 2  # number of particles
SPACEDIMS = 3  # spatial coordinate dimensions (e.g. Cartesian x,y,z)
BASIS_DIM = 4  # (dim of variational basis) = (nbr of grid points) = (nbr of segments - 1)

# specify the variational basis
BOX_SIZE = 10  #BASIS_DIM + 0  # physical length of one spatial dimension (in Fermi); relevant for specific bases, only!
BOX_ORIGIN = -BOX_SIZE / 2.
BASIS_SINE = ['SINE', [SPACEDIMS, BOX_SIZE, BOX_ORIGIN, MASS / HBARC]]
BASIS_HO = ['HO', [SPACEDIMS, X_EQ, (MASS / HBARC), OMEGA_BASIS]]
# each axis is devided into = (number of grid points) - 1

N_EIGENV = 10  # number of eigenvalues to be calculated with <eigsh>
""" main section of the program
    1. set up the Hamiltonian
    2. full Diagonalization
    3. approximate Diagonalization (extract only the N_EIGENV lowest EV's)
"""

ham = []
with benchmark("Matrix filling"):
    evs = []
    for cycl in range(2):
        LEC = [100, 1000]
        POT_HOINT = ['HOINT', LEC[cycl]]
        ham = calc_mhamilton(PARTNBR, SPACEDIMS, BASIS_DIM, BASIS_HO,
                             POT_HOINT)
        evals_small, evecs_small = eigsh(
            coo_matrix(ham), N_EIGENV, which='SA', maxiter=50000)
        evs.append(evals_small[:N_EIGENV])
    fig = plt.figure()
    ax1 = fig.add_subplot(211)

    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$E_n$ [MeV]')
    [plt.plot(evs[n], label=r'n=%d' % n) for n in range(len(evs))]
    ax1.legend(loc='lower right')
    ax1 = fig.add_subplot(212)

    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$E_n(C_{1000})-E_n(C_{100})$ [MeV]')
    [
        plt.plot(evs[n + 1] - evs[n], label=r'$E(C_%d)-E(C_%d)$' % (n + 1, n))
        for n in range(2)
    ]
    ax1.legend(loc='lower right')
    plt.show()
    exit()

sparsimonius = True  # False '=' full matrix diagonalization; True '=' approximate determination of the lowest <N_EIGEN> eigenvalues

if sparsimonius:
    with benchmark("Diagonalization -- sparse matrix structure (DVR)"):
        # calculate the lowest N eigensystem of the matrix in sparse format
        evals_small, evecs_small = eigsh(
            coo_matrix(ham), N_EIGENV, which='SA', maxiter=5000)
        for e in evals_small[:N_EIGENV]:
            print('%8.8f,' % e, end='')
        exit()
        print('Hamilton ( %d X %d ) matrix: %d/%d = %3.2f%% non-zero entries' %
              (np.shape(ham)[0], np.shape(ham)[1], coo_matrix(ham).nnz,
               (BASIS_DIM**
                (SPACEDIMS * PARTNBR))**2, 100. * coo_matrix(ham).nnz / float(
                    (BASIS_DIM**(SPACEDIMS * PARTNBR))**2)))
        print('DVR-sparse:\nE_n         = ',
              evals_small[:min(N_EIGENV,
                               np.shape(ham)[1])])
        print('E_n+1 - E_n = ',
              np.diff(evals_small[:min(N_EIGENV,
                                       np.shape(ham)[1])]))
else:
    with benchmark("Diagonalization -- full matrix structure (DVR)"):
        # calculate the eigenvalues of the sum of the Hamilton matrix (Hermitian)
        EV = np.sort(np.linalg.eigvalsh(ham))
        for e in EV[:N_EIGENV]:
            print('%8.8f,' % e, end='')
        exit()
        print('Hamilton (%dX%d) matrix: %d/%d = %3.2f%% non-zero entries\n' %
              (np.shape(ham)[0], np.shape(ham)[1], coo_matrix(ham).nnz,
               (BASIS_DIM**
                (SPACEDIMS * PARTNBR))**2, 100. * coo_matrix(ham).nnz / float(
                    (BASIS_DIM**(SPACEDIMS * PARTNBR))**2)))
        print('DVR-full:\nE_n         = ', np.real(EV)[:N_EIGENV])
        print('E_n+1 - E_n = ', np.diff(EV[:min(N_EIGENV, np.shape(ham)[1])]))

with benchmark("Calculate %d-particle %d-dimensional HO ME analytically" %
               (PARTNBR, SPACEDIMS)):
    nmax = 20
    # E_n = (n_1x + ... + n_Nz + (D+N)/2) * hbarc * omega
    # N non-interacting particles in HO trap:
    #DIM = PARTNBR * SPACEDIMS
    #OM = OMEGA_TRAP
    # 2-particle system with relative HO interaction:
    DIM = SPACEDIMS
    OM = 2 * np.sqrt(LEC_HOINT / MASS)
    anly_ev = eigenvalues_harmonic_osci(OM, nmax, DIM)[:max(N_EIGENV, 10)]
    print('E_n         = ', anly_ev)
    print('E_n+1 - E_n = ', np.diff(anly_ev))