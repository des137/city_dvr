import numpy as np
import numpy.linalg
from scipy.special import hermite, factorial
import matplotlib.pyplot as plt
from physics import NUCLEON_MASS, PLANCS_CONSTANT, C, JOULE_PER_EV, HBARC, eigenvalues_harmonic_osci
from dvr import calc_grid

np.set_printoptions(linewidth=300, suppress=True, precision=7)
"""
references are made to Phys. Rept. 324 (2000) 1-105

"""

MASS = NUCLEON_MASS / HBARC  # mass in [fm^-1]
OMEGA = 1.  # frequency in [fm]
XEQ = 0.  # oscillator origin in [fm]

GRID_DIM = 8

# def HO basis functions (Eq.B31), pass arguments in consistent units.
def phi(j, x, xeq, mass, omega):
    return 1. / np.sqrt(2**j * factorial(j)) * (
        mass * omega)**0.25 * hermite(j)(np.sqrt(mass * omega) * (
            x - xeq)) * np.exp(-0.5 * mass * omega * (x - xeq)**2)


# calculate U^+ X U , to convince yourself that you understand the syntax
eigen_sys = calc_grid(GRID_DIM, ['HO', [1, 0.0, MASS, OMEGA]])
eigen_pos = eigen_sys[0]

# calculate the matrix elements of the second-derivative operator via Eq. B.36
pot_op_mat = np.zeros((GRID_DIM, GRID_DIM))
kin_op_mat = np.zeros((GRID_DIM, GRID_DIM))
for alpha in range(GRID_DIM):
    for beta in range(GRID_DIM):
        for k in range(GRID_DIM):
            kin_op_mat[alpha, beta] += OMEGA * eigen_sys[1][k, alpha] * (
                k + 0.5) * eigen_sys[1][k, beta]
        if alpha == beta:
            kin_op_mat[alpha, beta] -= 0.5 * MASS * OMEGA**2 * (
                eigen_sys[0][alpha] - XEQ)**2
            # here, a different interaction could be considered; only for this
            # reason, this awkward construct instead of just H = omega \sum_k=1\toN-1 U^*_ka (k+1/2) U_kb is used
            pot_op_mat[alpha, beta] = 0.5 * MASS * OMEGA**2 * (
                eigen_sys[0][alpha] - XEQ)**2

#print(kin_op_mat)
#print(pot_op_mat)
hamiltonian_mat = kin_op_mat + pot_op_mat
eigen_sys_new = np.linalg.eigh(hamiltonian_mat)
print(eigen_sys_new[0][:5])
nmax = 20
print(eigenvalues_harmonic_osci(OMEGA, nmax, 1)[:5])

# visualize a basis function along with the eigenvalues of the position operator
test_grid = np.linspace(-0.2, 0.2, 100)
hermit_order = 6
phi_1 = phi(hermit_order, test_grid, XEQ, MASS, OMEGA)

fig = plt.figure()
ax1 = fig.add_subplot(111)

cordi=4
m = np.sum(phi(j, test_grid, XEQ, MASS, OMEGA)*eigen_sys_new[1][j, cordi] for j in range(GRID_DIM))
ax1.plot(test_grid,m)

ax1.plot(
    test_grid,
    phi_1,
    label=r'$\phi_%d(x)$' % hermit_order,
    linewidth=1,
    color='r',
    ls='-')
ax1.plot(eigen_pos, np.zeros(GRID_DIM), 'bo', label=r'$x_\alpha$', marker='o')
#
ax1.legend(loc='best')
plt.show()

exit()