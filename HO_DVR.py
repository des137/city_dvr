import numpy as np
import numpy.linalg
from scipy.special import hermite, factorial
import matplotlib.pyplot as plt
from physics import NUCLEON_MASS, PLANCS_CONSTANT, C, JOULE_PER_EV, HBARC, eigenvalues_harmonic_osci

np.set_printoptions(linewidth=300, suppress=True, precision=7)
"""
references are made to Phys. Rept. 324 (2000) 1-105

"""

MASS = NUCLEON_MASS
OMEGA = 2.
XEQ = 0.

GRID_DIM = 4
pos_op_mat = np.zeros((GRID_DIM, GRID_DIM))


# def HO basis functions (Eq.B31)
def phi(j, x, xeq, mass, omega):
    return 1. / np.sqrt(2**j * factorial(j)) * (
        mass * omega)**0.25 * hermite(j)(np.sqrt(mass * omega) * (
            x - xeq)) * np.exp(-0.5 * mass * omega * (x - xeq)**2)


# populate position-operator matrix in HO-basis representation (Eq. B.32)
for j in range(GRID_DIM):
    for k in range(GRID_DIM):
        if j == k:
            pos_op_mat[j, k] = XEQ
        if j == k - 1:
            pos_op_mat[j, k] = np.sqrt((j + 1) / (2 * MASS * OMEGA))
        if j == k + 1:
            pos_op_mat[j, k] = np.sqrt(j / (2 * MASS * OMEGA))

# calculate U^+ X U , to convince yourself that you understand the syntax
eigen_sys = np.linalg.eigh(pos_op_mat)
eigen_pos = eigen_sys[0]
print(np.dot(np.transpose(eigen_sys[1]), np.dot(pos_op_mat, eigen_sys[1])))

# calculate the matrix elements of the second-derivative operator via Eq. B.36
pot_op_mat = np.zeros((GRID_DIM, GRID_DIM))
kin_op_mat = np.zeros((GRID_DIM, GRID_DIM))
for alpha in range(GRID_DIM):
    for beta in range(GRID_DIM):
        for k in range(GRID_DIM):
            if alpha == beta:
                pot_op_mat[alpha, beta] = (eigen_sys[0][alpha] - XEQ)**2

            kin_op_mat[alpha, beta] += eigen_sys[1][k, alpha] * (
                k + 0.5) * eigen_sys[1][k, beta]
pot_op_mat *= (MASS * OMEGA)**2
kin_op_mat *= (-2 * MASS * OMEGA)
kin_op_mat = kin_op_mat + pot_op_mat
pot_op_mat *= 1. / (2 * MASS)
hamiltonian_mat = kin_op_mat + pot_op_mat
eigen_sys = np.linalg.eigh(hamiltonian_mat)
print(eigen_sys[0])
nmax = 20
print(eigenvalues_harmonic_osci(MASS * OMEGA**2, MASS, nmax)[:5])
exit()
# visualize a basis function along with the eigenvalues of the position operator
test_grid = np.linspace(-0.2, 0.2, 100)
hermit_order = 6
phi_1 = phi(hermit_order, test_grid, XEQ, MASS, OMEGA)

fig = plt.figure()
ax1 = fig.add_subplot(111)

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