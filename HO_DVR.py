import numpy as np
import numpy.linalg
from scipy.special import hermite, factorial
import matplotlib.pyplot as plt
from physics import NUCLEON_MASS, PLANCS_CONSTANT, C, JOULE_PER_EV, HBARC, eigenvalues_harmonic_osci

np.set_printoptions(linewidth=300, suppress=True, precision=7)
"""
references are made to Phys. Rept. 324 (2000) 1-105

"""
MASS = NUCLEON_MASS / HBARC  # mass in [fm^-1]
OMEGA = 1.9  # frequency in [fm]
XEQ = 0.  # oscillator origin in [fm]

GRID_DIM = 20
#maxherm  = 40

#randomhermit = np.random.randint(maxherm, size=GRID_DIM)

pos_op_mat = np.zeros((GRID_DIM, GRID_DIM))

# def HO basis functions (Eq.B31), pass arguments in consistent units.
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

hamiltonian_mat = kin_op_mat + pot_op_mat
eigen_sys_new = np.linalg.eigh(hamiltonian_mat)
print(eigen_sys_new[0][:5])
nmax = 20
print(eigenvalues_harmonic_osci(OMEGA, nmax, 1)[:5])
#exit()
# visualize a basis function along with the eigenvalues of the position operator
dimg = 1000
test_grid = np.linspace(-2, 2, dimg)
hermit_order = 20
phi_1 = phi(hermit_order, test_grid, XEQ, MASS, OMEGA)

fig = plt.figure()
ax1 = fig.add_subplot(111)

#m = [[] for m in range(GRID_DIM) ]

#m=[]
#for j in range(GRID_DIM):
#	m+=phi(j, test_grid, XEQ, MASS, OMEGA)*eigen_sys[1][j, 1]

#ax1.plot(test_grid,m)
	
#for alpha in  range(GRID_DIM):
#	tmp = np.zeros(dimg) 
#	for j in range(GRID_DIM):
#		tmp += np.array(phi(j, test_grid, XEQ, MASS, OMEGA))*eigen_sys[1][j, alpha]
#	m[alpha] = tmp

#ax1.plot(test_grid,m[7])

cordi=3
m = np.sum(phi(j, test_grid, XEQ, MASS, OMEGA)*eigen_sys[1][j, cordi] for j in range(GRID_DIM))
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