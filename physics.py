from itertools import product
import numpy as np

NEUTRON_MASS = 939.5654133  # mev/c^2
PROTON_MASS = 938.2720813  # mev/c^2
NUCLEON_MASS = 0.5 * (NEUTRON_MASS + PROTON_MASS)  # mev/c^2
PLANCS_CONSTANT = 6.62607004 * 10e-34  # m^2 kg / s
C = 299792458  # m/s
JOULE_PER_EV = 1.0 / 1.602176565e-19

#       h*c/(2*pi) in MeV*fm
HBARC = PLANCS_CONSTANT * C / (2 * np.pi) * JOULE_PER_EV * 10e12 * 10e-6


def eigenvalues_harmonic_osci(omega, nmax, dim):
    """ Returns energy eigenvalues of the dim-dimensional quantum-mechanical harmonic oscillator
		H = -hbarc^2/(2m)d^2/dx^2 + 1/2*m*omega^2 * x^2

	E_N = (nx + ny + nz + 3/2) * omega

	:omega: oscillator frequency
	:nmax: maximal oscillator quantum considered for each Cartesian dimension
	:dim: Cartesian dimensions

	:return: sorted eigenvalue array with [ E_(n1n2n3) < ... < E_(nmaxnmaxnmax)] in units of omega
	"""
    return np.sort(
        np.array([(sum(n) + dim / 2.) * omega
                  for n in product(np.arange(nmax), repeat=dim)]))
